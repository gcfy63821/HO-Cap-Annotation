#!/usr/bin/env python3
"""SQLite task store for the volunteer annotation cloud server.

A task = one EXPERIMENT (all 8 synchronized camera views of one sample). The
volunteer annotates each role across every view in one sitting, then submits
once. Status lifecycle:
    unassigned -> locked -> submitted -> approved | rejected
Locks expire (locked_until) so abandoned tasks return to the pool.
"""

import json
import re
import sqlite3
import time
from pathlib import Path

LOCK_TTL_SEC = 30 * 60  # a locked task returns to the pool after 30 min idle

# Fixed semantic roles, in the order that determines masks.h5 object ids.
# (object_id is assigned contiguously by prompts_to_masks.py per exp; primary,
# when present, is always id 1 — the object the hand-object optimization uses.)
ROLES = [
    ("primary_tool", "主操作工具"),
    ("auxiliary_tool", "辅助工具"),
    ("manipulated_object", "操作对象"),
]


def fallback_primary_name(exp_name):
    """Used only if the manifest lacks a primary_name: strip a trailing _<num>.
    (The authoritative name is computed on the GPU side via mesh_name_mapping.json
    and written into the manifest by precompute_embeddings.py.)"""
    return re.sub(r"_\d+$", "", exp_name)


def derive_roster(primary_name, aux_name="auxiliary_tool", obj_name="manipulated_object"):
    """The 3 role slots a volunteer sees. Primary name comes from the manifest
    (mesh_name_mapping.json); auxiliary / manipulated-object are generic names
    (only the primary tool is FoundationPose-tracked downstream)."""
    names = {"primary_tool": primary_name,
             "auxiliary_tool": aux_name, "manipulated_object": obj_name}
    return [{"role": r, "label": lbl, "name": names[r]} for r, lbl in ROLES]

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task         TEXT NOT NULL,
    exp          TEXT NOT NULL,
    cameras_json TEXT NOT NULL,
    roles_json   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'unassigned',
    annotator_id TEXT,
    locked_until REAL DEFAULT 0,
    preview_iou  REAL,
    submitted_at TEXT,
    note         TEXT,
    UNIQUE(task, exp)
);
"""


def connect(db_path):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    try:  # migrate older DBs that predate the note column
        conn.execute("ALTER TABLE tasks ADD COLUMN note TEXT")
    except sqlite3.OperationalError:
        pass
    return conn


def seed_from_manifest(conn, manifest_path):
    """Insert one task per EXPERIMENT (grouping all its cameras). Idempotent on
    (task, exp). Reads manifest schema 2 (per-camera keyframes/thumbs)."""
    manifest = json.loads(Path(manifest_path).read_text())
    exps = {}  # (task, exp) -> {cameras: [...], primary_name}
    for cam in manifest["cameras"]:
        key = (cam["task"], cam["exp"])
        e = exps.setdefault(key, {"cameras": [], "primary_name": cam.get("primary_name")})
        e["cameras"].append({
            "camera": cam["camera"], "cam_index": cam["cam_index"],
            "width": cam["width"], "height": cam["height"],
            "n_frames": cam.get("n_frames"),
            "keyframes": cam.get("keyframes", [0]), "thumbs": cam.get("thumbs", []),
        })
    n_new = 0
    for (task, exp), e in exps.items():
        cams = sorted(e["cameras"], key=lambda c: c["cam_index"])
        roster = derive_roster(e["primary_name"] or fallback_primary_name(exp))
        try:
            conn.execute(
                "INSERT INTO tasks (task, exp, cameras_json, roles_json) VALUES (?,?,?,?)",
                (task, exp, json.dumps(cams), json.dumps(roster, ensure_ascii=False)),
            )
            n_new += 1
        except sqlite3.IntegrityError:
            pass  # already seeded
    conn.commit()
    return n_new


def next_task(conn, annotator_id):
    """Atomically lock and return the next available task, or None."""
    now = time.time()
    cur = conn.execute(
        "SELECT * FROM tasks WHERE status='unassigned' "
        "   OR (status='locked' AND locked_until < ?) "
        "ORDER BY id LIMIT 1", (now,))
    row = cur.fetchone()
    if row is None:
        return None
    conn.execute(
        "UPDATE tasks SET status='locked', annotator_id=?, locked_until=? WHERE id=?",
        (annotator_id, now + LOCK_TTL_SEC, row["id"]))
    conn.commit()
    return dict(row)


def get_task(conn, task_id):
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return dict(row) if row else None


def open_task(conn, task_id, annotator_id):
    """Load a SPECIFIC task (for video switching). Lock it if it's free;
    a submitted/bad/other-locked task is still returned so it can be reviewed
    or re-annotated."""
    row = get_task(conn, task_id)
    if row is None:
        return None
    now = time.time()
    if row["status"] == "unassigned" or (row["status"] == "locked" and row["locked_until"] < now):
        conn.execute("UPDATE tasks SET status='locked', annotator_id=?, locked_until=? WHERE id=?",
                     (annotator_id, now + LOCK_TTL_SEC, task_id))
        conn.commit()
        row = get_task(conn, task_id)
    return row


def list_tasks(conn, annotator_id):
    rows = conn.execute("SELECT id, exp, status, annotator_id FROM tasks ORDER BY id").fetchall()
    return [{"id": r["id"], "exp": r["exp"], "status": r["status"],
             "mine": r["annotator_id"] == annotator_id} for r in rows]


def progress(conn, annotator_id):
    rows = conn.execute("SELECT status, annotator_id FROM tasks").fetchall()
    total = len(rows)
    submitted = sum(r["status"] == "submitted" for r in rows)
    bad = sum(r["status"] == "bad" for r in rows)
    mine = sum(r["status"] == "submitted" and r["annotator_id"] == annotator_id for r in rows)
    return {"total": total, "submitted": submitted, "bad": bad, "mine": mine,
            "remaining": total - submitted - bad}


def submit_task(conn, task_id, annotator_id, preview_iou):
    conn.execute(
        "UPDATE tasks SET status='submitted', annotator_id=?, preview_iou=?, submitted_at=? "
        "WHERE id=?",
        (annotator_id, preview_iou, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), task_id))
    conn.commit()


def flag_bad(conn, task_id, annotator_id, reason):
    """Mark a whole exp as a bad/unusable video so it's never re-served."""
    conn.execute(
        "UPDATE tasks SET status='bad', annotator_id=?, note=?, submitted_at=? WHERE id=?",
        (annotator_id, reason, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), task_id))
    conn.commit()


def stats(conn):
    rows = conn.execute("SELECT status, COUNT(*) c FROM tasks GROUP BY status").fetchall()
    return {r["status"]: r["c"] for r in rows}
