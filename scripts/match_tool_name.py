#!/usr/bin/env python
"""
Match a model folder name against task/experiment folder names.

Heuristic (roughly "first-token-of-task + context from exp"):

  1. Build a normalized lowercase string from the combined task+exp names,
     dropping underscores and digits:
       'mallet_crush_dough_20260104_bigwoodenfork_..._1'
       -> 'malletcrushdoughbigwoodenfork...'
  2. For each candidate model name (subdir of --models_folder), tokenize its
     name on '_' and check which tokens appear as substrings of the normalized
     combined string.
  3. Rank candidates by (#matched_tokens, longest_matched_token_len, model_name_len).
  4. If a task_name first-token is given, require the matched model to include
     that token — models that don't contain it are filtered out. This forces
     the tool *category* to be correct (e.g., 'fork' task only matches
     fork-shaped models).

Prints the best model name to stdout, or exits non-zero if no match.

Usage:
  python scripts/match_tool_name.py \
    --models_folder /.../HO-Cap-Annotation/data/models \
    --task_name mallet_crush_dough \
    --exp_name  20260104_largeplate_mallet_flatten_crush_largedough_1
"""
import argparse
import os
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def tokenize(s):
    """Lowercase + split on underscore / hyphen / digit boundaries, drop empties."""
    return [t for t in re.split(r'[_\-\d]+', s.lower()) if t]


def normalize(s):
    """Lowercase, drop all underscores/hyphens/digits for fuzzy substring match."""
    return re.sub(r'[_\-\d]+', '', s.lower())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models_folder', required=True)
    ap.add_argument('--task_name', required=True)
    ap.add_argument('--exp_name', default='')
    ap.add_argument('--require_category', action='store_true', default=True,
                    help='Require matched model to contain the task\'s first token')
    ap.add_argument('--no_require_category', dest='require_category',
                    action='store_false')
    ap.add_argument('--mapping_yaml', default=None,
                    help='Optional YAML with a top-level `mappings:` dict of '
                         'keyword->tool_name (produced by '
                         'scripts/data_inspector_viser.py). If a keyword '
                         '(normalized) appears as a substring of the '
                         'normalized exp_name, that tool wins — bypassing '
                         'the fuzzy auto-matcher entirely.')
    ap.add_argument('--require_mapping', action='store_true',
                    help='Only return a tool if --mapping_yaml has a keyword '
                         'matching exp_name. If no keyword matches, exit '
                         'with a dedicated non-zero code (3) instead of '
                         'falling back to the fuzzy auto-matcher. Useful '
                         'for "only annotate what I have explicitly mapped".')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    # ---- keyword mapping override (manual, from data_inspector_viser.py) ----
    if args.mapping_yaml:
        mp = Path(args.mapping_yaml)
        if mp.exists() and yaml is not None:
            try:
                data = yaml.safe_load(mp.read_text()) or {}
                mapping = dict(data.get('mappings', {}) or {})
            except Exception as e:
                print(f'[match] WARN: could not read {mp}: {e}', file=sys.stderr)
                mapping = {}
            exp_norm = normalize(args.exp_name)
            for kw, tool in mapping.items():
                if not kw or not tool:
                    continue
                if normalize(str(kw)) in exp_norm:
                    if args.verbose:
                        print(f'[match] mapping HIT: keyword="{kw}" -> tool="{tool}" '
                              f'(source: {mp.name})', file=sys.stderr)
                    print(tool)
                    return
            if args.require_mapping:
                print(f'[match] require_mapping set but no keyword matched '
                      f'exp="{args.exp_name}" in {mp.name} '
                      f'({len(mapping)} keyword(s)); skipping',
                      file=sys.stderr)
                sys.exit(3)
            if args.verbose and mapping:
                print(f'[match] mapping file had {len(mapping)} keyword(s) but '
                      f'none matched exp; falling back to auto-match',
                      file=sys.stderr)
        elif args.require_mapping:
            print(f'[match] require_mapping set but mapping file missing: {mp}',
                  file=sys.stderr)
            sys.exit(3)

    models_dir = Path(args.models_folder)
    if not models_dir.is_dir():
        print(f'ERROR: models folder not found: {models_dir}', file=sys.stderr)
        sys.exit(2)

    # Candidate model names: subdirs, exclude hidden / glob-artifact "*"
    candidates = [d.name for d in models_dir.iterdir()
                   if d.is_dir() and not d.name.startswith('.') and d.name != '*']
    if not candidates:
        print(f'ERROR: no models in {models_dir}', file=sys.stderr)
        sys.exit(2)

    task_tokens = tokenize(args.task_name)
    category = task_tokens[0] if task_tokens else None
    task_norm = normalize(args.task_name)
    exp_norm = normalize(args.exp_name)
    combined_norm = task_norm + exp_norm

    # New scoring scheme (no hard category filter):
    #   (exp_matches, all_matches, category_hit, longest_matched_token, -name_len)
    # The exp name describes the ACTUAL tool instance (colour, material, etc.),
    # so exp_matches is the primary signal. `category_hit` (task's first token
    # appearing among the model's tokens) is only a tiebreaker — it will NOT
    # override a stronger exp-match signal, which fixes cases like
    # task="spatula_spread_tomatosauce" + exp="...bluescooper..." -> blue_scooper
    # even though the task's category ("spatula") is absent from the models.
    scored = []
    for m in candidates:
        m_tokens = tokenize(m)
        matched_all = [t for t in m_tokens if t and t in combined_norm]
        if not matched_all:
            continue
        matched_exp = [t for t in m_tokens if t and t in exp_norm]
        category_hit = 0
        if category and any(category in t or t in category for t in m_tokens):
            category_hit = 1
        score = (
            len(matched_exp),                   # PRIMARY: exp-specific hits
            len(matched_all),                   # total hits
            category_hit,                        # prefer category match on ties
            max(len(t) for t in matched_all),   # longest matched token
            -len(m),                            # shorter model name = canonical
        )
        scored.append((score, m, matched_all))
    fallback = False

    if not scored:
        print(f'ERROR: no model matched task="{args.task_name}" exp="{args.exp_name}" '
              f'(category="{category}", searched {len(candidates)} models)',
              file=sys.stderr)
        sys.exit(1)

    scored.sort(key=lambda x: x[0], reverse=True)
    if fallback:
        print(f'[match] WARN: category "{category}" not in any model; '
              f'fell back to exp-name token match', file=sys.stderr)
    if args.verbose:
        print(f'[match] task="{args.task_name}" exp="{args.exp_name}"', file=sys.stderr)
        print(f'[match] category="{category}"  normalized="{combined_norm[:80]}..."',
              file=sys.stderr)
        for score, m, matched in scored[:5]:
            print(f'[match]   score={score}  {m}  matched_tokens={matched}',
                  file=sys.stderr)
    print(scored[0][1])


if __name__ == '__main__':
    main()
