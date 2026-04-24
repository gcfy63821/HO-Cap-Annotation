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
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

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
    combined_norm = normalize(f'{args.task_name}_{args.exp_name}')

    def _score_all(with_category_filter):
        out = []
        for m in candidates:
            m_tokens = tokenize(m)
            matched = [t for t in m_tokens if t and t in combined_norm]
            if not matched:
                continue
            if with_category_filter and category:
                if not any(category in t or t in category for t in m_tokens):
                    continue
            score = (
                len(matched),                       # more matched tokens = better
                max(len(t) for t in matched),       # longer matched token = better
                -len(m),                            # shorter model name = better (canonical)
            )
            out.append((score, m, matched))
        return out

    # First pass: require the task's category token (e.g. "mallet" for "mallet_*")
    scored = _score_all(with_category_filter=True) if args.require_category else []
    fallback = False
    if not scored:
        # Fallback: allow any model whose tokens appear in the combined string
        scored = _score_all(with_category_filter=False)
        fallback = bool(scored)

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
