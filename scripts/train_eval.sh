#!/usr/bin/env bash
# Trains, then evaluates every saved checkpoint with the Appendix F.1 protocol (supermimic.evaluate) and prints
# a table of the results. Extra arguments go to supermimic.train.
#   bash scripts/train_eval.sh runs/smoke --preset smoke
#   GPU=1 bash scripts/train_eval.sh runs/racket_w1 --preset racket
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: scripts/train_eval.sh <out_dir> [train args...]}
shift
export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
PY=${PY:-.venv/bin/python}
mkdir -p "$OUT"

$PY -m supermimic.train --out_dir "$OUT" "$@" 2>&1 | tee "$OUT/train.log"
for ck in $(ls "$OUT"/model_*.pt | sort -V); do
  echo "[eval] $ck"
  $PY -m supermimic.evaluate --ckpt "$ck" > /dev/null 2>> "$OUT/eval.log" || echo "eval failed: $ck" | tee -a "$OUT/eval.log"
done

$PY - "$OUT" << 'EOF'
import glob, json, os, sys
rows = sorted((json.load(open(f)) for f in glob.glob(os.path.join(sys.argv[1], "model_*_eval.json"))),
              key=lambda r: r["iter"])
print(f"{'iter':>7} {'failed':>8} {'norm_return':>12} {'body_err_cm':>12} {'racket_hit_err_cm':>18}")
for r in rows:
  print(f"{r['iter']:>7} {r['failed']:>4}/{r['motions']:<3} {r['norm_return_mean']:>12.3f} "
        f"{100 * r['mean_body_pos_err_m']:>12.2f} {100 * r['racket_hit_err_m']:>18.2f}")
EOF
