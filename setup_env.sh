#!/usr/bin/env bash
# Creates .venv in the repository root from scratch: Python 3.10, torch 2.7.1 (CUDA 12.6 build), mujoco and
# mujoco-warp 3.8.1, warp-lang 1.14.0, numpy 1.26.4 (the stack the results in README.md were trained with).
# Needs only an NVIDIA driver (nvidia-smi shows CUDA >= 12.6) and gcc; no system CUDA toolkit.
# Uses uv when it is installed (uv also downloads Python 3.10 if the system has none), otherwise
# $PYTHON -m venv + pip.
#
#   bash setup_env.sh
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash setup_env.sh   # PyPI mirror
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash setup_env.sh       # other CUDA build of torch 2.7.1
#   PYTHON=/usr/bin/python3.10 bash setup_env.sh                               # use this interpreter
set -euo pipefail
cd "$(dirname "$0")"

PY_VERSION=${PY_VERSION:-3.10}
INDEX_ARGS=()
if [ -n "${PIP_INDEX_URL:-}" ]; then INDEX_ARGS=(--index-url "$PIP_INDEX_URL"); fi

if command -v uv > /dev/null; then
  [ -x .venv/bin/python ] || uv venv --python "${PYTHON:-$PY_VERSION}" --prompt supermimic_a3 .venv
  install() { uv pip install --python .venv/bin/python "$@"; }
else
  if [ ! -x .venv/bin/python ]; then
    PYTHON=${PYTHON:-$(command -v "python$PY_VERSION" || command -v python3)}
    "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10 (or install uv)"'
    "$PYTHON" -m venv --prompt supermimic_a3 .venv
  fi
  .venv/bin/python -m pip install --progress-bar off --upgrade pip ${INDEX_ARGS[@]+"${INDEX_ARGS[@]}"}
  install() { .venv/bin/python -m pip install --progress-bar off "$@"; }
fi

# On Linux x86_64 the PyPI torch 2.7.1 wheel is the CUDA 12.6 build, so a PyPI mirror is enough. TORCH_INDEX
# installs it from a PyTorch wheel index instead (e.g. cu128 for RTX 50xx / B200, which cu126 does not support).
if [ -n "${TORCH_INDEX:-}" ]; then
  install torch==2.7.1 --index-url "$TORCH_INDEX"
fi
install -r requirements.txt ${INDEX_ARGS[@]+"${INDEX_ARGS[@]}"}

# Smoke test: versions, GPU visible to torch and warp, torch.compile works (needs gcc and Python headers).
.venv/bin/python - << 'EOF'
import sys
from importlib.metadata import version

import torch
import warp as wp


def ver(dist):
  try:
    return version(dist)
  except Exception:
    return "?"


print("python", sys.version.split()[0], "| torch", torch.__version__, "(CUDA", str(torch.version.cuda) + ")",
      "| mujoco", ver("mujoco"), "| mujoco-warp", ver("mujoco-warp"), "| warp", ver("warp-lang"), "| numpy", ver("numpy"))
assert torch.cuda.is_available(), "torch sees no GPU: check nvidia-smi and CUDA_VISIBLE_DEVICES"
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
wp.init()
assert wp.get_cuda_device_count() > 0, "warp sees no CUDA device"
f = torch.compile(lambda x: torch.sin(x) * 2 + 1)
assert torch.allclose(f(torch.ones(8, device="cuda")), torch.sin(torch.ones(8, device="cuda")) * 2 + 1)
print("setup ok:", sys.prefix)
EOF
