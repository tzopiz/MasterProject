#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME
python3 -c 'import sys; assert sys.version_info[:2] == (3, 10), "unexpected base Python"'
# --without-pip works even when the system Python does not ship ensurepip.
python3 -m venv --without-pip .tmj-runtime
python3 -m pip --python .tmj-runtime install --disable-pip-version-check -r payload/requirements.txt
.tmj-runtime/bin/python -c 'import json, platform, torch, numpy, scipy, sklearn; print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}), flush=True); assert torch.cuda.is_available(), "CUDA unavailable"'
exec .tmj-runtime/bin/python payload/tools/datasphere_osseous.py worker --config "$1" --max-runtime-seconds "$2"
