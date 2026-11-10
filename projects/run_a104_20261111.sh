#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-/home/lzy/anaconda3/bin/python}" -u utils/a104_session.py "$@"
