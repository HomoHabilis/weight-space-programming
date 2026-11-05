#!/usr/bin/env bash
# Download the authors' published logs, networks, prompts and answer keys (~370 MB) into data/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ -d data/lm_understand-logs/.git ]; then
  git -C data/lm_understand-logs pull --ff-only
else
  git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs data/lm_understand-logs
fi
echo "Logs in data/lm_understand-logs"
