#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# CUDA and data/cache env must be set before the Python process starts.
# python-dotenv can parse .env key/value pairs, but it cannot execute `source`.
source "${repo_root}/scripts/vlmeval_env.sh"
source "${repo_root}/scripts/seed_judge_env.sh" "${SEED_ENV_FILE:-${repo_root}/.seed.env}"

exec python "${repo_root}/run.py" "$@"
