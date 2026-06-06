#!/usr/bin/env bash
# Source this file before running run.py to use an Ark/Seed model as VLMEvalKit's judge.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source scripts/seed_judge_env.sh [path/to/.seed.env]" >&2
  exit 2
fi

seed_env_file="${1:-${SEED_ENV_FILE:-/root/autodl-fs/VLMEvalKit/.seed.env}}"
if [[ -f "$seed_env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$seed_env_file"
  set +a
fi

: "${ARK_BASE_URL:?ARK_BASE_URL is required. Source .seed.env or export it first.}"
: "${ARK_API_KEY:?ARK_API_KEY is required. Source .seed.env or export it first.}"
: "${ARK_MODEL:?ARK_MODEL is required. Source .seed.env or export it first.}"

export OPENAI_API_BASE="${OPENAI_API_BASE:-${ARK_BASE_URL%/}/chat/completions}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$ARK_API_KEY}"
export SEED_JUDGE_MODEL="${SEED_JUDGE_MODEL:-$ARK_MODEL}"
export SEED_TEMPERATURE="${SEED_TEMPERATURE:-0}"
export SEED_TOP_P="${SEED_TOP_P:-1}"
export SEED_MAX_TOKENS="${SEED_MAX_TOKENS:-1024}"
export SEED_JUDGE_ARGS="{\"temperature\":${SEED_TEMPERATURE},\"top_p\":${SEED_TOP_P},\"max_tokens\":${SEED_MAX_TOKENS}}"
