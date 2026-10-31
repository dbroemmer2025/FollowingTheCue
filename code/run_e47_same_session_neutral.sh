#!/usr/bin/env bash
# Regenerate the neutral control and all eligible uncued baselines in one runtime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/e47"
DEVICE=auto
SIDECAR_NAME=neutral_insertion_same_session
START_AT=
ONLY=
DRY_RUN=0
while (($#)); do
  case "$1" in
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --sidecar-name) SIDECAR_NAME="$2"; shift 2 ;;
    --start-at) START_AT="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

CELLS=(
  gemma_2_9b_it_movie_recommendations_thinking_v10
  gemma_2_9b_it_temporal_sequences_thinking_v10
  mistral_nemo_instruct_2407_movie_recommendations_thinking_v10
  mistral_nemo_instruct_2407_temporal_sequences_thinking_v10
  olmo_2_1124_7b_instruct_movie_recommendations_thinking_v10
  olmo_2_1124_7b_instruct_temporal_sequences_thinking_v10
  qwen2_5_7b_movie_recommendations_thinking_v10
  qwen2_5_7b_temporal_sequences_thinking_v10
  qwen2_7b_it_movie_recommendations_thinking_v10
  qwen2_7b_it_temporal_sequences_thinking_v10
)

if [[ -n "$START_AT" && -n "$ONLY" ]]; then
  printf '%s\n' 'Use --only or --start-at, not both' >&2
  exit 2
fi

selected="${ONLY:-$START_AT}"
if [[ -n "$selected" ]]; then
  found=0
  for cell in "${CELLS[@]}"; do
    [[ "$cell" == "$selected" ]] && found=1
  done
  if (( ! found )); then
    printf 'Unknown cell: %s\n' "$selected" >&2
    exit 2
  fi
fi

print_command() {
  printf '%q' "$1"
  shift
  if (($#)); then printf ' %q' "$@"; fi
  printf '\n'
}

if (( DRY_RUN )); then
  printf '%s\n' 'export E47_NEUTRAL_SESSION_ID="$(python -c '\''import uuid; print(uuid.uuid4())'\'')"'
else
  export E47_NEUTRAL_SESSION_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
  printf 'Session ID: %s\n' "$E47_NEUTRAL_SESSION_ID"
fi
started=0
for cell in "${CELLS[@]}"; do
  if [[ -n "$ONLY" && "$cell" != "$ONLY" ]]; then continue; fi
  if [[ -n "$START_AT" && "$cell" == "$START_AT" ]]; then started=1; fi
  if [[ -n "$START_AT" && "$started" == 0 ]]; then continue; fi
  run="$RUNS_ROOT/$cell"
  sidecar="$run/$SIDECAR_NAME"
  generate_cmd=(python "$SCRIPT_DIR/e47_neutral_insertion.py" generate
    --run "$run" --out "$sidecar" --device "$DEVICE")
  replay_cmd=(python "$SCRIPT_DIR/e47_neutral_insertion.py" replay
    --run "$run" --out "$sidecar" --device "$DEVICE"
    --n 100000 --replay-name full_uncued)
  report_cmd=(python "$SCRIPT_DIR/e47_neutral_insertion.py" report
    --run "$run" --out "$sidecar")
  audit_cmd=(python "$SCRIPT_DIR/e47_uncued_replay_audit.py"
    --run "$run" --sidecar-name "$SIDECAR_NAME"
    --replay-name full_uncued)
  if (( DRY_RUN )); then
    print_command "${generate_cmd[@]}"
    print_command "${replay_cmd[@]}"
    print_command "${report_cmd[@]}"
    print_command "${audit_cmd[@]}"
    continue
  fi
  if [[ -f "$sidecar/runtime.jsonl" ]]; then
    python - "$sidecar/runtime.jsonl" "$E47_NEUTRAL_SESSION_ID" <<'PY'
import json
import sys

path, session = sys.argv[1:]
with open(path) as stream:
    previous = {json.loads(line).get('session_id') for line in stream if line.strip()}
if previous != {session}:
    raise SystemExit(f'{path}: belongs to a different or unrecorded session; '
                     'use --sidecar-name with a fresh name')
PY
  fi
  "${generate_cmd[@]}"
  "${replay_cmd[@]}"
  "${report_cmd[@]}"
  "${audit_cmd[@]}"
done
