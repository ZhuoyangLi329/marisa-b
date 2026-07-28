#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: run_eft_v2_r0_shard.sh LABEL OUTPUT_JSONL LOG_DIRECTORY COMMAND [ARG ...]" >&2
    exit 2
fi

label=$1
output_jsonl=$2
log_directory=$3
shift 3

if [[ ! $label =~ ^[a-z0-9_]+$ ]]; then
    echo "invalid EFT-v2 R0 shard label: $label" >&2
    exit 2
fi

mkdir -p "$(dirname "$output_jsonl")" "$log_directory"

stderr_path="$log_directory/$label.err"
time_path="$log_directory/$label.time"
exitcode_path="$log_directory/$label.exitcode"

write_exitcode() {
    local status=$?
    printf '%s\n' "$status" > "$exitcode_path"
}
trap write_exitcode EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n ${MARISA_B_ENV_ACTIVATE:-} ]]; then
    if [[ ! -f $MARISA_B_ENV_ACTIVATE ]]; then
        printf 'missing MARISA_B_ENV_ACTIVATE file: %s\n' \
            "$MARISA_B_ENV_ACTIVATE" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "$MARISA_B_ENV_ACTIVATE" >/dev/null
fi

/usr/bin/time \
    -o "$time_path" \
    -f 'elapsed_seconds=%e max_rss_kb=%M user_seconds=%U system_seconds=%S' \
    "$@" > "$output_jsonl" 2> "$stderr_path"
status=$?
exit "$status"
