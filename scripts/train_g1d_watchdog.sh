#!/usr/bin/env bash
# Keep the g1d_scan_v1_full training run alive. Run on Linux; Ctrl-C stops supervision.
set -eu
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 WANDB_MODE=offline XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONUNBUFFERED=1
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning:torchvision.io._video_deprecation_warning"
exec 9>.g1d_scan_v1_full.lock
flock -n 9 || { echo "Another g1d_scan_v1_full watchdog is running." >&2; exit 1; }

trainer=
timer=
stop() {
    trap '' INT TERM HUP
    if [[ -n $timer ]]; then kill "$timer" 2>/dev/null || true; fi
    if [[ -n $trainer ]]; then
        kill -TERM -- "-$trainer" 2>/dev/null || true
        wait "$trainer" 2>/dev/null || true
    fi
    exit 0
}
trap stop INT TERM HUP

while true; do
    printf '[%s] Starting pi05_g1d_full g1d_scan_v1_full with --resume\n' "$(date -Is)"
    setsid uv run --no-sync scripts/train.py pi05_g1d_full --exp-name=g1d_scan_v1_full --resume 9>&- &
    trainer=$!
    while true; do
        sleep 1800 9>&- &
        timer=$!
        wait "$timer"
        timer=
        if ! kill -0 "$trainer" 2>/dev/null; then break; fi
        printf '[%s] Trainer alive (PID %s)\n' "$(date -Is)" "$trainer"
    done
    status=0
    wait "$trainer" || status=$?
    trainer=
    if (( status == 0 )); then
        echo "Training completed successfully."
        exit 0
    fi
    printf '[%s] Trainer exited with status %s; restarting.\n' "$(date -Is)" "$status" >&2
done
