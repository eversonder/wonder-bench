#!/usr/bin/env bash
# Run every week-1 benchmark in sequence (each needs the Pi to itself), then write REPORT.md.
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
export PYTHONUNBUFFERED=1
out=results/$(hostname)-$(date +%Y%m%d)

cool() { # start each benchmark from a similar temperature so runs are comparable
  while t=$(cat /sys/class/thermal/thermal_zone0/temp); [ "$t" -gt 60000 ]; do sleep 10; done
}

cool; (cd bench && python train.py)
cool; bash bench/llm.sh
cool; (cd bench && python asr.py)
cool; (cd bench && python mimi.py)
cool; (cd bench && python tts.py)
python report.py "$out"
