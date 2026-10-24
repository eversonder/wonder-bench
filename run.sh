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

# each step is skipped if its result already exists, so an interrupted run resumes where it stopped
step() { [ -s "$out/$1.json" ] && echo "skip $1 (have $out/$1.json)" || { cool; echo "== $1"; shift; "$@"; }; }
step train bash -c 'cd bench && python train.py'
step llm   bash bench/llm.sh
step asr   bash -c 'cd bench && python asr.py'
step mimi  bash -c 'cd bench && python mimi.py'
step tts   bash -c 'cd bench && python tts.py'
step decoder bash -c 'cd bench && python decoder.py'
python report.py "$out"
