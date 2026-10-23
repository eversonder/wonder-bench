"""Shared bits: where results go, system info, and a background Pi 5 telemetry sampler."""
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

# Ubuntu on the Pi leaves /dev/vcio root-only; read telemetry through passwordless sudo there.
VC = ["vcgencmd"] if os.access("/dev/vcio", os.R_OK | os.W_OK) else ["sudo", "-n", "vcgencmd"]
RESULTS = Path(os.environ.get("WB_RESULTS", Path(__file__).resolve().parents[1] / "results")) / f"{platform.node()}-{time.strftime('%Y%m%d')}"


def save(name, data):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"wrote {path}")
    return path


def _sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def sysinfo():
    model = Path("/proc/device-tree/model")
    return {
        "host": platform.node(),
        "model": model.read_text().strip("\x00\n") if model.exists() else platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "mem_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1) if hasattr(os, "sysconf") else None,
        "governor": _sh("cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "throttled": _sh(*VC, "get_throttled"),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _pmic_watts():
    """Pi 5 only: sum V*I over the PMIC rails. Excludes USB peripherals and conversion losses, so it reads low."""
    out = _sh(*VC, "pmic_read_adc")
    amps, volts = {}, {}
    for name, kind, val in re.findall(r"(\S+)_[AV]\s+(current|volt)\(\d+\)=([\d.]+)", out):
        (amps if kind == "current" else volts)[name] = float(val)
    return round(sum(a * volts[n] for n, a in amps.items() if n in volts), 3) if amps else None


def _temp_c():
    t = Path("/sys/class/thermal/thermal_zone0/temp")
    return round(int(t.read_text()) / 1000, 1) if t.exists() else None


def _arm_mhz():
    m = re.search(r"=(\d+)", _sh(*VC, "measure_clock", "arm"))
    return round(int(m.group(1)) / 1e6) if m else None


class Telemetry:
    """with Telemetry() as t: ...  →  t.summary() gives mean/peak watts, max temp, min clock, throttle flags."""

    def __init__(self, every=0.5):
        self.every, self.samples, self._stop = every, [], threading.Event()
        self.enabled = shutil.which("vcgencmd") is not None

    def _run(self):
        while not self._stop.wait(self.every):
            self.samples.append({"t": time.time(), "w": _pmic_watts(), "c": _temp_c(), "mhz": _arm_mhz()})

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.enabled:
            self._thread.join()

    def summary(self):
        if not self.samples:
            return {"telemetry": "unavailable (not a Pi, or vcgencmd missing)"}
        col = lambda k: [s[k] for s in self.samples if s[k] is not None]
        w, c, mhz = col("w"), col("c"), col("mhz")
        return {
            "watts_mean": round(sum(w) / len(w), 2) if w else None,
            "watts_peak": max(w) if w else None,
            "temp_max_c": max(c) if c else None,
            "arm_mhz_min": min(mhz) if mhz else None,
            "throttled_after": _sh(*VC, "get_throttled"),
            "samples": len(self.samples),
        }
