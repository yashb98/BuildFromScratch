#!/usr/bin/env python3
"""Dense, crash-durable thermal logger for the GB10 (added 2026-07-08).

The box hard-locks under sustained 420M-training load every ~5-10h with NO kernel
trace (crashkernel=0M) and — critically — its temperature was never instrumented, so
the "it's crashing from heat" hypothesis had zero data. This is a PURE-OBSERVABILITY
sampler: it has NO authority to kill anything (that stays with sentinel.py's single
hardened kill path), it only records the box's thermal envelope so we can (a) confirm
or refute the heat theory and (b) have the last-second-before-a-freeze reading on disk
for forensics.

Every sample is flushed + fsync'd, so the final line written before an instantaneous
hard-lock is durable (the whole point). Sources (all no-sudo): nvidia-smi for the GPU
die temp / power / SM clock / thermal+power throttle flags, /sys/class/thermal for the
7 Grace-SoC ACPI zones, /proc/meminfo for the unified-pool usage.

Usage:  python3 thermal_log.py [--interval SEC] [--out PATH] [--pid TRAINER_PID]
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import time
from datetime import datetime, timezone


def nvidia_fields():
    """gpu temp C, power W, SM clock MHz, and throttle flags. '' for any N/A field."""
    q = ("temperature.gpu,power.draw,clocks.sm,"
         "clocks_throttle_reasons.hw_thermal_slowdown,"
         "clocks_throttle_reasons.sw_thermal_slowdown,"
         "clocks_throttle_reasons.hw_power_brake_slowdown,"
         "clocks_throttle_reasons.sw_power_cap")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7 and parts[0]:
            return {
                "gpu_c": parts[0], "pwr_w": parts[1], "sm_mhz": parts[2],
                "hw_therm": parts[3], "sw_therm": parts[4],
                "hw_pbrake": parts[5], "sw_pcap": parts[6],
            }
    return {}


def soc_zones():
    """List of (zone_index, milli-degC-as-C) for every ACPI thermal zone."""
    zones = []
    for p in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            with open(p) as f:
                c = int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        idx = p.split("thermal_zone")[1].split("/")[0]
        zones.append((idx, c))
    return zones


def pool_usage():
    """Unified-memory pool usage fraction from /proc/meminfo, or None."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                info[k] = int(rest.split()[0])  # kiB
        total, avail = info["MemTotal"], info["MemAvailable"]
        return (total - avail) / total
    except (OSError, KeyError, ValueError, IndexError):
        return None


def pid_alive(pid):
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=10.0, help="sample period seconds")
    ap.add_argument("--out", default=str(
        (os.path.dirname(os.path.abspath(__file__)) or ".") + "/thermal.log"))
    ap.add_argument("--pid", type=int, default=None, help="trainer pid to note liveness")
    a = ap.parse_args()

    f = open(a.out, "a", buffering=1)  # line-buffered
    f.write(f"# thermal_log start {datetime.now(timezone.utc).isoformat()} "
            f"interval={a.interval}s pid={a.pid}\n")
    f.write("# ts_utc epoch gpu_c pwr_w sm_mhz throttle pool% soc_hot_c soc_zones... pid_alive\n")
    f.flush(); os.fsync(f.fileno())

    while True:
        nv = nvidia_fields()
        zones = soc_zones()
        pool = pool_usage()
        hot = max((c for _, c in zones), default=None)
        # compact throttle summary: T=hw/sw thermal, P=power-brake/cap; '-' if none/N-A
        thr = []
        if nv.get("hw_therm") == "Active" or nv.get("sw_therm") == "Active":
            thr.append("THERMAL")
        if nv.get("hw_pbrake") == "Active" or nv.get("sw_pcap") == "Active":
            thr.append("POWER")
        thr_s = "+".join(thr) if thr else "-"
        zone_s = " ".join(f"z{idx}={c:.1f}" for idx, c in zones)
        pool_s = f"{pool*100:.1f}%" if pool is not None else "na"
        hot_s = f"{hot:.1f}C" if hot is not None else "na"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = (f"{ts} {time.time():.0f} gpu={nv.get('gpu_c','na')}C "
               f"pwr={nv.get('pwr_w','na')}W sm={nv.get('sm_mhz','na')}MHz "
               f"thr={thr_s} pool={pool_s} soc_hot={hot_s} {zone_s}")
        alive = pid_alive(a.pid)
        if alive is not None:
            row += f" pid_alive={alive}"
        f.write(row + "\n")
        f.flush(); os.fsync(f.fileno())  # durable: survive an instant hard-lock
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
