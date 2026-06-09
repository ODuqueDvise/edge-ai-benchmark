"""Recoleccion de metadatos por corrida: plataforma, runtime, estado termico, modelo."""
from __future__ import annotations
import hashlib, os, platform, subprocess, time, glob


def _run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL,
                                       text=True, timeout=10).strip()
    except Exception:
        return None


def sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def thermal_zones_c():
    """Temperaturas (C) de /sys/class/thermal. Util en Jetson y RPi."""
    out = {}
    for zp in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            name = open(os.path.join(zp, "type")).read().strip()
            milli = int(open(os.path.join(zp, "temp")).read().strip())
            out[name] = round(milli / 1000.0, 2)
        except Exception:
            pass
    return out


def nvpmodel_mode():
    """Modo de potencia activo en Jetson (None si no aplica)."""
    return _run("nvpmodel -q 2>/dev/null | tr '\\n' ' '")


def collect(model_path=None, backend_name=None, backend_version=None,
            device_tag=None, power_mode=None, extra=None):
    md = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_tag": device_tag,
        "power_mode_declared": power_mode,
        "nvpmodel": nvpmodel_mode(),
        "platform": {
            "node": platform.node(),
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "backend": {"name": backend_name, "version": backend_version},
        "model": {"path": model_path, "sha256": sha256_file(model_path)},
        "thermal_c_start": thermal_zones_c(),
    }
    if extra:
        md.update(extra)
    return md
