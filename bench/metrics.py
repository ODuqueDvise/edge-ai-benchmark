"""Agregación de latencia y energía. Sin dependencias de hardware."""
from __future__ import annotations
import numpy as np


def summarize_latency(latencies_ms):
    """Resume una lista de latencias por inferencia (ms).

    Reporta percentiles además de la media, porque en el borde la cola de la
    distribucion (los casos lentos) suele importar tanto como el promedio.
    """
    a = np.asarray(latencies_ms, dtype=np.float64)
    if a.size == 0:
        raise ValueError("lista de latencias vacia")
    mean = float(a.mean())
    return {
        "n": int(a.size),
        "mean_ms": mean,
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min_ms": float(a.min()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
        "throughput_ips": (1000.0 / mean) if mean > 0 else None,
    }


def integrate_energy(timestamps_s, powers_w):
    """Energia (J) por integracion trapezoidal de potencia (W) sobre el tiempo (s).

    Pensado para la lectura cruzada de los sensores internos de la Jetson. La
    fuente PRIMARIA de energia es el medidor externo (ver protocolo, seccion 6);
    estas muestras solo sirven de referencia.
    """
    t = np.asarray(timestamps_s, dtype=np.float64)
    p = np.asarray(powers_w, dtype=np.float64)
    if t.size < 2:
        return {"energy_j": None, "avg_power_w": float(p.mean()) if p.size else None,
                "samples": int(p.size)}
    energy = float(np.trapz(p, t))
    return {
        "energy_j": energy,
        "avg_power_w": float(p.mean()),
        "duration_s": float(t[-1] - t[0]),
        "samples": int(p.size),
    }


def energy_per_inference(energy_j, n_inferences):
    if energy_j is None or not n_inferences:
        return None
    return energy_j / n_inferences
