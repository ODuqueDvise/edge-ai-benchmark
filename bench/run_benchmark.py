#!/usr/bin/env python3
"""Punto de entrada: ejecuta una corrida de benchmark y escribe un JSON trazable.

Cada corrida = una condicion experimental (un backend/proveedor en un equipo).
La salida incluye latencias CRUDAS por inferencia, agregados, metadatos y, en
Jetson, la serie de potencia interna como referencia cruzada.

Ejemplos:
  # Jetson, GPU (TensorRT EP), latencia con entrada sintetica
  python -m bench.run_benchmark --model models/cnn.onnx --backend ort \
      --provider tensorrt --device-tag jetson-gpu --input-shape 1,3,224,224 \
      --warmup 50 --iters 1000

  # Jetson, CPU (misma maquina) -> condicion de aislamiento del acelerador
  python -m bench.run_benchmark --model models/cnn.onnx --backend ort \
      --provider cpu --device-tag jetson-cpu --input-shape 1,3,224,224

  # Raspberry Pi 5, CPU
  python -m bench.run_benchmark --model models/cnn.onnx --backend ort \
      --provider cpu --device-tag rpi-cpu --input-shape 1,3,224,224
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

from . import metrics, harness, metadata
from .backends import make_backend
from .power import JetsonPowerSampler
from .datasets import synthetic_input


def parse_args():
    p = argparse.ArgumentParser(description="Arnes de benchmark edge AI (GPU vs CPU).")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", default="ort", choices=["ort", "tflite"])
    p.add_argument("--provider", default="cpu", help="ort: tensorrt|cuda|cpu ; tflite: cpu")
    p.add_argument("--device-tag", required=True, help="jetson-gpu | jetson-cpu | rpi-cpu")
    p.add_argument("--power-mode", default=None, help="etiqueta informativa (p.ej. MAXN)")
    p.add_argument("--input-name", default=None)
    p.add_argument("--input-shape", default=None, help="coma-separado, p.ej. 1,3,224,224")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main():
    a = parse_args()
    backend = make_backend(a.backend, provider=a.provider, intra_op_threads=a.threads)
    backend.load(a.model, input_name=a.input_name)

    if not a.input_shape:
        raise SystemExit("--input-shape es obligatorio para la corrida de latencia "
                         "(la medicion de precision se activa por separado).")
    shape = tuple(int(s) for s in a.input_shape.split(","))
    x = synthetic_input(shape, dtype=np.dtype(a.dtype))
    provider = lambda i: x  # misma entrada; mide costo de computo, no de E/S

    sampler = JetsonPowerSampler()
    win_start = time.time()
    with sampler:
        lat = harness.run_latency(backend.infer, provider,
                                  warmup=a.warmup, iters=a.iters,
                                  sync_fn=backend.synchronize)
    win_end = time.time()

    ts, ws = sampler.series()
    energy = metrics.integrate_energy(ts, ws) if ts else {"energy_j": None, "samples": 0}
    summary = metrics.summarize_latency(lat)

    md = metadata.collect(
        model_path=a.model, backend_name=backend.name,
        backend_version=str(backend.version), device_tag=a.device_tag,
        power_mode=a.power_mode,
        extra={"active_providers": getattr(backend, "active_providers", None),
               "thermal_c_end": metadata.thermal_zones_c(),
               "input_shape": list(shape), "dtype": a.dtype,
               "warmup": a.warmup, "iters": a.iters,
               "window": {"start_epoch_s": win_start, "end_epoch_s": win_end}})

    result = {
        "metadata": md,
        "latency_summary": summary,
        "latency_raw_ms": lat,                       # CRUDO: permite recalcular/auditar
        "power_internal_crossref": {                 # NO comparar entre equipos
            **energy,
            "note": "Referencia cruzada (sensores Jetson). Fuente primaria: medidor externo.",
        },
    }

    os.makedirs(a.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    fname = "%s_%s_%s_%s.json" % (a.device_tag, a.backend, a.provider, stamp)
    out = os.path.join(a.out_dir, fname)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("Escrito:", out)
    print("  p50=%.3f ms  p95=%.3f ms  p99=%.3f ms  thr=%.1f ips" % (
        summary["p50_ms"], summary["p95_ms"], summary["p99_ms"],
        summary["throughput_ips"] or 0.0))
    if energy.get("avg_power_w"):
        print("  potencia media (ref. interna)=%.2f W" % energy["avg_power_w"])


if __name__ == "__main__":
    main()
