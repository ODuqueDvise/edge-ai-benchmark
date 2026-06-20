#!/usr/bin/env python3
"""Mide precision (top-1 y top-5) del modelo sobre el conjunto de prueba.

Usa el mismo backend que el arnes de latencia, asi que la precision se mide por
(variante, condicion): TensorRT FP16/INT8 puede dar una precision distinta a la de
CPU FP32, y eso es justo lo que el OE1 quiere cuantificar.

Ejemplo:
  python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
      --provider cpu --device-tag jetson-cpu --dataset datasets/imagenetv2 --limit 2000
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

from . import metadata
from .backends import make_backend
from .datasets import iter_imagefolder, topk_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="ort", choices=["ort", "tflite"])
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--device-tag", required=True)
    ap.add_argument("--dataset", required=True, help="raiz con subcarpetas por indice de clase (ImageNet-V2)")
    ap.add_argument("--input-name", default=None)
    ap.add_argument("--limit", type=int, default=None, help="limitar numero de imagenes (prueba rapida)")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    backend = make_backend(a.backend, provider=a.provider)
    backend.load(a.model, input_name=a.input_name)

    n = top1 = top5 = 0
    t0 = time.time()
    for x, label in iter_imagefolder(a.dataset, limit=a.limit):
        out = backend.infer(x)
        h1, h5 = topk_hits(out[0], label, k=5)
        top1 += h1
        top5 += h5
        n += 1
        if n % 500 == 0:
            print("  ...%d imagenes  top1=%.4f" % (n, top1 / n))
    dur = time.time() - t0
    if n == 0:
        raise SystemExit("No se leyeron imagenes. Revisa --dataset (subcarpetas por indice de clase).")

    acc1, acc5 = top1 / n, top5 / n
    md = metadata.collect(model_path=a.model, backend_name=backend.name,
                          backend_version=str(backend.version), device_tag=a.device_tag,
                          extra={"dataset": a.dataset, "n_images": n,
                                 "active_providers": getattr(backend, "active_providers", None)})
    res = {"metadata": md, "accuracy": {"top1": acc1, "top5": acc5, "n_images": n, "eval_seconds": dur}}

    os.makedirs(a.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    model_tag = os.path.splitext(os.path.basename(a.model))[0]
    out = os.path.join(a.out_dir, "acc_%s_%s_%s_%s_%s.json" % (a.device_tag, model_tag, a.backend, a.provider, stamp))
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("Escrito:", out)
    print("  top-1=%.4f  top-5=%.4f  n=%d  (%.1fs)" % (acc1, acc5, n, dur))


if __name__ == "__main__":
    main()
