#!/usr/bin/env python3
"""Cuantiza un ONNX FP32 a INT8 estático (PTQ, formato QDQ) — primera técnica del OE1.

Ver docs/DISENO_INT8_OE1.md y DECISIONS D13. Esquema S8S8, pesos per-canal, calibración
Entropy (def) o Percentile, sobre --calib-dir (imágenes con el MISMO preprocesamiento que
el arnés, reusando bench.datasets.preprocess_image). Produce <out> + su SHA-256 + la lista
de imágenes de calibración usadas (evidencia).

IMPORTANTE: NO calibrar sobre el ImageNet-V2 de evaluación (sería fuga de datos). Usar un
conjunto aparte (~256–500 imágenes de ImageNet-1k val). La evaluación oficial sigue siendo
el V2 completo.

Requiere (solo en la máquina que cuantiza): onnxruntime, onnx, numpy, pillow.
Ejemplo:
  python scripts/quantize_int8.py --model models/cnn_baseline.onnx \
      --calib-dir datasets/calib_imagenet1k --out models/cnn_baseline_int8.onnx --limit 300
"""
from __future__ import annotations
import argparse, glob, hashlib, os, sys

# permitir 'import bench.datasets' (preprocesamiento idéntico al del arnés)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.datasets import preprocess_image  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def list_images(d, limit):
    files = []
    for ext in ("jpg", "jpeg", "png", "JPEG", "JPG", "PNG"):
        files += glob.glob(os.path.join(d, "**", "*." + ext), recursive=True)
    files = sorted(set(files))
    if not files:
        sys.exit("No hay imágenes (jpg/png) en %s" % d)
    return files[:limit] if limit else files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ONNX FP32 de entrada (p.ej. models/cnn_baseline.onnx)")
    ap.add_argument("--calib-dir", required=True, help="carpeta de imágenes de calibración (NO el V2 de evaluación)")
    ap.add_argument("--out", required=True, help="ONNX INT8 de salida (p.ej. models/cnn_baseline_int8.onnx)")
    ap.add_argument("--limit", type=int, default=300, help="nº de imágenes de calibración (def 300)")
    ap.add_argument("--method", default="entropy", choices=["entropy", "percentile", "minmax"])
    ap.add_argument("--no-per-channel", action="store_true", help="desactiva per-canal (no recomendado)")
    a = ap.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import (quantize_static, QuantType, QuantFormat,
                                          CalibrationMethod, CalibrationDataReader)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # nombre real de la entrada del modelo
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    files = list_images(a.calib_dir, a.limit)
    per_channel = not a.no_per_channel
    print("Calibración: %d imágenes de %s | entrada '%s' | método %s | per-canal %s"
          % (len(files), a.calib_dir, input_name, a.method, per_channel))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(files)

        def get_next(self):
            f = next(self.it, None)
            return None if f is None else {input_name: preprocess_image(f)}

    # 1) pre-procesar (inferencia de formas + fusión): mejora la cuantización
    pre = a.out + ".pre.onnx"
    quant_pre_process(a.model, pre)

    # 2) cuantización estática, formato QDQ, S8S8 (activaciones+pesos int8), per-canal
    method = {"entropy": CalibrationMethod.Entropy,
              "percentile": CalibrationMethod.Percentile,
              "minmax": CalibrationMethod.MinMax}[a.method]
    quantize_static(pre, a.out, Reader(),
                    quant_format=QuantFormat.QDQ,
                    per_channel=per_channel,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    calibrate_method=method)
    if os.path.exists(pre):
        os.remove(pre)

    # evidencia: lista de archivos de calibración usados (no las imágenes)
    man = a.out + ".calib.txt"
    with open(man, "w") as f:
        f.write("# calibración de %s | método=%s | per_channel=%s | n=%d\n"
                % (os.path.basename(a.out), a.method, per_channel, len(files)))
        for fn in files:
            f.write(fn + "\n")

    sz_in, sz_out = os.path.getsize(a.model) / 1e6, os.path.getsize(a.out) / 1e6
    print("Cuantizado: %s (%.1f MB -> %.1f MB, %.1f×)" % (a.out, sz_in, sz_out, sz_in / max(sz_out, 1e-9)))
    print("SHA-256 :", sha256(a.out))
    print("Lista de calibración:", man)

    # verificación rápida: carga y corre en ORT CPU
    try:
        s = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
        o = s.run(None, {input_name: preprocess_image(files[0])})
        print("Verificación ORT OK; salida:", o[0].shape)
    except Exception as e:
        print("[aviso] no se pudo verificar el INT8 con onnxruntime:", e)


if __name__ == "__main__":
    main()
