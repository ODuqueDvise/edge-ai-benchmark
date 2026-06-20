#!/usr/bin/env python3
"""Genera una tabla de calibración INT8 de TensorRT desde un modelo FP32 + imágenes.

Plan B del OE1 (DECISIONS D14): en jetson-gpu, TensorRT cuantiza el modelo en precisión
completa usando esta tabla (proveedor TensorRT con trt_int8_enable +
trt_int8_calibration_table_name). NO se usa con modelos QDQ —TensorRT no admite tabla de
calibración si el modelo tiene nodos Q/DQ—. La tabla la consume el proveedor con
trt_int8_use_native_calibration_table=False.

Reusa el MISMO preprocesamiento del arnés (bench.datasets) y el MISMO conjunto de
calibración que el INT8 de CPU. Corre en CPU: no necesita GPU. Genera, en --out-dir,
los archivos calibration.flatbuffers / .cache / .json.

Requiere: onnxruntime, onnx, numpy, pillow.
Ejemplo:
  python scripts/make_trt_calib_table.py --model models/resnet50_baseline.onnx \
      --calib-dir datasets/calib_imagenet1k --out-dir models/trt_calib_resnet50 --limit 300
"""
from __future__ import annotations
import argparse, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.datasets import preprocess_image  # noqa: E402


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
    ap.add_argument("--model", required=True, help="ONNX FP32 (NO el QDQ)")
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="carpeta donde se escribe la tabla de calibración")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--method", default="minmax", choices=["minmax", "entropy", "percentile"])
    a = ap.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                          create_calibrator, write_calibration_table)

    os.makedirs(a.out_dir, exist_ok=True)
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    files = list_images(a.calib_dir, a.limit)
    print("Calibración TensorRT: %d imágenes | entrada '%s' | método %s" % (len(files), input_name, a.method))

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(files)

        def get_next(self):
            f = next(self.it, None)
            return None if f is None else {input_name: preprocess_image(f)}

    method = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
              "percentile": CalibrationMethod.Percentile}[a.method]
    calibrator = create_calibrator(
        a.model, op_types_to_calibrate=[],
        augmented_model_path=os.path.join(a.out_dir, "augmented.onnx"),
        calibrate_method=method)
    calibrator.collect_data(Reader())

    # API varía por versión: compute_data() (nuevo) devuelve TensorsData; compute_range() (viejo) un dict.
    try:
        data = calibrator.compute_data()
    except AttributeError:
        data = calibrator.compute_range()
    write_calibration_table(data, dir=a.out_dir)

    aug = os.path.join(a.out_dir, "augmented.onnx")
    if os.path.exists(aug):
        os.remove(aug)
    produced = sorted(f for f in os.listdir(a.out_dir) if f.startswith("calibration"))
    print("Tabla escrita en %s: %s" % (a.out_dir, produced))
    print("La consume el proveedor TensorRT con trt_int8_calibration_table_name='calibration.flatbuffers'.")


if __name__ == "__main__":
    main()
