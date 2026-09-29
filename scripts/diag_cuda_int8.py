#!/usr/bin/env python3
"""Diagnostico D25: ¿ofrece el proveedor CUDA de ONNX Runtime una ruta INT8 real para el grafo QDQ?

Corre en la Jetson. Para cada modelo abre dos sesiones sobre CUDAExecutionProvider, una con el
FP32 y otra con el QDQ (*_int8.onnx), con el registro de ORT en nivel INFO para que imprima la
ASIGNACION DE NODOS por proveedor, y mide una latencia corta (no entra al analisis: iters < 2000).

Lectura del resultado (los tres criterios a la vez):
  1. Carga: si la sesion QDQ falla al construirse, no hay ruta.
  2. Asignacion: buscar en la salida "Node(s) placed on [CPUExecutionProvider]". Si hay nodos
     Conv/MatMul/QLinear* en CPU, el proveedor no ejecuta la cuantizacion en la GPU.
  3. Velocidad: si el QDQ no es mas rapido que el FP32 en el MISMO proveedor, no es una ruta INT8
     aunque cargue (Q/DQ simulados sobre nucleos FP32).
Solo si (1) carga, (2) todo en CUDA y (3) acelera, existe una ruta INT8 equivalente a la de la CPU.

Uso (en la Jetson, dentro del venv del arnes):
  python3 scripts/diag_cuda_int8.py 2>&1 | tee results/diag/diag_cuda_int8_$(date -u +%Y%m%d-%H%M%S).log
"""
import sys, time, os, statistics as st
import numpy as np
import onnxruntime as ort

PAIRS = [("models/resnet50_baseline.onnx", "models/resnet50_baseline_int8.onnx"),
         ("models/cnn_baseline.onnx",      "models/cnn_baseline_int8.onnx")]
WARMUP, ITERS = 20, 200

def session(path):
    so = ort.SessionOptions()
    so.log_severity_level = 1          # INFO: imprime la asignacion de nodos por proveedor
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

def latency_ms(sess):
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    for _ in range(WARMUP): sess.run(None, {name: x})
    t = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); sess.run(None, {name: x}); t.append((time.perf_counter() - t0) * 1e3)
    return st.median(t), np.percentile(t, 95)

def main():
    print("onnxruntime", ort.__version__, "| proveedores disponibles:", ort.get_available_providers())
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("SIN CUDAExecutionProvider en esta instalacion: el diagnostico no aplica."); return 2
    for fp32, qdq in PAIRS:
        for label, path in (("FP32", fp32), ("QDQ INT8", qdq)):
            print("\n" + "=" * 78 + "\n%s  %s\n" % (label, path) + "=" * 78, flush=True)
            try:
                s = session(path)
            except Exception as e:
                print("NO CARGA en CUDA EP:", repr(e)[:300]); continue
            print("proveedores activos en la sesion:", s.get_providers())
            p50, p95 = latency_ms(s)
            print("latencia p50 %.3f ms | p95 %.3f ms (warmup %d, iters %d; diagnostico, no oficial)" % (p50, p95, WARMUP, ITERS), flush=True)
    print("\nInterpretar con los tres criterios del encabezado. Registrar en BITACORA antes de escribir nada en el articulo.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
