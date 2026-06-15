#!/usr/bin/env python3
"""Exporta un modelo CNN a un ONNX autocontenido (un solo archivo) y reporta su SHA-256.

Robusto frente a torch >=2.x: el exportador nuevo puede guardar los pesos como
datos externos (.onnx + .onnx.data). Este script los incrusta en un unico .onnx,
para que el modelo se comparta como un solo archivo verificable por checksum.

Requiere (solo en la maquina que exporta): torch, torchvision, onnx, onnxruntime, numpy.
Ejemplo:
  python scripts/export_model.py --model-name mobilenet_v2 --output models/cnn_baseline.onnx
"""
import argparse, hashlib, os


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="mobilenet_v2",
                    help="nombre de un modelo de torchvision.models (def: mobilenet_v2)")
    ap.add_argument("--output", default="models/cnn_baseline.onnx")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--input-shape", default="1,3,224,224")
    a = ap.parse_args()

    import torch
    import torchvision.models as M

    shape = tuple(int(s) for s in a.input_shape.split(","))
    model = getattr(M, a.model_name)(weights="DEFAULT").eval()
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    torch.onnx.export(model, torch.randn(*shape), a.output, opset_version=a.opset,
                      input_names=["input"], output_names=["output"])

    # Garantizar UN solo archivo autocontenido: incrustar datos externos si los hubo.
    import onnx
    m = onnx.load(a.output)                                  # carga pesos externos si existen
    onnx.save_model(m, a.output, save_as_external_data=False)
    ext = a.output + ".data"
    if os.path.exists(ext):
        os.remove(ext)

    print("Exportado:", a.output, "(%.1f MB)" % (os.path.getsize(a.output) / 1e6))
    print("SHA-256 :", sha256(a.output))

    try:
        import numpy as np
        import onnxruntime as ort
        s = ort.InferenceSession(a.output, providers=["CPUExecutionProvider"])
        o = s.run(None, {s.get_inputs()[0].name: np.random.randn(*shape).astype("float32")})
        print("Verificacion ORT OK; salida:", o[0].shape)
    except Exception as e:
        print("[aviso] no se pudo verificar con onnxruntime:", e)


if __name__ == "__main__":
    main()
