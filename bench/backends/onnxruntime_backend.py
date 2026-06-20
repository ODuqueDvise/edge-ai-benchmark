"""Backend ONNX Runtime con proveedor seleccionable.

Un solo marco cubre las tres condiciones, lo que reduce el sesgo por runtime:
  provider='tensorrt' -> TensorrtExecutionProvider  (Jetson GPU)
  provider='cuda'     -> CUDAExecutionProvider       (Jetson GPU)
  provider='cpu'      -> CPUExecutionProvider        (Jetson CPU / RPi CPU)
"""
from __future__ import annotations
import numpy as np

_PROVIDERS = {
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


class OnnxRuntimeBackend:
    name = "onnxruntime"

    def __init__(self, provider="cpu", intra_op_threads=0, trt_int8_table=None):
        self.provider = provider
        self.intra_op_threads = intra_op_threads
        self.trt_int8_table = trt_int8_table   # ruta a calibration.flatbuffers (INT8 en TensorRT, plan B D14)
        self._sess = None
        self._input_name = None
        self._ort = None

    def load(self, model_path, input_name=None, **kw):
        import onnxruntime as ort
        self._ort = ort
        so = ort.SessionOptions()
        if self.intra_op_threads:
            so.intra_op_num_threads = int(self.intra_op_threads)
        providers = _PROVIDERS.get(self.provider, ["CPUExecutionProvider"])
        if self.provider == "tensorrt" and self.trt_int8_table:
            import os
            d = os.path.dirname(os.path.abspath(self.trt_int8_table)) or "."
            trt_opts = {
                "trt_int8_enable": True,
                "trt_int8_calibration_table_name": os.path.basename(self.trt_int8_table),
                "trt_int8_use_native_calibration_table": False,   # tabla generada por la herramienta de ORT
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": d,                       # ORT busca la tabla en esta carpeta
            }
            providers = [("TensorrtExecutionProvider", trt_opts),
                         "CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        active = self._sess.get_providers()
        # Aviso temprano si el proveedor pedido no quedo activo (cae a CPU en silencio).
        wanted = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        if wanted not in active:
            print("[ADVERTENCIA] proveedor %r no activo; activos=%s. "
                  "La GPU podria no estar en uso." % (wanted, active))
        self._input_name = input_name or self._sess.get_inputs()[0].name
        return self

    def infer(self, x):
        return self._sess.run(None, {self._input_name: x})

    def synchronize(self):
        return None  # ONNX Runtime.run() es sincrono

    @property
    def version(self):
        return self._ort.__version__ if self._ort else None

    @property
    def active_providers(self):
        return self._sess.get_providers() if self._sess else None

    @property
    def input_name(self):
        return self._input_name
