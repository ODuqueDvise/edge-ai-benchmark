from .base import Backend

def make_backend(kind, provider="cpu", **kw):
    kind = (kind or "ort").lower()
    if kind == "ort":
        from .onnxruntime_backend import OnnxRuntimeBackend
        return OnnxRuntimeBackend(provider=provider, **kw)
    if kind == "tflite":
        from .tflite_backend import TFLiteBackend
        kw.pop("trt_int8_table", None)   # opcion exclusiva de ORT/TensorRT
        return TFLiteBackend(**kw)
    raise ValueError("backend desconocido: %r (use 'ort' o 'tflite')" % kind)
