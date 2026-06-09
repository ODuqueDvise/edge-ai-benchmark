from .base import Backend

def make_backend(kind, provider="cpu", **kw):
    kind = (kind or "ort").lower()
    if kind == "ort":
        from .onnxruntime_backend import OnnxRuntimeBackend
        return OnnxRuntimeBackend(provider=provider, **kw)
    if kind == "tflite":
        from .tflite_backend import TFLiteBackend
        return TFLiteBackend(**kw)
    raise ValueError("backend desconocido: %r (use 'ort' o 'tflite')" % kind)
