"""Backend TFLite (CPU). Alternativa al CPUExecutionProvider de ONNX Runtime."""
from __future__ import annotations


class TFLiteBackend:
    name = "tflite"

    def __init__(self, intra_op_threads=0):
        self.intra_op_threads = intra_op_threads
        self._interp = None
        self._in = None
        self._out = None
        self._ver = None

    def load(self, model_path, **kw):
        try:
            from tflite_runtime.interpreter import Interpreter
            self._ver = "tflite_runtime"
        except Exception:
            from tensorflow.lite import Interpreter  # type: ignore
            self._ver = "tf.lite"
        threads = self.intra_op_threads or None
        self._interp = Interpreter(model_path=model_path, num_threads=threads)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        return self

    def infer(self, x):
        self._interp.set_tensor(self._in["index"], x)
        self._interp.invoke()
        return self._interp.get_tensor(self._out["index"])

    @property
    def version(self):
        return self._ver

    @property
    def input_name(self):
        return self._in["name"] if self._in else None
