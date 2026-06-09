"""Bucle de medicion de latencia: calentamiento, medicion, estado estable."""
from __future__ import annotations
import time


def run_latency(infer_fn, input_provider, warmup=50, iters=1000, sync_fn=None):
    """Ejecuta el bucle de latencia y devuelve la lista de tiempos por inferencia (ms).

    infer_fn(x): ejecuta UNA inferencia. Debe ser bloqueante: si el backend es
        asincrono (GPU), la sincronizacion debe ocurrir dentro de infer_fn o via
        sync_fn, para no medir solo el encolado.
    input_provider(i): devuelve la entrada de la i-esima iteracion.
    sync_fn(): opcional; se llama tras infer_fn para forzar la sincronizacion
        del dispositivo antes de detener el cronometro.
    """
    # Calentamiento: se descartan estas inferencias hasta estabilizar
    # frecuencias y temperatura. No se miden.
    for i in range(warmup):
        infer_fn(input_provider(i))
    if sync_fn:
        sync_fn()

    latencies_ms = []
    for i in range(iters):
        x = input_provider(i)
        t0 = time.perf_counter_ns()
        infer_fn(x)
        if sync_fn:
            sync_fn()
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)
    return latencies_ms
