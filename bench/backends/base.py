from __future__ import annotations
import abc


class Backend(abc.ABC):
    """Contrato de un backend de inferencia.

    Cada condicion experimental (jetson-gpu, jetson-cpu, rpi-cpu) se obtiene
    eligiendo backend + proveedor, manteniendo el resto del arnes identico.
    """
    name = "base"

    @abc.abstractmethod
    def load(self, model_path, **kw):
        ...

    @abc.abstractmethod
    def infer(self, x):
        """Una inferencia. Debe ser bloqueante (sincronizar si es asincrona)."""
        ...

    def synchronize(self):
        """Forzar sincronizacion del dispositivo. No-op si el runtime es sincrono."""
        return None

    @property
    def version(self):
        return None

    @property
    def input_name(self):
        return None
