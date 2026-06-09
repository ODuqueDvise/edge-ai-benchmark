"""Muestreo de potencia. REFERENCIA CRUZADA, no fuente primaria.

La fuente primaria de energia es un medidor externo en la linea de alimentacion,
identico para Jetson y Raspberry Pi (ver protocolo, seccion 6). La Raspberry Pi 5
no tiene telemetria de potencia; por eso estas muestras internas de la Jetson NO
deben usarse para comparar consumo entre equipos, solo para validar el orden de
magnitud dentro de la Jetson.

El muestreador lee tegrastats en segundo plano y extrae VDD_IN (potencia de
entrada del modulo). En equipos sin tegrastats queda inerte.
"""
from __future__ import annotations
import re, shutil, subprocess, threading, time

_VDD_IN = re.compile(r"VDD_IN\s+(\d+)mW")


class JetsonPowerSampler:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.samples = []   # (epoch_s, watts)
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self.available = shutil.which("tegrastats") is not None

    def _reader(self):
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = _VDD_IN.search(line)
            if m:
                self.samples.append((time.time(), int(m.group(1)) / 1000.0))

    def __enter__(self):
        if self.available:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=2)

    def series(self):
        if not self.samples:
            return [], []
        ts, ws = zip(*self.samples)
        return list(ts), list(ws)
