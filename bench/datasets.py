"""Carga del conjunto de prueba para medir precision (clasificacion).

Implementacion minima e intencionalmente generica: el conjunto y la metrica
concretos se definen en la metodologia del proyecto. Aqui solo se fija el
contrato: la precision se mide sobre el MISMO conjunto en todas las condiciones.
"""
from __future__ import annotations
import numpy as np


def synthetic_input(shape, dtype=np.float32, seed=0):
    """Entrada sintetica para medir SOLO latencia (no precision)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape).astype(dtype)


def iter_labeled_dataset(path):
    """Itera (entrada, etiqueta) del conjunto de prueba.

    A implementar segun el dataset elegido en la metodologia. Debe devolver las
    entradas ya preprocesadas, en el mismo orden para todas las condiciones.
    """
    raise NotImplementedError(
        "Implementar la carga del dataset elegido en la metodologia. "
        "Mantener el mismo conjunto y orden en todas las condiciones.")


def top1_accuracy(pred_logits, labels):
    pred = np.argmax(np.asarray(pred_logits), axis=-1)
    return float((pred == np.asarray(labels)).mean())
