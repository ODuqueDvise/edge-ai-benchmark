"""Carga del conjunto de prueba y precision (clasificacion ImageNet).

Preprocesamiento IDENTICO al de MobileNetV2 (torchvision): redimensionar el lado
corto a 256, recorte central 224, normalizar con media/desv de ImageNet. Sin torch:
solo PIL + numpy, para que corra liviano en Jetson y RPi.

Formato del dataset esperado (ImageNet-V2):
    <root>/<indice_de_clase>/imagen.jpg
con subcarpetas nombradas por el indice de clase entero (0..999).

El iterador INTERCALA clases (round-robin): la i-esima imagen de cada clase, en
rondas. Asi `--limit N` toma una muestra REPRESENTATIVA (repartida entre clases),
no las primeras clases en bloque. El orden es determinista, asi que dos equipos con
el mismo `--limit` miden exactamente las mismas imagenes.
"""
from __future__ import annotations
import os
import numpy as np

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
_EXTS = (".jpg", ".jpeg", ".png")


def synthetic_input(shape, dtype=np.float32, seed=0):
    """Entrada sintetica para medir SOLO latencia (no precision)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape).astype(dtype)


def preprocess_image(path, resize=256, crop=224):
    """Devuelve un tensor [1,3,crop,crop] float32 con el preprocesamiento de MobileNetV2."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = resize, int(round(h * resize / w))
    else:
        nw, nh = int(round(w * resize / h)), resize
    img = img.resize((nw, nh), Image.BILINEAR)
    left = (nw - crop) // 2
    top = (nh - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))
    x = np.asarray(img, dtype=np.float32) / 255.0   # HWC
    x = x.transpose(2, 0, 1)                          # CHW
    x = (x - _MEAN) / _STD
    return x[np.newaxis, ...].astype(np.float32)      # [1,3,crop,crop]


def _index(root):
    """Lista [(label, dir, [archivos ordenados])] por clase, en orden determinista."""
    names = [d for d in os.listdir(root)
             if os.path.isdir(os.path.join(root, d)) and d.isdigit()]
    out = []
    for cls in sorted(names, key=int):
        d = os.path.join(root, cls)
        files = sorted(fn for fn in os.listdir(d) if fn.lower().endswith(_EXTS))
        if files:
            out.append((int(cls), d, files))
    return out


def iter_imagefolder(root, limit=None):
    """Itera (entrada [1,3,224,224], etiqueta:int) INTERCALANDO clases (round-robin).

    Ronda i: la i-esima imagen de cada clase. Con `--limit` la muestra queda
    repartida entre todas las clases (representativa) y es determinista.
    """
    per_class = _index(root)
    if not per_class:
        return
    maxlen = max(len(files) for _, _, files in per_class)
    count = 0
    for i in range(maxlen):
        for label, d, files in per_class:
            if i < len(files):
                yield preprocess_image(os.path.join(d, files[i])), label
                count += 1
                if limit and count >= limit:
                    return


def topk_hits(logits, label, k=5):
    """Devuelve (acierto_top1, acierto_topk) para una salida y su etiqueta."""
    order = np.argsort(np.asarray(logits).ravel())[::-1]
    return int(order[0] == label), int(label in order[:k])
