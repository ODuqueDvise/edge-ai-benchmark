# Descargar el subconjunto de ImageNet de entrenamiento (para el reentrenamiento de la poda)

Guía para bajar un subconjunto **balanceado por clase** de ImageNet-1k de entrenamiento, que
servirá para el *fine-tuning* de recuperación de los modelos podados (segunda técnica del OE1).
Se corre **en el Legion, dentro de WSL/Ubuntu**, con el entorno `prune-env` activado, y se
guarda en el sistema de archivos de Linux (ext4) para que el entrenamiento lea rápido.

Por qué balanceado: el reentrenamiento debe ver todas las clases por igual; tomar "las primeras
N imágenes" daría un conjunto sesgado. Este procedimiento guarda N imágenes de **cada una** de
las 1000 clases.

## 0. Antes de empezar
- Legion con WSL2 + CUDA ya listo (ver `SETUP_LEGION_CUDA.md`) y `prune-env` activado.
- Acceso a ImageNet-1k en Hugging Face. Ya aceptaste los términos para el conjunto de
  calibración, así que tu cuenta tiene acceso; falta autenticar **esta máquina**.

## 1. Dependencias y autenticación

```bash
source ~/prune-env/bin/activate
pip install -U datasets huggingface_hub pillow
huggingface-cli login          # pega tu token de https://huggingface.co/settings/tokens (rol read)
```

**Qué deberías ver:** `Login successful`.

## 2. Descargar el subconjunto

`PER_CLASS=200` (~200 imágenes × 1000 clases ≈ 200 000 imágenes, ~20–30 GB) es un buen punto de
partida. Si quieres ir más liviano para una primera prueba, baja a 100; si quieres mejor
recuperación, sube a 300.

```bash
python - << 'PY'
from datasets import load_dataset
import os

OUT = os.path.expanduser("~/imagenet_train_subset")
PER_CLASS = 200
os.makedirs(OUT, exist_ok=True)

ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True).shuffle(seed=42, buffer_size=10000)
counts, full, saved = {}, 0, 0
for ex in ds:
    lab = ex["label"]                                  # 0..999
    if counts.get(lab, 0) >= PER_CLASS:
        continue
    d = os.path.join(OUT, "%04d" % lab)                # carpeta por clase, con ceros a la izquierda
    os.makedirs(d, exist_ok=True)
    n = counts.get(lab, 0)
    ex["image"].convert("RGB").save(os.path.join(d, "%04d_%04d.jpg" % (lab, n)))
    counts[lab] = n + 1
    saved += 1
    if counts[lab] == PER_CLASS:
        full += 1
    if saved % 2000 == 0:
        print("guardadas %d | clases completas %d/1000" % (saved, full))
    if full == 1000:
        break
print("LISTO: %d imágenes en %d clases en %s" % (saved, len(counts), OUT))
PY
```

**Expectativa honesta:** esto **transmite (stream) una porción grande** de ImageNet de
entrenamiento para juntar las 200 de cada clase —descarga más de lo que guarda—, así que tarda:
de un par de horas a toda la noche, según tu conexión. Déjalo corriendo. Si se corta, vuelve a
lanzarlo: las clases ya completas se saltan, pero como el orden de stream cambia, lo más simple
si se interrumpe es borrar `~/imagenet_train_subset` y reiniciar con calma de noche.

## 3. Verificar

```bash
# nº de clases (debe ser 1000) y total de imágenes (~200 000 con PER_CLASS=200)
find ~/imagenet_train_subset -mindepth 1 -maxdepth 1 -type d | wc -l
find ~/imagenet_train_subset -type f -iname '*.jpg' | wc -l
du -sh ~/imagenet_train_subset
```

**Qué deberías ver:** `1000` carpetas, ~`200000` archivos, ~`20–30 GB`.

## 4. Una nota importante para después (no te bloquea ahora)

Las carpetas se nombran por el índice de clase de Hugging Face (`0000`…`0999`). Cuando armemos
el flujo de poda, hay que **confirmar que ese índice coincide con el orden de clases que esperan
los modelos de torchvision** —es el error clásico de "etiquetas cruzadas"—. La verificación es
barata: evaluar el modelo preentrenado sobre este subconjunto debe dar una exactitud sana
(cercana a la conocida); si sale cerca de lo aleatorio, hay que reordenar el mapeo. Eso lo
resolvemos al construir el pipeline, no hace falta hoy.

---

Cuando termine, pásame lo que reporte el paso 3 (clases, total, tamaño). Con el subconjunto en
disco y el Legion listo, el siguiente paso es el flujo de **poda estructurada** (torch-pruning /
DepGraph) + reentrenamiento de recuperación + export a ONNX, y de ahí la medición con el
orquestador.
