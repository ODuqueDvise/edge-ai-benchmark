# Quickstart — Medición de precisión (ImageNet-V2)

Mide top-1 / top-5 del modelo en cada condición, para el OE1 (pérdida de precisión
por optimización). Replicable por Orlando (Jetson) y Luis (RPi) con los mismos pasos.

## 0. Reglas de coherencia (no romper)

- **Mismo dataset:** variante *matched-frequency* (10.000 imágenes), idéntica en ambos equipos.
- **Mismo modelo:** `models/cnn_baseline.onnx`, SHA-256 `609015cbb6ed30c7c456a2911a79bd2d303953e269a2d901da138dfcd56eb0dd`.
- **Mismo subconjunto:** si usas `--limit`, ambos usan el mismo valor (el orden de
  iteración es determinista: carpetas y archivos ordenados, así que se miden las
  mismas imágenes en el mismo orden).
- La precisión se mide por **(variante de modelo, condición)**: la CPU FP32 y la GPU
  TensorRT pueden dar precisión distinta, y eso es parte del resultado.

## 1. Descargar el dataset (una vez por equipo)

ImageNet-V2 matched-frequency, desde HuggingFace `vaishaal/ImageNetV2`:

```bash
pip install -U huggingface_hub
huggingface-cli download vaishaal/ImageNetV2 imagenetv2-matched-frequency.tar.gz \
    --repo-type dataset --local-dir datasets
tar -xzf datasets/imagenetv2-matched-frequency.tar.gz -C datasets
# Resultado: datasets/imagenetv2-matched-frequency/  con subcarpetas 0..999
```

> Si el nombre exacto del archivo cambió, míralo en
> https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main y ajusta el comando.

Verifica que quedó completo (debe dar 10000):

```bash
find datasets/imagenetv2-matched-frequency -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
```

## 2. Entorno

```bash
source .venv/bin/activate           # el venv del equipo (ver QUICKSTART_JETSON / QUICKSTART_RPI)
pip install -r requirements.txt     # incluye Pillow para el preprocesamiento
bash scripts/collect_env.sh         # congela versiones del equipo (guardalo, es parte de la doc)
```

## 3. Verificación de preprocesamiento (OBLIGATORIA antes de confiar)

Corre la línea base en CPU sobre un subconjunto y mira el top-1:

```bash
python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag jetson-cpu \
    --dataset datasets/imagenetv2-matched-frequency --limit 2000
```

El top-1 debe caer en torno a 50-bajos / 60% (ImageNet-V2 es más difícil que la
validación clásica, por eso no verás el ~72% publicado). **Si da algo cercano a lo
aleatorio (~0.1%), el preprocesamiento o las etiquetas están mal** — no es
degradación real; detente y revísalo. Anota el número base obtenido.

## 4. Medir precisión por condición

Para las cifras finales, corre el set completo (sin `--limit`):

```bash
# Jetson GPU (TensorRT)
python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
    --provider tensorrt --device-tag jetson-gpu --dataset datasets/imagenetv2-matched-frequency

# Jetson CPU
python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag jetson-cpu --dataset datasets/imagenetv2-matched-frequency

# Raspberry Pi 5 (Luis)
python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu --dataset datasets/imagenetv2-matched-frequency
```

Cada corrida escribe `results/acc_<condición>_..._<fecha>.json` con la precisión y
los metadatos (checksum del modelo, proveedores activos, dataset, nº de imágenes).

## 5. Documentar el proceso

La traza ya queda casi automática: cada JSON de `results/` registra el checksum del
modelo, los proveedores y el dataset. Para cerrarlo:

1. Guarda la salida de `scripts/collect_env.sh` de cada equipo (versiones congeladas).
2. Registra cada corrida en `results/RESULTS_LOG.md` (una fila por medición).
3. Sube los JSON crudos y el log al repositorio en tu rama:
   `git add results/*.json results/RESULTS_LOG.md && git commit -m "..." && git push`.
4. Anota la **versión del dataset** (variante + nº de imágenes) y el **checksum del
   modelo** en el log; son lo que hace la medición auditable.

## 6. Replicación por Luis (RPi)

Mismos pasos. Antes de medir, dos verificaciones obligatorias:

```bash
sha256sum models/cnn_baseline.onnx     # debe dar 609015cb...0dd
find datasets/imagenetv2-matched-frequency -type f | wc -l   # debe dar 10000
```

Si ambas coinciden, la corrida de `rpi-cpu` es directamente comparable con las de la
Jetson. Si no, no midas hasta igualarlas.
