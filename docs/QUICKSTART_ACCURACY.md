# Quickstart — Medición de precisión (ImageNet-V2)

Mide top-1 / top-5 del modelo en cada condición, para el OE1 (pérdida de precisión
por optimización). Replicable por Orlando (Jetson) y Luis (RPi) con los mismos pasos.

## 0. Reglas de coherencia (no romper)

- **Mismo dataset:** variante *matched-frequency* (10.000 imágenes), idéntica en ambos equipos.
- **Mismos modelos:** los dos baselines, cada uno verificado por su checksum —
  MobileNetV2 `models/cnn_baseline.onnx` (`609015cb…56eb0dd`) y ResNet-50
  `models/resnet50_baseline.onnx` (checksum que imprime la exportación).
- **Mismo subconjunto:** si usas `--limit`, ambos usan el mismo valor (el orden de
  iteración es determinista: se miden las mismas imágenes en el mismo orden).
- La precisión se mide por **(variante de modelo, condición)**: CPU FP32 y GPU
  TensorRT pueden dar precisión distinta, y eso es parte del resultado.

## 1. Descargar el dataset (una vez por equipo)

Hazlo **con el venv activado** (ahí viven pip y las herramientas):

```bash
cd ~/edge-ai-benchmark          # raiz del repo
source .venv/bin/activate
python -m pip install -U huggingface_hub
```

Descarga confirmando el nombre real del archivo (no lo adivines):

```bash
python - << 'PY'
from huggingface_hub import list_repo_files, hf_hub_download
files = list_repo_files("vaishaal/ImageNetV2", repo_type="dataset")
cand = [f for f in files if "matched-frequency" in f and f.endswith((".tar.gz", ".tgz", ".tar"))]
print("Candidatos:", cand)
p = hf_hub_download("vaishaal/ImageNetV2", cand[0], repo_type="dataset", local_dir="datasets")
print("DESCARGADO:", p)
PY
```

Extrae (usa `-xf`, GNU tar autodetecta el formato; `-xzf` falla si no es gzip):

```bash
tar -xf datasets/imagenetv2-matched-frequency*.tar* -C datasets
```

La carpeta resultante es **`datasets/imagenetv2-matched-frequency-format-val/`** con
subcarpetas 0..999. Verifica que está completa (debe dar **10000**):

```bash
find datasets/imagenetv2-matched-frequency-format-val -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
```

## 2. Entorno

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt    # incluye Pillow
bash scripts/collect_env.sh                   # congela versiones (guardalo, es parte de la doc)
```

## 3. Verificación de preprocesamiento (OBLIGATORIA antes de confiar)

```bash
python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag jetson-cpu \
    --dataset datasets/imagenetv2-matched-frequency-format-val --limit 2000
```

El top-1 debe caer en torno a 50-bajos / 60% (ImageNet-V2 es más difícil que la
validación clásica). **Si da algo cercano a lo aleatorio (~0.1%), el preprocesamiento
o las etiquetas están mal** — no es degradación real; detente y revísalo. Anota el
número base.

## 4. Medir precisión por condición (cifras finales: sin `--limit`)

```bash
# Jetson GPU (TensorRT)
python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
    --provider tensorrt --device-tag jetson-gpu \
    --dataset datasets/imagenetv2-matched-frequency-format-val

# Jetson CPU
python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag jetson-cpu \
    --dataset datasets/imagenetv2-matched-frequency-format-val

# Raspberry Pi 5 (Luis)
python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu \
    --dataset datasets/imagenetv2-matched-frequency-format-val
```

Repite los tres comandos para el segundo modelo cambiando `--model` a
`models/resnet50_baseline.onnx`. Cada corrida escribe
`results/acc_<condición>_<modelo>_..._<fecha>.json` con la precisión y los metadatos
(checksum del modelo, proveedores, dataset, nº de imágenes); el nombre incluye el modelo.

## 5. Documentar el proceso

1. Guarda la salida de `scripts/collect_env.sh` de cada equipo.
2. Registra cada corrida en `results/RESULTS_LOG.md` (una fila por medición).
3. Sube los JSON crudos y el log: `git add results/ && git commit -m "..." && git push`.
4. Anota la **versión del dataset** (variante + nº de imágenes) y el **checksum del
   modelo**; es lo que hace la medición auditable.

## 6. Replicación por Luis (RPi)

Mismos pasos. Antes de medir, dos verificaciones obligatorias:

```bash
sha256sum models/cnn_baseline.onnx models/resnet50_baseline.onnx   # deben coincidir con los publicados
find datasets/imagenetv2-matched-frequency-format-val -type f | wc -l   # debe dar 10000
```

Si ambas coinciden, la corrida de `rpi-cpu` es comparable con las de la Jetson.
