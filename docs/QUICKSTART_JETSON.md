# Quickstart — Jetson Orin Nano (mi frente)

Pasos para validar la cadena de exportación y dejar el arnés corriendo en las dos
condiciones de la Jetson (jetson-gpu y jetson-cpu). Ejecutar sobre JetPack 6.2.

## 1. Fijar el estado del equipo (reproducibilidad)

```bash
sudo nvpmodel -m 0          # modo de máximo desempeño (verificar el ID con: sudo nvpmodel -q)
sudo jetson_clocks          # fija frecuencias
bash scripts/collect_env.sh # guarda versiones, modo de potencia y temperaturas en un archivo
```

## 2. Entorno e instalación de runtimes

Una sola vez por máquina, instala Git LFS para que los `*.onnx` (incluido el que exportes
aquí) viajen como archivo real y no como puntero: `sudo apt install -y git-lfs && git lfs install`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CPU (condición jetson-cpu): onnxruntime de PyPI sirve en aarch64.
pip install onnxruntime

# GPU (condición jetson-gpu): NO uses el wheel de PyPI (es solo CPU en aarch64).
# Usa el wheel onnxruntime-gpu provisto por NVIDIA para tu JetPack (Jetson Zoo),
# con proveedores TensorRT/CUDA. Verifica con el bloque del paso 4.
```

> Para exportar el modelo necesitas torch y torchvision en el equipo donde exportes
> (puede ser la Jetson o tu portátil). El artefacto ONNX resultante se versiona en el
> repo vía Git LFS y llega con el clon; no se re-exporta en cada equipo.

## 3. Exportar el modelo base (una sola vez) y publicar su checksum

```bash
pip install torch torchvision        # solo donde exportes
python scripts/export_model.py --model-name mobilenet_v2 --output models/cnn_baseline.onnx --opset 18
python scripts/export_model.py --model-name resnet50     --output models/resnet50_baseline.onnx --opset 18
# Anota el SHA-256 que imprime cada uno: es el que Luis verifica antes de medir.
sha256sum models/cnn_baseline.onnx models/resnet50_baseline.onnx
```

## 4. Verificar que la GPU está activa (evitar caída silenciosa a CPU)

```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Debe listar TensorrtExecutionProvider y/o CUDAExecutionProvider para jetson-gpu.
```

## 5. Líneas base en las dos condiciones de la Jetson

Constantes congeladas: `--warmup 100 --iters 2000`. Repite cada bloque por modelo
cambiando `--model` (`models/cnn_baseline.onnx` y `models/resnet50_baseline.onnx`);
el nombre del JSON incluye el modelo, así que no se pisan.

```bash
# GPU (TensorRT EP)
python -m bench.run_benchmark --model models/cnn_baseline.onnx --backend ort \
    --provider tensorrt --device-tag jetson-gpu --input-shape 1,3,224,224 \
    --warmup 100 --iters 2000 --power-mode MAXN

# CPU en la MISMA Jetson (aísla el aporte de la GPU)
python -m bench.run_benchmark --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag jetson-cpu --input-shape 1,3,224,224 \
    --warmup 100 --iters 2000 --power-mode MAXN
```

Cada corrida deja un JSON en `results/`. Súbelos al repositorio. Si en el paso 4
no aparece el proveedor de GPU, el arnés además imprime una advertencia al cargar.
