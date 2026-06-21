# Quickstart — Raspberry Pi 5 (frente Luis, condicion rpi-cpu)

Objetivo: medir el MISMO modelo con el MISMO arnes que la Jetson, en CPU.
La RPi 5 no tiene GPU para ML, asi que solo corre la condicion `rpi-cpu`.

> ¿Primera vez? Sigue **`docs/GUIA_LUIS_RPI.md`**: el camino completo de cero a resultados
> (incluye git/SSH, dataset y precisión). Esta página es la referencia rápida de comandos.

## 0. Regla de coherencia (no la rompas)

- Usa los MISMOS archivos de modelo que Orlando, NO los reexportes. Dos baselines:
  `models/cnn_baseline.onnx` (MobileNetV2, SHA-256 `609015cb…56eb0dd`) y
  `models/resnet50_baseline.onnx` (ResNet-50, SHA-256 `05e5bc14…290dc`).
- Mismos parametros (constantes congeladas): `--input-shape 1,3,224,224 --warmup 100 --iters 2000`.

## 1. Clonar el repo y traer los modelos

```bash
sudo apt install -y git-lfs && git lfs install   # una sola vez por máquina: trae los .onnx reales, no punteros
git clone https://github.com/ODuqueDvise/edge-ai-benchmark.git
cd edge-ai-benchmark
# Los dos baselines vienen con el clon: todos los *.onnx se versionan vía Git LFS. Verifica los checksums:
sha256sum models/cnn_baseline.onnx models/resnet50_baseline.onnx
# Deben coincidir EXACTO con los publicados. Si no, no midas: git lfs install && git lfs pull.
```

## 2. Entorno y runtime de CPU

```bash
sudo apt update && sudo apt install -y python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install onnxruntime        # CPU; en RPi (aarch64) el wheel de PyPI sirve
```

## 3. Fijar desempeno de CPU (reproducibilidad)

```bash
# Gobernador en maximo desempeno
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Asegura refrigeracion activa: la RPi 5 hace throttling termico sin disipador/ventilador.
bash scripts/collect_env.sh    # congela versiones del equipo en un archivo
```

## 4. Linea base en CPU

```bash
# Repite por modelo cambiando --model (el nombre del JSON incluye el modelo).
python -m bench.run_benchmark --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu --input-shape 1,3,224,224 --warmup 100 --iters 2000
python -m bench.run_benchmark --model models/resnet50_baseline.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu --input-shape 1,3,224,224 --warmup 100 --iters 2000
```

Deja el JSON en `results/`. Subelo al repo (rama propia) para consolidar con los de la Jetson.

## Notas

- Energia: la RPi 5 no tiene telemetria de potencia. La energia se mide con el
  MISMO medidor externo que la Jetson, alineando la ventana con las marcas de
  tiempo del JSON (campos `window.start_epoch_s` / `window.end_epoch_s`).
- No actualices versiones a mitad de campana; registra el entorno con collect_env.sh.
