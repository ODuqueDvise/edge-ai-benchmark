# Quickstart — Raspberry Pi 5 (frente Luis, condicion rpi-cpu)

Objetivo: medir el MISMO modelo con el MISMO arnes que la Jetson, en CPU.
La RPi 5 no tiene GPU para ML, asi que solo corre la condicion `rpi-cpu`.

## 0. Regla de coherencia (no la rompas)

- Usa el MISMO archivo de modelo que Orlando, NO lo reexportes.
  Modelo canonico: `models/cnn_baseline.onnx`, SHA-256:
  `609015cbb6ed30c7c456a2911a79bd2d303953e269a2d901da138dfcd56eb0dd`
- Mismos parametros: `--input-shape 1,3,224,224 --warmup 50 --iters 1000`.

## 1. Clonar el repo y traer el modelo

```bash
git clone https://github.com/ODuqueDvise/edge-ai-benchmark.git
cd edge-ai-benchmark
# El modelo viene en el repo (models/cnn_baseline.onnx). Verifica el checksum:
sha256sum models/cnn_baseline.onnx
# Debe coincidir EXACTO con el de arriba. Si no, no midas: pide el archivo a Orlando.
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
python -m bench.run_benchmark --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu --input-shape 1,3,224,224 --warmup 50 --iters 1000
```

Deja el JSON en `results/`. Subelo al repo (rama propia) para consolidar con los de la Jetson.

## Notas

- Energia: la RPi 5 no tiene telemetria de potencia. La energia se mide con el
  MISMO medidor externo que la Jetson, alineando la ventana con las marcas de
  tiempo del JSON (campos `window.start_epoch_s` / `window.end_epoch_s`).
- No actualices versiones a mitad de campana; registra el entorno con collect_env.sh.
