# edge-ai-benchmark

Banco de medición para el proyecto de grado *"Evaluación de técnicas de optimización
de modelos de aprendizaje automático y del impacto de la aceleración por GPU frente a
la ejecución en CPU en dispositivos de cómputo de borde"* (Maestría en IA, Universidad
de La Salle).

Mide **latencia**, **precisión** y **consumo** de un modelo de inferencia de forma
idéntica en cada condición experimental, para que las comparaciones sean válidas,
reproducibles y trazables.

## Inicio rápido por rol

- **Visión completa del proceso:** `docs/RUNBOOK.md` (orden de todo, de principio a fin).
- **Jetson Orin Nano (Orlando):** `docs/QUICKSTART_JETSON.md`
- **Raspberry Pi 5 (Luis):** `docs/QUICKSTART_RPI.md`
- **Precisión (descarga del dataset y corridas):** `docs/QUICKSTART_ACCURACY.md`
- **Energía (INA226 + CP2112):** `docs/POWER_MEASUREMENT.md`

## Condiciones experimentales

Una sola base de código cubre las tres condiciones, eligiendo backend y proveedor:

| Condición    | Equipo            | Acelerador | Backend / proveedor             |
|--------------|-------------------|------------|---------------------------------|
| `jetson-gpu` | Jetson Orin Nano  | GPU        | ONNX Runtime + TensorRT/CUDA EP |
| `jetson-cpu` | Jetson Orin Nano  | CPU        | ONNX Runtime CPU EP             |
| `rpi-cpu`    | Raspberry Pi 5    | CPU        | ONNX Runtime CPU EP             |

> **`jetson-cpu` no es opcional.** Aísla el aporte de la GPU sobre el mismo SoC y RAM.
> `jetson-gpu` vs `rpi-cpu` mezcla acelerador con microarquitectura de CPU; es comparación de dispositivo.

## Modelo canónico

MobileNetV2 preentrenado, ONNX autocontenido en `models/cnn_baseline.onnx`.
SHA-256: `609015cbb6ed30c7c456a2911a79bd2d303953e269a2d901da138dfcd56eb0dd`.
Se comparte por archivo y se verifica por checksum; **no se reexporta por equipo**.
Para regenerarlo: `python scripts/export_model.py --model-name mobilenet_v2 --output models/cnn_baseline.onnx`.

## Constantes congeladas del protocolo

Calentamiento (warmup) **100** · inferencias por serie (M) **2000** · ejecuciones independientes (R) **5** ·
modo de potencia **MAXN** (Jetson) / gobernador **performance** (RPi) · entrada **1,3,224,224** ·
precisión sobre el set **completo (10.000)**. No se cambian sin consenso (ver `docs/DECISIONS.md`).

## Flujo de medición (comandos)

El detalle por equipo está en los quickstarts. Resumen:

```bash
# Preparación (ver QUICKSTART del equipo): venv, runtime, fijar desempeño, collect_env.sh
bash scripts/collect_env.sh

# Latencia (R=5 por condición; ejemplo jetson-gpu)
python -m bench.run_benchmark --model models/cnn_baseline.onnx --backend ort \
    --provider tensorrt --device-tag jetson-gpu --input-shape 1,3,224,224 --warmup 100 --iters 2000 --power-mode MAXN

# Precisión (set completo)
python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag jetson-cpu --dataset datasets/imagenetv2-matched-frequency-format-val

# Energía: registrar con el logger INA226 durante la corrida y calcular sobre la ventana (ver POWER_MEASUREMENT.md)
python scripts/ina226_cp2112_logger.py --rshunt 0.002 --out power_log.csv     # en el host de registro (Mac/Windows)
python scripts/energy_from_window.py --log power_log.csv --result results/<corrida>.json

# Analizar variabilidad entre corridas (decidir/verificar R)
python scripts/analyze_runs.py "results/jetson-gpu_*tensorrt*.json"

# Traer los *.json de un equipo de medicion al Mac (para calcular energia o consolidar)
bash scripts/fetch_results.sh orlando@orlando-desktop.local

# Consolidar y subir resultados (pull + regenerar log + commit + push), en una orden
bash scripts/sync_results.sh
```

## Estructura del proyecto

```
bench/
  harness.py              bucle de latencia (calentamiento, medición)
  metrics.py              percentiles e integración de energía
  metadata.py             metadatos por corrida (versiones, modo de potencia, checksum, temperaturas)
  power.py                muestreo interno de la Jetson (referencia cruzada)
  datasets.py             preprocesamiento ImageNet y carga del conjunto (muestreo intercalado)
  backends/               ONNX Runtime (CPU/CUDA/TensorRT) y TFLite
  run_benchmark.py        CLI de latencia  -> results/*.json
  run_accuracy.py         CLI de precisión -> results/acc_*.json
scripts/
  export_model.py         exporta el modelo base a un ONNX autocontenido + checksum
  collect_env.sh          vuelca versiones y estado del equipo
  ina226_logger.py        logger de energía por I2C nativo (Linux / Pi auxiliar)
  ina226_cp2112_logger.py logger de energía vía CP2112 (Mac/Windows)
  energy_from_window.py   integra energía sobre la ventana y la guarda en results/energy_*.json
  analyze_runs.py         resume corridas y variabilidad entre ellas (decidir R)
  build_results_log.py    genera results/RESULTS_LOG.md desde los JSON
  sync_results.sh         pull + regenerar log + add + commit + push (Jetson/RPi)
  fetch_results.sh        trae los *.json de un equipo (Jetson/RPi) al Mac por SSH
docs/
  RUNBOOK.md              proceso end-to-end + flujo de git
  QUICKSTART_JETSON.md    puesta a punto de la Jetson
  QUICKSTART_RPI.md       puesta a punto de la RPi 5
  QUICKSTART_ACCURACY.md  descarga del dataset y medición de precisión
  POWER_MEASUREMENT.md    medición de energía (INA226; nativo y CP2112)
  REGISTRO.md             procedimiento de registro y puntos de chequeo
  DECISIONS.md            registro de decisiones (ADR breve)
  BITACORA.md             bitácora cronológica de estado
config/example.yaml       plantilla de configuración por condición
models/cnn_baseline.onnx  modelo canónico (MobileNetV2)
results/                  resultados crudos (*.json, versionados); RESULTS_LOG.md se genera y NO se versiona
```

## Registro y trazabilidad

El estado y las decisiones se registran en puntos de chequeo definidos; ver
`docs/REGISTRO.md`. Qué se versiona: código, guías, `results/*.json` y el modelo
canónico. Qué **no**: `datasets/` (cada equipo descarga el suyo), `.venv/`,
`__pycache__/`, y `results/RESULTS_LOG.md` (es generado; se reconstruye con
`build_results_log.py`).

## Estado

- Núcleo del arnés probado; validado de punta a punta en la **Jetson** (`jetson-gpu`, `jetson-cpu`)
  con el modelo canónico. Hallazgo de línea base: la GPU es ~5x más rápida que la CPU sin pérdida de precisión.
- Pendiente: condición **`rpi-cpu`** (Luis), medición de **energía** con medidor externo,
  y la **Fase 2 / OE1** (aplicar las técnicas de optimización).

## Mantenimiento de la documentación

Cualquier cambio en la **estructura del proyecto** o en los **comandos** debe reflejarse
en este README y en las guías de `docs/`, en el mismo commit del cambio. La documentación
desactualizada es peor que no tenerla.
