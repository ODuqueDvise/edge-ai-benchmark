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

## Modelos canónicos

Dos modelos comprometidos (decisión del director, jun 2026; ver `docs/DECISIONS.md` D9):

| Modelo | Archivo | Rol en el contraste | SHA-256 |
|---|---|---|---|
| MobileNetV2 | `models/cnn_baseline.onnx` | modelo ya eficiente (poco margen) | `609015cbb6ed30c7c456a2911a79bd2d303953e269a2d901da138dfcd56eb0dd` |
| ResNet-50 | `models/resnet50_baseline.onnx` | modelo denso con margen | `05e5bc14444e89b9b47b36c663bc40e061db8d20389d833dcde3c7da667290dc` |

Ambos: ONNX autocontenido (opset 18), entrada `1,3,224,224`, preprocesamiento ImageNet.
Se comparten por archivo y se verifican por checksum; **no se reexportan por equipo**.
Para regenerarlos (en el equipo de exportación):

```bash
python scripts/export_model.py --model-name mobilenet_v2 --output models/cnn_baseline.onnx --opset 18
python scripts/export_model.py --model-name resnet50     --output models/resnet50_baseline.onnx --opset 18
```

> **Distribución.** MobileNetV2 (~14 MB) está versionado en el repo. ResNet-50 (~100 MB)
> **no** se versiona (supera el límite de 100 MB por archivo de GitHub): se comparte por
> archivo —`scp` en la LAN, o un *release* de GitHub / unidad compartida— y se verifica por checksum.

Los resultados quedan etiquetados por modelo: el nombre del JSON lo incluye
(`<condición>_<modelo>_<backend>_<proveedor>_<fecha>.json`) y `RESULTS_LOG.md`
separa las filas por modelo automáticamente.

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

## Medición automatizada (un comando, desde el Mac)

`measure_remote.py` orquesta una condición de punta a punta por SSH: chequeos (reloj NTP,
checksum del modelo, autotest del medidor) → logger de energía local → latencia remota (R) →
la Jetson commitea sus JSON → el Mac hace `pull` → guardia de proveedor (aborta si la GPU cae
a CPU) → energía → commit. Aborta en rojo ante cualquier anomalía; un final verde significa
"medido bien", no "corrió sin reventar". Los JSON de dispositivo viajan por **git** (no rsync).

**Prerrequisitos (una vez):** SSH por llave (`ssh-copy-id usuario@host`, sin contraseña) y
relojes sincronizados por NTP en ambas máquinas. El medidor INA226+CP2112 va en el Mac.

```bash
# Una condición (jetson-gpu) con energía y verificación de modelo
python3 scripts/measure_remote.py --host orlando@orlando-desktop.local \
    --device-tag jetson-gpu --provider tensorrt --model models/resnet50_baseline.onnx \
    --shunt 0.1 --expect-sha 05e5bc14444e89b9b47b36c663bc40e061db8d20389d833dcde3c7da667290dc

# Las dos condiciones de la Jetson de un modelo, de un tiro
bash scripts/measure_jetson_model.sh models/resnet50_baseline.onnx 05e5bc14444e89b9b47b36c663bc40e061db8d20389d833dcde3c7da667290dc

# Ver el plan sin ejecutar nada
python3 scripts/measure_remote.py --host … --device-tag jetson-cpu --provider cpu \
    --model models/resnet50_baseline.onnx --dry-run
```

Sin `--shunt` solo mide latencia (no energía). Con `--accuracy` añade precisión (lenta, sin
logger). Los comandos manuales de arriba quedan como respaldo.

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
  fetch_results.sh        trae los *.json de un equipo (Jetson/RPi) al Mac por SSH (respaldo)
  measure_remote.py       orquestador: una condición de punta a punta por SSH (Mac)
  measure_jetson_model.sh wrapper: ambas condiciones de la Jetson de un modelo
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
models/cnn_baseline.onnx      baseline canónico 1 (MobileNetV2)
models/resnet50_baseline.onnx baseline 2 (ResNet-50) — LOCAL, no versionado (>100MB; scp+checksum)
results/                  resultados crudos (*.json, versionados); RESULTS_LOG.md se genera y NO se versiona
```

## Registro y trazabilidad

El estado y las decisiones se registran en puntos de chequeo definidos; ver
`docs/REGISTRO.md`. Qué se versiona: código, guías, `results/*.json` y el modelo
canónico. Qué **no**: `datasets/` (cada equipo descarga el suyo), `.venv/`,
`__pycache__/`, y `results/RESULTS_LOG.md` (es generado; se reconstruye con
`build_results_log.py`).

## Estado

- Línea base V0 completa en la **Jetson** (`jetson-gpu`, `jetson-cpu`) con MobileNetV2:
  la GPU es ~5x más rápida que la CPU a igual precisión y ~4.3x menos energía por inferencia.
- Alcance confirmado por el director (jun 2026): **dos modelos** (MobileNetV2 + ResNet-50) y
  técnicas en orden **INT8 → poda estructurada → destilación**. Ver `docs/DECISIONS.md` D9–D12.
- Pendiente: exportar **ResNet-50** y su línea base; condición **`rpi-cpu`** (Luis);
  **energía** en jetson y rpi; **Fase 2 / OE1** (optimización).

## Mantenimiento de la documentación

Cualquier cambio en la **estructura del proyecto** o en los **comandos** debe reflejarse
en este README y en las guías de `docs/`, en el mismo commit del cambio. La documentación
desactualizada es peor que no tenerla.
