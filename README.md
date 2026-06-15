# edge-ai-benchmark

Arnés de medición para el proyecto de grado *"Evaluación de técnicas de
optimización de modelos de aprendizaje automático y del impacto de la aceleración
por GPU frente a la ejecución en CPU en dispositivos de cómputo de borde"*
(Maestría en IA, Universidad de La Salle).

Mide **latencia**, **precisión** y **consumo** de un modelo de inferencia de forma
idéntica en cada condición experimental, para que las comparaciones sean válidas,
reproducibles y trazables.

## Inicio rápido por rol

- **Jetson Orin Nano (Orlando):** `docs/QUICKSTART_JETSON.md`
- **Raspberry Pi 5 (Luis):** `docs/QUICKSTART_RPI.md`
- **Medición de energía (ambos):** `docs/POWER_MEASUREMENT.md`

## Condiciones experimentales

Una sola base de código cubre las tres condiciones, eligiendo backend y proveedor:

| Condición    | Equipo            | Acelerador | Backend / proveedor                 |
|--------------|-------------------|------------|-------------------------------------|
| `jetson-gpu` | Jetson Orin Nano  | GPU        | ONNX Runtime + TensorRT/CUDA EP     |
| `jetson-cpu` | Jetson Orin Nano  | CPU        | ONNX Runtime CPU EP                 |
| `rpi-cpu`    | Raspberry Pi 5    | CPU        | ONNX Runtime CPU EP                 |

> **`jetson-cpu` no es opcional.** Es la condición que aísla el aporte de la GPU
> sobre el mismo SoC y la misma RAM. La comparación `jetson-gpu` vs `rpi-cpu` mezcla
> acelerador con microarquitectura de CPU; úsala solo como comparación de dispositivo.

## Modelo canónico

El modelo de línea base es **MobileNetV2 preentrenado**, exportado a un ONNX
autocontenido en `models/cnn_baseline.onnx`.

- SHA-256: `609015cbb6ed30c7c456a2911a79bd2d303953e269a2d901da138dfcd56eb0dd`
- **Se comparte por archivo y se verifica por checksum; no se reexporta por equipo**
  (dos exportaciones no dan bytes idénticos).
- Para regenerarlo (requiere torch, torchvision, onnx, onnxruntime):
  `python scripts/export_model.py --model-name mobilenet_v2 --output models/cnn_baseline.onnx`

## Flujo de medición

1. **Obtener el modelo** (ya viene en el repo; verifica el checksum):
   `sha256sum models/cnn_baseline.onnx`
2. **Preparar el equipo** según el quickstart de tu rol (fijar desempeño, instalar
   runtime, registrar entorno con `scripts/collect_env.sh`).
3. **Correr el benchmark** por condición:

```bash
# Jetson, GPU
python -m bench.run_benchmark --model models/cnn_baseline.onnx --backend ort \
    --provider tensorrt --device-tag jetson-gpu --input-shape 1,3,224,224 --warmup 50 --iters 1000 --power-mode MAXN

# Jetson, CPU (misma máquina: aísla el acelerador)
python -m bench.run_benchmark --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag jetson-cpu --input-shape 1,3,224,224 --warmup 50 --iters 1000 --power-mode MAXN

# Raspberry Pi 5, CPU
python -m bench.run_benchmark --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu --input-shape 1,3,224,224 --warmup 50 --iters 1000
```

4. **Medir energía** (recomendado): registrar con el logger INA226 durante la
   corrida y calcular la energía sobre la ventana. Ver `docs/POWER_MEASUREMENT.md`.

Cada corrida escribe un JSON en `results/` con latencias **crudas** por inferencia
(p50/p90/p95/p99/media/desv. se recalculan de ahí), metadatos completos (versión del
stack, modo de potencia, temperaturas, checksum del modelo, proveedores activos) y
las marcas de tiempo de la ventana para alinear el medidor de energía.

## Energía: advertencia de método

La fuente **primaria** de energía es un medidor externo en la línea de alimentación,
idéntico para Jetson y RPi (la RPi 5 no tiene telemetría de potencia). Los sensores
internos de la Jetson son solo **referencia cruzada** y **no se usan para comparar
consumo entre equipos**. Alinea la ventana del medidor con `window.start_epoch_s` /
`window.end_epoch_s` del JSON usando `scripts/energy_from_window.py`.

## Equidad de runtimes

Centrar las tres condiciones en ONNX Runtime (cambiando solo el proveedor de
ejecución) reduce el sesgo de comparar un runtime optimizado en GPU contra uno
ingenuo en CPU.

## Estructura

```
bench/
  harness.py            bucle de latencia (calentamiento, medición)
  metrics.py            percentiles e integración de energía
  metadata.py           metadatos por corrida
  power.py              muestreo interno Jetson (referencia cruzada)
  datasets.py           contrato del conjunto de prueba (precisión)
  backends/             ONNX Runtime (CPU/CUDA/TensorRT) y TFLite
  run_benchmark.py      CLI -> results/*.json
scripts/
  export_model.py            exporta el modelo base a un ONNX autocontenido + checksum
  collect_env.sh             vuelca versiones/estado del equipo
  ina226_logger.py           logger de energía por I2C nativo (Linux, p.ej. Pi auxiliar)
  ina226_cp2112_logger.py    logger de energía vía CP2112 (Mac/Windows)
  energy_from_window.py      integra energía sobre la ventana de cada corrida
docs/
  QUICKSTART_JETSON.md       puesta a punto de la Jetson (jetson-gpu, jetson-cpu)
  QUICKSTART_RPI.md          puesta a punto de la RPi 5 (rpi-cpu)
  POWER_MEASUREMENT.md       esquema de medición de energía (INA226; nativo y CP2112)
config/example.yaml     plantilla de configuración por condición
models/cnn_baseline.onnx  modelo canónico (MobileNetV2)
```

## Estado

- Núcleo del arnés (métricas, bucle, metadatos): probado.
- Arnés validado de punta a punta en la **Jetson** (`jetson-gpu` y `jetson-cpu`) con
  el modelo canónico.
- Pendiente: condición **`rpi-cpu`** (Luis), aplicar las **técnicas de optimización**
  del OE1, y la medición de energía con **medidor externo** (logger INA226 + CP2112,
  validar con el autotest en cada equipo).
- Pendiente de metodología: técnicas de optimización, dataset y métrica de precisión.
