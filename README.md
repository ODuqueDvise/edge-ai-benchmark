# edge-ai-benchmark

Arnés de medición para el proyecto de grado *"Evaluación de técnicas de
optimización de modelos de aprendizaje automático y del impacto de la aceleración
por GPU frente a la ejecución en CPU en dispositivos de cómputo de borde"*
(Maestría en IA, Universidad de La Salle).

El arnés mide **latencia**, **precisión** y **consumo** de un modelo de inferencia
de forma idéntica en cada condición experimental, para que las comparaciones sean
válidas, reproducibles y trazables. Implementa el protocolo de medición del proyecto.

## Condiciones experimentales

Una sola base de código cubre las tres condiciones, eligiendo backend y proveedor:

| Condición    | Equipo            | Acelerador | Backend / proveedor                 |
|--------------|-------------------|------------|-------------------------------------|
| `jetson-gpu` | Jetson Orin Nano  | GPU        | ONNX Runtime + TensorRT/CUDA EP     |
| `jetson-cpu` | Jetson Orin Nano  | CPU        | ONNX Runtime CPU EP (o TFLite)      |
| `rpi-cpu`    | Raspberry Pi 5    | CPU        | ONNX Runtime CPU EP (o TFLite)      |

> **`jetson-cpu` no es opcional.** Es la condición que aísla el aporte de la GPU
> sobre el mismo SoC y la misma RAM. La comparación `jetson-gpu` vs `rpi-cpu`
> mezcla acelerador con microarquitectura de CPU; úsala solo como comparación de
> dispositivo, no como medida del aporte de la GPU. Esta condición debe figurar
> como tarea explícita en el plan de trabajo (hoy solo aparece en el riesgo R1).

## Uso

```bash
pip install -r requirements.txt        # + el runtime de cada equipo
bash scripts/collect_env.sh            # congela versiones del equipo en un archivo

# Jetson, GPU
python -m bench.run_benchmark --model models/cnn.onnx --backend ort \
    --provider tensorrt --device-tag jetson-gpu --input-shape 1,3,224,224 \
    --warmup 50 --iters 1000 --power-mode MAXN

# Jetson, CPU (misma máquina: aísla el acelerador)
python -m bench.run_benchmark --model models/cnn.onnx --backend ort \
    --provider cpu --device-tag jetson-cpu --input-shape 1,3,224,224

# Raspberry Pi 5, CPU
python -m bench.run_benchmark --model models/cnn.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu --input-shape 1,3,224,224
```

Cada corrida escribe un JSON en `results/` con: latencias **crudas** por
inferencia (p50/p90/p95/p99/media/desv. se recalculan de ahí), metadatos
completos (versión del stack, modo de potencia, temperaturas inicio/fin, checksum
del modelo, proveedores activos) y la serie de potencia interna de la Jetson como
referencia cruzada.

## Energía: advertencia de método

La fuente **primaria** de energía es un medidor de potencia externo en la línea de
alimentación, idéntico para Jetson y RPi (la RPi 5 no tiene telemetría de
potencia). Los sensores internos de la Jetson que registra el arnés son solo
**referencia cruzada** y **no deben usarse para comparar consumo entre equipos**.
Alinea la ventana del medidor externo con los campos `window.start_epoch_s` /
`window.end_epoch_s` del JSON.

## Equidad de runtimes

Centrar las tres condiciones en ONNX Runtime (cambiando solo el proveedor de
ejecución) reduce el sesgo de comparar un runtime optimizado en GPU contra uno
ingenuo en CPU. Si se usa TFLite como alternativa en CPU, documentarlo: es un
factor que afecta la comparación.

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
scripts/collect_env.sh  volcado de versiones del equipo
config/example.yaml     plantilla de configuración por condición
```

## Estado

Borrador inicial. El núcleo (métricas, bucle, metadatos) está probado; los
backends requieren una prueba de humo en cada equipo con un modelo real antes de
la campaña de medición. Pendiente de la metodología: modelo CNN base, técnicas de
optimización, dataset y métrica de precisión.
