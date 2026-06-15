# Registro de resultados

Una fila por corrida. Los JSON crudos en `results/` son la fuente; esta tabla es el
resumen legible. Modelo base (V0) SHA-256: `609015cb...0dd`. Dataset: ImageNet-V2
matched-frequency (10000 img). Latencia: 1000 inf. Precisión: indicar nº de imágenes.

## Latencia (línea base V0)

| Fecha (UTC) | Condición  | p50 (ms) | p95 (ms) | thr (ips) | Notas |
|-------------|------------|----------|----------|-----------|-------|
| 2026-06-15  | jetson-gpu | 2.49     | 2.53     | 401       | validación (no oficial) |
| 2026-06-15  | jetson-cpu | 12.26    | 14.25    | 80        | validación (no oficial) |
|             | rpi-cpu    |          |          |           | pendiente (Luis) |

## Precisión (línea base V0, ImageNet-V2)

| Fecha (UTC) | Condición  | top-1  | top-5  | n img | Notas |
|-------------|------------|--------|--------|-------|-------|
| 2026-06-15  | jetson-cpu | 0.6070 | 0.8175 | 2000  | subconjunto representativo (verificación); falta set completo |
|             | jetson-gpu |        |        |       | pendiente (set completo) |
|             | jetson-cpu |        |        | 10000 | pendiente (set completo, oficial) |
|             | rpi-cpu    |        |        |       | pendiente (Luis) |

## Energía (línea base V0)

| Fecha (UTC) | Condición | Energía/inf (mJ) | Pot. media (W) | Pot. reposo (W) | Notas |
|-------------|-----------|------------------|----------------|-----------------|-------|
|             | jetson-gpu |                 |                |                 | pendiente (INA226+CP2112) |
|             | jetson-cpu |                 |                |                 | pendiente |
|             | rpi-cpu    |                 |                |                 | pendiente (Luis) |
