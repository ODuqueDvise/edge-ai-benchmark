# Registro de resultados

Una fila por corrida. Los JSON crudos en `results/` son la fuente; esta tabla es el
resumen legible. Modelo base (V0) SHA-256: `609015cb...0dd`. Dataset: ImageNet-V2
matched-frequency (10000 img).

| Fecha (UTC) | Equipo/condición | Variante | top-1 | top-5 | Latencia p50 (ms) | Energía/inf (mJ) | n img | Notas |
|-------------|------------------|----------|-------|-------|-------------------|------------------|-------|-------|
| 2026-06-15  | jetson-gpu       | V0 base  |       |       | 2.49 (validación) |                  |       | corrida de validación, no oficial |
| 2026-06-15  | jetson-cpu       | V0 base  |       |       | 12.26 (validación)|                  |       | corrida de validación, no oficial |
|             | rpi-cpu          | V0 base  |       |       |                   |                  |       | pendiente (Luis) |
