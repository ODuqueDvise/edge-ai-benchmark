# Análisis OE3 — GPU vs CPU a través de las técnicas (Jetson)

> Tendencia central = media geométrica (escala log). IC al 95% sobre las R corridas
> independientes (t de Student). Cola = percentiles de las muestras crudas combinadas.
> rpi-cpu pendiente (Luis). Generado por `scripts/analyze_oe3.py`.

## 1. Resumen por condición

| Modelo | Técnica | Disp. | R | Media geom. (ms) | p50 | p95 | p99 | top-1 | E. neta (mJ) |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| ResNet-50 | V0 | GPU | 5 | 6.586 | 6.588 | 6.672 | 6.736 | 0.694 | 43.1 |
| ResNet-50 | V0 | CPU | 5 | 90.094 | 89.854 | 93.970 | 95.701 | 0.695 | 370.0 |
| ResNet-50 | INT8 | GPU | 5 | 2.551 | 2.451 | 3.228 | 4.048 | 0.683 | 8.4 |
| ResNet-50 | INT8 | CPU | 5 | 36.013 | 35.748 | 37.157 | 40.358 | 0.674 | 161.7 |
| ResNet-50 | Poda | GPU | 5 | 5.113 | 5.115 | 5.190 | 5.353 | 0.510 | 27.6 |
| ResNet-50 | Poda | CPU | 5 | 47.905 | 47.539 | 50.103 | 53.242 | 0.511 | 202.4 |
| ResNet-50 | Poda+KD | GPU | 1 | 5.107 | 5.099 | 5.168 | 5.250 | 0.579 | — |
| MobileNetV2 | V0 | GPU | 5 | 2.552 | 2.472 | 2.842 | 3.776 | 0.596 | 12.1 |
| MobileNetV2 | V0 | CPU | 5 | 12.611 | 12.329 | 14.300 | 16.250 | 0.596 | 52.3 |
| MobileNetV2 | INT8 | GPU | 5 | 1.857 | 1.805 | 2.544 | 2.591 | 0.591 | 3.0 |
| MobileNetV2 | INT8 | CPU | 5 | 12.509 | 12.349 | 13.584 | 14.218 | 0.589 | 48.3 |
| MobileNetV2 | Poda | GPU | 5 | 2.466 | 2.345 | 3.170 | 3.842 | 0.453 | 7.1 |
| MobileNetV2 | Poda | CPU | 5 | 8.796 | 8.619 | 10.185 | 10.909 | 0.454 | 37.4 |
| MobileNetV2 | Poda+KD | GPU | 1 | 2.475 | 2.344 | 3.185 | 3.848 | 0.508 | — |

## 2. Brecha GPU↔CPU por técnica (tamaño de efecto e IC95)

> Razón = latencia(CPU) / latencia(GPU). Cuánto más rápida es la GPU.

| Modelo | Técnica | Brecha GPU↔CPU | IC95 |
|---|---|--:|--:|
| ResNet-50 | V0 | 13.68× | [13.63, 13.73] |
| ResNet-50 | INT8 | 14.12× | [14.03, 14.20] |
| ResNet-50 | Poda | 9.37× | [9.33, 9.41] |
| MobileNetV2 | V0 | 4.94× | [4.58, 5.33] |
| MobileNetV2 | INT8 | 6.74× | [6.68, 6.79] |
| MobileNetV2 | Poda | 3.57× | [3.55, 3.59] |

## 3. Efecto de cada técnica sobre la latencia (vs V0), por dispositivo

> Razón = latencia(V0) / latencia(técnica). >1 acelera; <1 frena.

| Modelo | Disp. | Técnica | Aceleración vs V0 | IC95 |
|---|---|---|--:|--:|
| ResNet-50 | GPU | INT8 | 2.58× | [2.57, 2.59] |
| ResNet-50 | GPU | Poda | 1.29× | [1.28, 1.29] |
| ResNet-50 | CPU | INT8 | 2.50× | [2.49, 2.52] |
| ResNet-50 | CPU | Poda | 1.88× | [1.87, 1.89] |
| MobileNetV2 | GPU | INT8 | 1.37× | [1.27, 1.48] |
| MobileNetV2 | GPU | Poda | 1.03× | [0.96, 1.12] |
| MobileNetV2 | CPU | INT8 | 1.01× | [1.00, 1.02] |
| MobileNetV2 | CPU | Poda | 1.43× | [1.42, 1.44] |

## 4. Precisión (ImageNet-V2, top-1) por técnica

| Modelo | V0 | INT8 | Poda (FT) | Poda+KD |
|---|--:|--:|--:|--:|
| ResNet-50 | 0.694 | 0.683 | 0.510 | 0.579 |
| MobileNetV2 | 0.596 | 0.591 | 0.453 | 0.508 |

---
CSV ordenado por corrida para ART/R: `results/oe3_tidy_runs.csv` (62 filas).
Figura de la brecha GPU↔CPU: `results/oe3_brecha_gpu_cpu.png` (+ .pdf).
