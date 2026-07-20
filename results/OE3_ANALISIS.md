# Análisis OE3 — GPU vs CPU a través de las técnicas (Jetson y RPi 5)

> Tendencia central = media geométrica (escala log). IC al 95% sobre las R corridas
> independientes (t de Student). Cola = percentiles de las muestras crudas combinadas.
> rpi-cpu: campaña oficial fría y auditada de ENTORNO ÚNICO (kernel 7.0.0-1014-raspi);
> R alto en RPi absorbe la multimodalidad entre procesos (BITACORA 14/19 jul 2026).
> Generado por `scripts/analyze_oe3.py`.

## 1. Resumen por condición

| Modelo | Técnica | Disp. | R | Media geom. (ms) | p50 | p95 | p99 | top-1 | E. neta (mJ) |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| ResNet-50 | V0 | GPU | 5 | 6.586 | 6.588 | 6.672 | 6.736 | 0.694 | 43.1 |
| ResNet-50 | V0 | CPU | 5 | 90.094 | 89.854 | 93.970 | 95.701 | 0.695 | 370.0 |
| ResNet-50 | V0 | RPi-CPU | 10 | 151.433 | 151.557 | 156.070 | 198.119 | 0.695 | — |
| ResNet-50 | INT8 | GPU | 5 | 2.551 | 2.451 | 3.228 | 4.048 | 0.683 | 8.4 |
| ResNet-50 | INT8 | CPU | 5 | 36.013 | 35.748 | 37.157 | 40.358 | 0.674 | 161.7 |
| ResNet-50 | INT8 | RPi-CPU | 5 | 78.489 | 74.461 | 90.175 | 109.023 | 0.674 | — |
| ResNet-50 | Poda | GPU | 5 | 5.113 | 5.115 | 5.190 | 5.353 | 0.510 | 27.6 |
| ResNet-50 | Poda | CPU | 5 | 47.905 | 47.539 | 50.103 | 53.242 | 0.511 | 202.4 |
| ResNet-50 | Poda | RPi-CPU | 5 | 63.131 | 62.142 | 65.939 | 85.656 | 0.511 | — |
| ResNet-50 | Poda+KD | GPU | 1 | 5.107 | 5.099 | 5.168 | 5.250 | 0.579 | — |
| MobileNetV2 | V0 | GPU | 5 | 2.552 | 2.472 | 2.842 | 3.776 | 0.596 | 12.1 |
| MobileNetV2 | V0 | CPU | 5 | 12.611 | 12.329 | 14.300 | 16.250 | 0.596 | 52.3 |
| MobileNetV2 | V0 | RPi-CPU | 21 | 22.830 | 22.611 | 26.552 | 41.293 | 0.596 | — |
| MobileNetV2 | INT8 | GPU | 5 | 1.857 | 1.805 | 2.544 | 2.591 | 0.591 | 3.0 |
| MobileNetV2 | INT8 | CPU | 5 | 12.509 | 12.349 | 13.584 | 14.218 | 0.589 | 48.3 |
| MobileNetV2 | INT8 | RPi-CPU | 10 | 26.788 | 27.773 | 30.373 | 41.477 | 0.589 | — |
| MobileNetV2 | Poda | GPU | 5 | 2.466 | 2.345 | 3.170 | 3.842 | 0.453 | 7.1 |
| MobileNetV2 | Poda | CPU | 5 | 8.796 | 8.619 | 10.185 | 10.909 | 0.454 | 37.4 |
| MobileNetV2 | Poda | RPi-CPU | 10 | 11.472 | 11.094 | 12.690 | 15.739 | 0.454 | — |
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

### 2b. Brecha de despliegue Jetson-GPU ↔ RPi-CPU (tamaño de efecto e IC95)

> Razón = latencia(RPi-CPU) / latencia(Jetson-GPU): acelerador embebido frente a la CPU
> de placa. Referencia de despliegue entre plataformas (3ª columna de OE3).

| Modelo | Técnica | Brecha Jetson-GPU↔RPi-CPU | IC95 |
|---|---|--:|--:|
| ResNet-50 | V0 | 22.99× | [22.62, 23.37] |
| ResNet-50 | INT8 | 30.77× | [28.04, 33.76] |
| ResNet-50 | Poda | 12.35× | [12.07, 12.63] |
| MobileNetV2 | V0 | 8.94× | [8.27, 9.68] |
| MobileNetV2 | INT8 | 14.42× | [13.25, 15.69] |
| MobileNetV2 | Poda | 4.65× | [4.51, 4.80] |

## 3. Efecto de cada técnica sobre la latencia (vs V0), por dispositivo

> Razón = latencia(V0) / latencia(técnica). >1 acelera; <1 frena.

| Modelo | Disp. | Técnica | Aceleración vs V0 | IC95 |
|---|---|---|--:|--:|
| ResNet-50 | GPU | INT8 | 2.58× | [2.57, 2.59] |
| ResNet-50 | GPU | Poda | 1.29× | [1.28, 1.29] |
| ResNet-50 | CPU | INT8 | 2.50× | [2.49, 2.52] |
| ResNet-50 | CPU | Poda | 1.88× | [1.87, 1.89] |
| ResNet-50 | RPi-CPU | INT8 | 1.93× | [1.77, 2.11] |
| ResNet-50 | RPi-CPU | Poda | 2.40× | [2.34, 2.46] |
| MobileNetV2 | GPU | INT8 | 1.37× | [1.27, 1.48] |
| MobileNetV2 | GPU | Poda | 1.03× | [0.96, 1.12] |
| MobileNetV2 | CPU | INT8 | 1.01× | [1.00, 1.02] |
| MobileNetV2 | CPU | Poda | 1.43× | [1.42, 1.44] |
| MobileNetV2 | RPi-CPU | INT8 | 0.85× | [0.78, 0.93] |
| MobileNetV2 | RPi-CPU | Poda | 1.99× | [1.89, 2.09] |

## 4. Precisión (ImageNet-V2, top-1) por técnica

| Modelo | V0 | INT8 | Poda (FT) | Poda+KD |
|---|--:|--:|--:|--:|
| ResNet-50 | 0.694 | 0.683 | 0.510 | 0.579 |
| MobileNetV2 | 0.596 | 0.591 | 0.453 | 0.508 |

## 5. ART factorial (chequeo de robustez; la inferencia principal es efecto + IC)

> Aligned Rank Transform sobre el log de la media geométrica por corrida; factores
> dispositivo × técnica (V0/INT8/Poda). La interacción dispositivo×técnica ES el
> hallazgo de OE3: la técnica mueve la brecha. p reportado como confirmación no
> paramétrica, no como criterio de decisión (D11). Verificable en R/ARTool con el CSV.

| Modelo | Diseño | Efecto | F | gl | p | η²p |
|---|---|---|--:|--:|--:|--:|
| ResNet-50 | Jetson: GPU vs CPU | dispositivo | 76.7 | 1, 24 | <0.001 | 0.76 |
| ResNet-50 | Jetson: GPU vs CPU | técnica | 100.7 | 2, 24 | <0.001 | 0.89 |
| ResNet-50 | Jetson: GPU vs CPU | disp×téc | 433.3 | 2, 24 | <0.001 | 0.97 |
| ResNet-50 | 3 dispositivos | dispositivo | 149.8 | 2, 41 | <0.001 | 0.88 |
| ResNet-50 | 3 dispositivos | técnica | 152.1 | 2, 41 | <0.001 | 0.88 |
| ResNet-50 | 3 dispositivos | disp×téc | 203.4 | 4, 41 | <0.001 | 0.95 |
| MobileNetV2 | Jetson: GPU vs CPU | dispositivo | 82.7 | 1, 24 | <0.001 | 0.77 |
| MobileNetV2 | Jetson: GPU vs CPU | técnica | 107.5 | 2, 24 | <0.001 | 0.90 |
| MobileNetV2 | Jetson: GPU vs CPU | disp×téc | 117.6 | 2, 24 | <0.001 | 0.91 |
| MobileNetV2 | 3 dispositivos | dispositivo | 112.1 | 2, 62 | <0.001 | 0.78 |
| MobileNetV2 | 3 dispositivos | técnica | 47.7 | 2, 62 | <0.001 | 0.61 |
| MobileNetV2 | 3 dispositivos | disp×téc | 53.6 | 4, 62 | <0.001 | 0.78 |

---
CSV ordenado por corrida para ART/R: `results/oe3_tidy_runs.csv` (123 filas).
Figura de la brecha GPU↔CPU: `results/oe3_brecha_gpu_cpu.png` (+ .pdf).
Figura de la brecha de despliegue: `results/oe3_brecha_despliegue.png` (+ .pdf).
