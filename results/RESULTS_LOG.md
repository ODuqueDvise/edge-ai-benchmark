# Registro de resultados (GENERADO)

> Generado por `scripts/build_results_log.py` desde `results/*.json`. **No editar a mano.**
> Re-generar tras nuevas corridas (idealmente despues de `git pull`, para incluir ambos equipos).
> Constantes congeladas: warmup 100, iters 2000, R 5, MAXN (Jetson) / governor performance (RPi), entrada 1,3,224,224.

## Latencia

| Condicion | Modelo | R | p50 media±desv (ms) | p95 media (ms) | p99 media (ms) | thr media (ips) |
|---|---|---|---|---|---|---|
| jetson-cpu | 2bf58c6f | 1 | 0.601 ± 0.000 | 0.724 | 0.948 | 1604.5 |
| jetson-cpu | V0 base (MobileNetV2) | 1 | 12.263 ± 0.000 | 14.250 | 15.523 | 79.6 |
| jetson-cpu | V0 base (MobileNetV2) | 5 | 12.280 ± 0.035 | 14.270 | 16.285 | 79.3 |
| jetson-gpu | 2bf58c6f | 1 | 0.470 ± 0.000 | 0.499 | 0.520 | 2113.4 |
| jetson-gpu | V0 base (MobileNetV2) | 1 | 2.487 ± 0.000 | 2.532 | 2.578 | 401.1 |
| jetson-gpu | V0 base (MobileNetV2) | 10 | 2.642 ± 0.184 | 3.113 | 3.611 | 372.8 |

## Precision (ImageNet-V2)

| Condicion | Modelo | n img | top-1 | top-5 |
|---|---|---|---|---|
| jetson-cpu | V0 base (MobileNetV2) | 2000 | 0.6070 | 0.8175 |
| jetson-cpu | V0 base (MobileNetV2) | 2000 | 0.6740 | 0.8640 |
| jetson-cpu | V0 base (MobileNetV2) | 10000 | 0.5961 | 0.8188 |
| jetson-gpu | V0 base (MobileNetV2) | 10000 | 0.5959 | 0.8189 |

## Energia (potencia media, referencia interna; medidor externo pendiente)

| Condicion | Modelo | pot. media (W, ref. interna) |
|---|---|---|
| jetson-cpu | 2bf58c6f | 7.72 |
| jetson-cpu | V0 base (MobileNetV2) | 9.03 |
| jetson-gpu | 2bf58c6f | 7.01 |
| jetson-gpu | V0 base (MobileNetV2) | 8.98 |
