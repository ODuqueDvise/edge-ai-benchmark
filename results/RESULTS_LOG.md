# Registro de resultados (GENERADO)

> Generado por `scripts/build_results_log.py` desde `results/*.json`. **No editar a mano.**
> Re-generar tras nuevas corridas (idealmente despues de `git pull`, para incluir ambos equipos).
> Constantes congeladas: warmup 100, iters 2000, R 5, MAXN (Jetson) / governor performance (RPi), entrada 1,3,224,224.

## Latencia

| Condicion | Modelo | R | p50 media±desv (ms) | p95 media (ms) | p99 media (ms) | thr media (ips) |
|---|---|---|---|---|---|---|
| jetson-cpu | V0 base (MobileNetV2) | 5 | 12.280 ± 0.035 | 14.270 | 16.285 | 79.3 |
| jetson-gpu | V0 base (MobileNetV2) | 5 | 2.468 ± 0.009 | 2.497 | 2.517 | 404.7 |

## Precision (ImageNet-V2)

| Condicion | Modelo | n img | top-1 | top-5 |
|---|---|---|---|---|
| jetson-cpu | V0 base (MobileNetV2) | 10000 | 0.5961 | 0.8188 |
| jetson-gpu | V0 base (MobileNetV2) | 10000 | 0.5959 | 0.8189 |

## Energia (potencia media, referencia interna; medidor externo pendiente)

| Condicion | Modelo | pot. media (W, ref. interna) |
|---|---|---|
| jetson-cpu | V0 base (MobileNetV2) | 9.04 |
| jetson-gpu | V0 base (MobileNetV2) | 9.57 |
