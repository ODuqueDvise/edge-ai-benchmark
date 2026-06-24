# Quickstart — Raspberry Pi 5 (rpi-cpu) · tarjeta de comandos

Referencia rápida para la condición `rpi-cpu`. **El camino completo** —clonar, Git LFS, entorno,
gobernador de CPU, dataset, git/SSH y la energía— está en **`docs/GUIA_LUIS_RPI.md`**. Esta página
asume que ya hiciste ese setup y solo lista los comandos de medición.

**Constantes congeladas (no cambiar):** `--input-shape 1,3,224,224 --warmup 100 --iters 2000`, R = 5.
Usa los modelos del repositorio (vienen por Git LFS), **NO los reexportes**. Modelos por técnica:

- V0: `cnn_baseline`, `resnet50_baseline`
- INT8: `cnn_baseline_int8`, `resnet50_baseline_int8`
- Poda: `cnn_pruned`, `resnet50_pruned`

## Latencia (R = 5 por modelo)

```bash
source .venv/bin/activate
for m in cnn_baseline resnet50_baseline cnn_baseline_int8 resnet50_baseline_int8 cnn_pruned resnet50_pruned; do
  for i in 1 2 3 4 5; do
    python -m bench.run_benchmark --model models/$m.onnx --backend ort \
      --provider cpu --device-tag rpi-cpu --input-shape 1,3,224,224 --warmup 100 --iters 2000
  done
done
```

## Precisión (set completo, una por modelo)

```bash
for m in cnn_baseline resnet50_baseline cnn_baseline_int8 resnet50_baseline_int8 cnn_pruned resnet50_pruned; do
  python -m bench.run_accuracy --model models/$m.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu --dataset datasets/imagenetv2-matched-frequency-format-val
done
```

Esperado (top-1, MobileNetV2 / ResNet-50): V0 ≈ 0.59 / 0.69 · INT8 ≈ 0.59 / 0.67 · poda ≈ 0.45 / 0.51
(la poda baja la precisión **a propósito**). Si sale ~0.001, es problema de dataset: detente.

## Subir

```bash
bash scripts/sync_results.sh "rpi-cpu <lo que mediste>"
```

## Notas

- **Energía: diferida.** Necesita el medidor externo (INA226 + CP2112) con shunt **R010 (0.01 Ω)**; no la
  corras aún. Procedimiento en `docs/GUIA_LUIS_RPI.md` §11 y `docs/POWER_MEASUREMENT.md`.
- No actualices versiones a mitad de campaña; registra el entorno con `bash scripts/collect_env.sh`.
