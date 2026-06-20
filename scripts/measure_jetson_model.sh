#!/usr/bin/env bash
# Corre la línea base / variante de un modelo en las DOS condiciones de la Jetson
# (jetson-gpu/tensorrt y jetson-cpu/cpu) con un solo comando, sobre measure_remote.py.
# Cada condición: chequeos -> latencia R -> energía -> commit (ver measure_remote.py).
#
# Uso:
#   bash scripts/measure_jetson_model.sh models/resnet50_baseline.onnx
#   bash scripts/measure_jetson_model.sh models/resnet50_baseline.onnx <sha256>
#   bash scripts/measure_jetson_model.sh models/resnet50_baseline.onnx <sha256> --accuracy
#   bash scripts/measure_jetson_model.sh models/resnet50_baseline.onnx --dry-run
# Variables de entorno opcionales: HOST (def orlando@orlando-desktop.local), SHUNT (def 0.1), IDLE (def 7.8)
set -u
MODEL="${1:?Uso: bash scripts/measure_jetson_model.sh models/<modelo>.onnx [sha256] [flags extra]}"
shift
SHA=""
if [ "${1:-}" ] && [[ "${1:-}" != --* ]]; then SHA="$1"; shift; fi   # 2º arg sin guion = sha esperado

HOST="${HOST:-orlando@orlando-desktop.local}"
SHUNT="${SHUNT:-0.1}"   # R100 = 0.1 Ω (Jetson). La RPi usaría R010 = 0.01.
IDLE="${IDLE:-7.8}"
DIR="$(cd "$(dirname "$0")" && pwd)"
SHA_ARG=(); [ -n "$SHA" ] && SHA_ARG=(--expect-sha "$SHA")

TAGS=(jetson-gpu jetson-cpu)
PROVS=(tensorrt cpu)
for i in 0 1; do
  echo ">>> ${TAGS[$i]} / ${PROVS[$i]}"
  python3 "$DIR/measure_remote.py" --host "$HOST" \
    --device-tag "${TAGS[$i]}" --provider "${PROVS[$i]}" \
    --model "$MODEL" --shunt "$SHUNT" --idle-watts "$IDLE" "${SHA_ARG[@]}" "$@" || exit 1
done
echo "Listo: ambas condiciones de la Jetson para $MODEL"
