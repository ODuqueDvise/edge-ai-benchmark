#!/usr/bin/env bash
# Trae los resultados (*.json) de un equipo de medicion (Jetson/RPi) al Mac,
# para calcular energia y consolidar el log. Usa rsync por SSH (solo copia lo nuevo).
#
# Uso:
#   bash scripts/fetch_results.sh <usuario@host> [ruta_repo_remoto]
#   bash scripts/fetch_results.sh orlando@orlando-desktop.local
#   bash scripts/fetch_results.sh luis@raspberrypi.local edge-ai-benchmark
#
# La ruta remota por defecto es 'edge-ai-benchmark' (relativa al home del equipo).
set -u
HOST="${1:?Uso: bash scripts/fetch_results.sh usuario@host [ruta_repo_remoto]}"
REPO="${2:-edge-ai-benchmark}"

cd "$(dirname "$0")/.." || { echo "No encuentro la raiz del repo."; exit 1; }
mkdir -p results

echo "Trayendo resultados de ${HOST}:${REPO}/results/ ..."
if rsync -av --include='*.json' --exclude='*' "${HOST}:${REPO}/results/" ./results/; then
  echo "Listo. Regenera el log:  python3 scripts/build_results_log.py"
else
  echo "Fallo el rsync. Revisa: SSH al host, la ruta del repo remoto, y que rsync este instalado en ambos."
  exit 1
fi
