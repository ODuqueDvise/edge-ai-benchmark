#!/usr/bin/env bash
# Sincroniza resultados con el repo. Portable (Jetson/RPi). Una sola orden:
#   pull --rebase  ->  regenerar RESULTS_LOG.md  ->  add results/  ->  commit  ->  push
# Uso: bash scripts/sync_results.sh ["mensaje opcional"]
set -u
cd "$(dirname "$0")/.." || { echo "No encuentro la raiz del repo."; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "No es un repo git."; exit 1; }

# 1. Traer remoto primero (incluye resultados del otro equipo)
git pull --rebase --autostash || { echo "Conflicto en pull; resuelve a mano y reintenta."; exit 1; }

# 2. Regenerar el log desde TODOS los JSON presentes (local + remoto)
python3 scripts/build_results_log.py || echo "(aviso) no se pudo regenerar el log; sigo igual"

# 3. Agregar solo resultados (JSON + log generado)
git add results/
if git diff --cached --quiet; then
  echo "No hay resultados nuevos. Repo al dia."
  exit 0
fi

# 4. Commit (mensaje automatico) + push
HOST="$(hostname)"
N="$(git diff --cached --name-only -- results/ | grep -c '\.json$' || true)"
STAMP="$(date -u +%Y-%m-%dT%H:%MZ)"
git commit -m "${1:-Resultados ${HOST}: ${N} JSON (${STAMP})}"
git push || { echo "Fallo el push (auth, o el remoto avanzo: reintenta el script)."; exit 1; }
echo "Listo: resultados y RESULTS_LOG sincronizados."
