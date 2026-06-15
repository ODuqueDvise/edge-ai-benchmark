#!/usr/bin/env bash
# Sincroniza los resultados de mediciones con el repo. Portable: Jetson y RPi.
#
# Uso:
#   bash scripts/sync_results.sh ["mensaje de commit opcional"]
#
# Hace: agrega results/ -> commit (mensaje automatico si no das uno) ->
#       pull --rebase (integra lo que el otro equipo haya subido) -> push.
# Maneja: sin resultados nuevos, conflictos (se detiene con aviso) y fallo de push.
set -u

# Ubicarse en la raiz del repo (sin importar desde donde se invoque)
cd "$(dirname "$0")/.." || { echo "No encuentro la raiz del repo."; exit 1; }
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Esto no es un repo git. Ubicate en el clon de edge-ai-benchmark."; exit 1
fi

# 1. Agregar solo los resultados (no toca codigo ni otros cambios)
git add results/ 2>/dev/null || true

if git diff --cached --quiet; then
  echo "No hay resultados nuevos para subir. Traigo el remoto por si acaso..."
  git pull --rebase --autostash || { echo "Conflicto al integrar; resuelve a mano y reintenta."; exit 1; }
  echo "Repo al dia. Nada que subir."
  exit 0
fi

# 2. Commit (mensaje automatico: host + nº de JSON + fecha UTC; o el que pases)
HOST="$(hostname)"
N="$(git diff --cached --name-only -- results/ | grep -c '\.json$' || true)"
STAMP="$(date -u +%Y-%m-%dT%H:%MZ)"
MSG="${1:-Resultados ${HOST}: ${N} JSON nuevos (${STAMP})}"
git commit -m "$MSG" || { echo "Nada que confirmar."; exit 0; }

# 3. Integrar remoto y publicar
if ! git pull --rebase --autostash; then
  echo "CONFLICTO al integrar (probablemente RESULTS_LOG.md editado en dos sitios)."
  echo "Resuelve el conflicto, luego: git rebase --continue && git push"
  exit 1
fi
if ! git push; then
  echo "Fallo el push. Revisa autenticacion (SSH/PAT) o reintenta el script."
  exit 1
fi
echo "Listo: resultados sincronizados con el repo."
