#!/usr/bin/env bash
# Vuelca el entorno de un equipo a un archivo, para congelar versiones por corrida.
set -u
OUT="${1:-env_$(hostname)_$(date -u +%Y%m%d-%H%M%S).txt}"
{
  echo "=== fecha (UTC) ==="; date -u
  echo "=== uname ==="; uname -a
  echo "=== os-release ==="; cat /etc/os-release 2>/dev/null
  echo "=== nv_tegra_release (Jetson) ==="; cat /etc/nv_tegra_release 2>/dev/null
  echo "=== nvpmodel ==="; nvpmodel -q 2>/dev/null
  echo "=== jetson_clocks --show ==="; jetson_clocks --show 2>/dev/null | head
  echo "=== CUDA ==="; nvcc --version 2>/dev/null
  echo "=== python ==="; python3 --version
  echo "=== pip (runtimes) ==="; python3 -m pip list 2>/dev/null | grep -Ei 'onnx|tflite|tensorflow|numpy|tensorrt'
  echo "=== gobernador CPU (RPi) ==="; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u
  echo "=== termico ==="; for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type): $(cat $z/temp)"; done 2>/dev/null
} | tee "$OUT"
echo "Entorno guardado en $OUT"
