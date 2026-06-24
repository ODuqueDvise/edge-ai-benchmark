# Guía paso a paso — Luis (Raspberry Pi 5)

Guía completa y autocontenida, de cero a resultados subidos. Pensada para seguir
**en orden, copiando y pegando**. Cada paso trae un "qué deberías ver" para que sepas
si salió bien antes de continuar.

## Dónde estamos y qué te toca

El proyecto compara **GPU vs CPU** en el borde con dos modelos: **MobileNetV2** y
**ResNet-50**. Orlando ya midió ambos en la Jetson (latencia, precisión y energía).
**Tu parte es la condición `rpi-cpu`**: correr los dos modelos en la Raspberry Pi 5
(que no tiene GPU para ML) con el **mismo arnés y los mismos parámetros**, para que
las cifras sean comparables.

- **Ahora:** latencia + precisión de los dos modelos en `rpi-cpu`.
- **Después (cuando llegue el hardware):** energía. Necesita un shunt **R010 (0.01 Ω)**
  porque el R100 de la Jetson se satura con la corriente de la RPi. No te bloquees con esto.

Tiempo aproximado: ~30–40 min de configuración + las mediciones (la precisión es lenta).

### Antes de empezar, ten a la mano
- Raspberry Pi 5 con Raspberry Pi OS de 64 bits, encendida, con internet y un disipador/ventilador (la RPi 5 baja la frecuencia si se calienta).
- Una cuenta de GitHub. **Pídele a Orlando que te agregue como colaborador** del repositorio `ODuqueDvise/edge-ai-benchmark` (sin eso no podrás hacer `push`).
- **Git LFS instalado** (los modelos `.onnx` se versionan con LFS): `sudo apt install -y git-lfs && git lfs install`.

---

## 1. Clonar el repositorio

```bash
cd ~
sudo apt install -y git-lfs && git lfs install   # una sola vez por máquina: trae los .onnx reales, no punteros
git clone https://github.com/ODuqueDvise/edge-ai-benchmark.git
cd edge-ai-benchmark
ls
```
**Qué deberías ver:** carpetas `bench/`, `scripts/`, `docs/`, `models/`, etc.

## 2. Configurar git para poder subir (una sola vez)

Esto evita el error `GH007` (que aparece si tu correo de GitHub es privado) y deja
el `push` por SSH.

```bash
# 2.1 Tu identidad. Usa TU correo noreply de GitHub (NO el de Orlando):
#     GitHub -> Settings -> Emails -> "Keep my email private" -> copia el correo
#     con forma  12345678+tuusuario@users.noreply.github.com
git config --global user.name  "Luis David Lenes"
git config --global user.email "TU_ID+tuusuario@users.noreply.github.com"
git config --global pull.rebase true

# 2.2 Llave SSH y registrarla en GitHub
ssh-keygen -t ed25519 -C "luis-rpi"        # Enter en todo (sin passphrase está bien)
cat ~/.ssh/id_ed25519.pub
#   Copia esa línea completa y pégala en: GitHub -> Settings -> SSH and GPG keys -> New SSH key

# 2.3 Apunta el repo a SSH y prueba
git remote set-url origin git@github.com:ODuqueDvise/edge-ai-benchmark.git
ssh -T git@github.com
```
**Qué deberías ver:** al final, un saludo tipo `Hi tuusuario! You've successfully authenticated...`.
Si dice *permission denied*, la llave no quedó registrada (repite 2.2) o Orlando aún no te agregó como colaborador.

## 3. Preparar el entorno de Python

```bash
sudo apt update && sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install onnxruntime          # versión de CPU (en la RPi el wheel de PyPI sirve)
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
**Qué deberías ver:** una lista que incluye `CPUExecutionProvider`.
**Importante:** de aquí en adelante, trabaja siempre con el entorno activado (verás `(.venv)` al inicio de la línea). Si abres otra terminal, vuelve a `cd ~/edge-ai-benchmark && source .venv/bin/activate`.

## 4. Tener los dos modelos y verificarlos

Ambos modelos ya vinieron con el clon (se versionan en el repo vía Git LFS), así que
no tienes que recibir ResNet-50 por aparte. Verifica que son idénticos a los de Orlando
—si un solo byte difiere, las mediciones no son comparables—:

```bash
sha256sum models/cnn_baseline.onnx models/resnet50_baseline.onnx
```
**Qué deberías ver, exactamente:**
```
609015cbb6ed30c7c456a2911a79bd2d303953e269a2d901da138dfcd56eb0dd  models/cnn_baseline.onnx
05e5bc14444e89b9b47b36c663bc40e061db8d20389d833dcde3c7da667290dc  models/resnet50_baseline.onnx
```
Si alguno no coincide, **no midas**: confirma que Git LFS quedó instalado (`git lfs install`)
y vuelve a traer los archivos con `git lfs pull`.

## 5. Descargar el dataset (ImageNet-V2, una vez)

```bash
pip install -U huggingface_hub
python - << 'PY'
from huggingface_hub import list_repo_files, hf_hub_download
files = list_repo_files("vaishaal/ImageNetV2", repo_type="dataset")
cand = [f for f in files if "matched-frequency" in f and f.endswith((".tar.gz", ".tgz", ".tar"))]
print("Descargando:", cand[0])
p = hf_hub_download("vaishaal/ImageNetV2", cand[0], repo_type="dataset", local_dir="datasets")
print("OK:", p)
PY
tar -xf datasets/imagenetv2-matched-frequency*.tar* -C datasets
find datasets/imagenetv2-matched-frequency-format-val -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
```
**Qué deberías ver:** el último número debe ser **10000**. Si es 0, revisa que la carpeta sea `datasets/imagenetv2-matched-frequency-format-val/`.

## 6. Fijar el desempeño de la CPU (reproducibilidad)

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
bash scripts/collect_env.sh      # guarda versiones y estado del equipo (queda como evidencia)
```
**Qué deberías ver:** la palabra `performance` repetida (una por núcleo). Asegúrate de tener el ventilador/disipador puesto.

## 7. Medir LATENCIA (los dos modelos, R = 5)

R = 5 significa **cinco corridas por modelo**. El bucle las hace solo:

```bash
# MobileNetV2 (5 corridas)
for i in 1 2 3 4 5; do
  python -m bench.run_benchmark --model models/cnn_baseline.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu --input-shape 1,3,224,224 --warmup 100 --iters 2000
done
# ResNet-50 (5 corridas)
for i in 1 2 3 4 5; do
  python -m bench.run_benchmark --model models/resnet50_baseline.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu --input-shape 1,3,224,224 --warmup 100 --iters 2000
done
```
**Qué deberías ver:** cada corrida imprime `Escrito: results/rpi-cpu_<modelo>_..._.json` y un `p50=… ms`. ResNet-50 será bastante más lento que MobileNetV2 (es normal).

## 8. Medir PRECISIÓN (los dos modelos)

Primero una **prueba corta** para confirmar que todo está bien (con `--limit 2000`),
y solo si el número es sano, la corrida completa. **Aviso:** la precisión en CPU es
lenta (ResNet-50 sobre 10000 imágenes puede tardar bastante; puedes dejarla corriendo).

```bash
# Prueba corta (MobileNetV2)
python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
  --provider cpu --device-tag rpi-cpu --dataset datasets/imagenetv2-matched-frequency-format-val --limit 2000
```
**Qué deberías ver:** un `top-1` alrededor de **0.59** para MobileNetV2 (y ~**0.69** para ResNet-50).
Si sale cerca de **0.001** (lo aleatorio), algo está mal en el dataset; **detente** y avísale a Orlando — no es degradación real.

Si el número es sano, corre las dos completas (sin `--limit`):

```bash
python -m bench.run_accuracy --model models/cnn_baseline.onnx --backend ort \
  --provider cpu --device-tag rpi-cpu --dataset datasets/imagenetv2-matched-frequency-format-val
python -m bench.run_accuracy --model models/resnet50_baseline.onnx --backend ort \
  --provider cpu --device-tag rpi-cpu --dataset datasets/imagenetv2-matched-frequency-format-val
```

## 9. Variantes optimizadas: INT8 y poda (latencia + precisión, SIN energía)

Cuando tengas lista la línea base V0, repite latencia y precisión para las dos técnicas ya
optimizadas. **No necesitas el medidor de energía**: el dongle (INA226 + CP2112) es solo para
energía, que sigue diferida. Esto es latencia + precisión, exactamente igual que V0.

Primero, baja los modelos optimizados (vienen en el repo por **Git LFS**):

```bash
git lfs install        # una sola vez, si no lo has hecho
git pull               # trae los .onnx INT8 y podados
ls -lh models/*_int8.onnx models/*_pruned.onnx   # deben pesar MB reales; si dan ~130 bytes son punteros LFS sin descargar -> repite git lfs install + git pull
```

**INT8** — archivos `*_int8.onnx`:

```bash
# Latencia (5 corridas cada uno)
for m in cnn_baseline_int8 resnet50_baseline_int8; do
  for i in 1 2 3 4 5; do
    python -m bench.run_benchmark --model models/$m.onnx --backend ort \
      --provider cpu --device-tag rpi-cpu --input-shape 1,3,224,224 --warmup 100 --iters 2000
  done
done
# Precisión (completa)
for m in cnn_baseline_int8 resnet50_baseline_int8; do
  python -m bench.run_accuracy --model models/$m.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu --dataset datasets/imagenetv2-matched-frequency-format-val
done
```
**Qué esperar (precisión):** ~**0.59** (MobileNetV2 INT8) y ~**0.67** (ResNet-50 INT8) — casi igual que V0.

**Poda** — archivos `*_pruned.onnx`:

```bash
# Latencia (5 corridas cada uno)
for m in cnn_pruned resnet50_pruned; do
  for i in 1 2 3 4 5; do
    python -m bench.run_benchmark --model models/$m.onnx --backend ort \
      --provider cpu --device-tag rpi-cpu --input-shape 1,3,224,224 --warmup 100 --iters 2000
  done
done
# Precisión (completa)
for m in cnn_pruned resnet50_pruned; do
  python -m bench.run_accuracy --model models/$m.onnx --backend ort \
    --provider cpu --device-tag rpi-cpu --dataset datasets/imagenetv2-matched-frequency-format-val
done
```
**Qué esperar (precisión):** ~**0.45** (MobileNetV2 podado) y ~**0.51** (ResNet-50 podado). **Ojo:** la poda
baja la precisión **a propósito** —es el costo de la recuperación limitada, ya documentado—; ese número
más bajo es lo esperado, **no es un error**. (Si sale ~0.001, eso sí es problema de dataset: detente y avisa.)

Al terminar, sube todo (sección siguiente) con un mensaje claro, p. ej.
`bash scripts/sync_results.sh "rpi-cpu INT8 + poda (MobileNetV2 + ResNet-50)"`.

## 10. Subir los resultados

Un solo comando hace `pull` + regenerar el log + `commit` + `push`:

```bash
bash scripts/sync_results.sh "rpi-cpu baseline V0 (MobileNetV2 + ResNet-50)"
```
**Qué deberías ver:** al final, `Listo: resultados y RESULTS_LOG sincronizados.`
- Si dice **GH007 / email privado**: tu `user.email` del paso 2.1 no es el noreply; corrígelo y repite.
- Si dice **permission denied / authentication**: falta la llave SSH (paso 2.2) o que Orlando te agregue como colaborador.
- Si dice **conflicto en el pull**: avísale a Orlando antes de forzar nada.

## 11. Energía (DIFERIDA — no la necesitas todavía)

La RPi 5 no tiene sensor de potencia, así que la energía se mide con el **mismo
medidor externo** (INA226 + CP2112) que la Jetson, pero con un shunt **R010 (0.01 Ω)**
en vez del R100. Cuando tengas ese hardware, el procedimiento está en
`docs/POWER_MEASUREMENT.md`, y se puede automatizar con el orquestador desde tu portátil:
`python3 scripts/measure_remote.py --host <tu_usuario@tu_rpi> --device-tag rpi-cpu --provider cpu --model models/<modelo>.onnx --shunt 0.01`.
Por ahora, déjala pendiente. Tu portátil puede ser Windows o Mac; si es Windows, prepara primero el entorno anfitrión (lo más simple es WSL2): ver `docs/SETUP_HOST_WINDOWS.md`.

---

## Lista de verificación final

En `results/` deberías terminar con, al menos:
- 5 JSON `rpi-cpu_cnn_baseline_ort_cpu_*.json` (latencia MobileNetV2)
- 5 JSON `rpi-cpu_resnet50_baseline_ort_cpu_*.json` (latencia ResNet-50)
- 2 JSON `acc_rpi-cpu_*_ort_cpu_*.json` (precisión, uno por modelo)

Cuando estén subidos, avísale a Orlando para consolidar la comparación de dispositivo.

## Si algo se rompe (lo más común)

- **`command not found` o `ModuleNotFoundError`:** olvidaste activar el entorno → `source .venv/bin/activate`.
- **El checksum no coincide:** el `.onnx` quedó como puntero de LFS o desactualizado → `git lfs install` y `git lfs pull`, no midas.
- **El dataset da 0 imágenes:** la carpeta no es `imagenetv2-matched-frequency-format-val/` → revisa el paso 5.
- **No actualices versiones a mitad de campaña** (Python, onnxruntime, etc.). Si algo cambió, corre `scripts/collect_env.sh` otra vez y avisa.
