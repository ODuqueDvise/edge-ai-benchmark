# Runbook — proceso end-to-end

Orden de ejecución de todo el flujo, con puntero a cada guía. Pensado para que
Luis (RPi) y yo (Jetson) trabajemos igual y de forma trazable.

## Orden general

1. **Configurar el equipo**
   - Jetson Orin Nano → `docs/QUICKSTART_JETSON.md`
   - Raspberry Pi 5 → `docs/GUIA_LUIS_RPI.md` (guía completa paso a paso) o `docs/QUICKSTART_RPI.md` (referencia rápida)
2. **Modelos canónicos** — dos modelos comprometidos (ver `docs/DECISIONS.md` D9):
   - MobileNetV2 → `models/cnn_baseline.onnx` (sha `609015cb…56eb0dd`), ya en el repo.
   - ResNet-50 → `models/resnet50_baseline.onnx`; se exporta una vez y queda versionado
     en el repo vía Git LFS (igual que el resto de `*.onnx`), así que viene con el clon:
     `python scripts/export_model.py --model-name resnet50 --output models/resnet50_baseline.onnx --opset 18`
   Verifica antes de medir: `sha256sum models/<archivo>.onnx`. El nombre del JSON de cada
   corrida incluye el modelo, así que ambos coexisten sin pisarse en `results/`.
3. **Línea base de LATENCIA** (3 condiciones) — comandos en `README.md`.
4. **Línea base de PRECISIÓN** — `docs/QUICKSTART_ACCURACY.md` (descarga ImageNet-V2,
   verifica preprocesamiento, corre el set completo en las 3 condiciones).
5. **ENERGÍA** — `docs/POWER_MEASUREMENT.md` (cuando llegue el INA226 + CP2112;
   validar con `--selftest` antes de confiar). La dirección I2C va SIEMPRE fija
   con `--addr` (Jetson 0x40, RPi/Luis 0x44); `--scan` es solo diagnóstico, no
   se usa en el camino de medición.
6. **Fase 2 / OE1** — aplicar las técnicas en orden **INT8 → poda estructurada → destilación
   (al final, por su costo)**, sobre los dos modelos, y llenar la matriz. En la Orin la poda
   solo baja latencia si es estructurada (ver `docs/DECISIONS.md` D10). El INT8 se hace con
   `scripts/quantize_int8.py` (diseño en `docs/DISENO_INT8_OE1.md`, D13).

## Reglas de coherencia (transversales)

- Mismo modelo (verificar checksum), mismo dataset (variante + nº de imágenes),
  mismos parámetros (`--input-shape 1,3,224,224`, `--warmup`, `--iters`, `--limit`).
- Versiones del stack congeladas (registrar con `scripts/collect_env.sh`).
- Resultados crudos (`results/*.json`) y el `RESULTS_LOG.md` se versionan en el repo.
- El medidor de energía debe ser el mismo método/instrumento en ambos equipos.

## Constantes congeladas del protocolo (junio 2026)

Decididas con corrida piloto en la Jetson (CV del p50 entre corridas = 0.56%, equipo
estable). **No se cambian sin consenso**; si se cambian, las mediciones previas dejan
de ser comparables.

- Calentamiento (K): **100** inferencias descartadas.
- Inferencias por serie (M): **2000**.
- Ejecuciones independientes (R): **5** (idealmente con reinicio entre ellas).
- Modo de potencia: Jetson **MAXN** (`nvpmodel -m 0` + `jetson_clocks`); RPi gobernador **performance**.
- Forma de entrada: **1,3,224,224**.
- Precision: set **completo (10000)**, sin `--limit`, para cifras oficiales.

## Flujo de trabajo con git (paso a paso)

### Traer los últimos cambios (antes de trabajar)

Una sola vez por máquina, antes de clonar o actualizar, instala Git LFS para que los
`*.onnx` lleguen como archivo real y no como puntero de texto (Linux: `sudo apt install -y
git-lfs`; macOS: `brew install git-lfs`; luego `git lfs install`).

```bash
cd edge-ai-benchmark
git pull
# si hay conflicto por cambios locales y quieres alinear EXACTO con el remoto:
#   git fetch origin && git reset --hard origin/master
```

### Subir cambios (orden robusto: commit ANTES de pull)

Con `pull.rebase` activo, `git pull` falla si hay cambios sin commitear. Por eso se
confirma primero y el pull va después (el rebase reproduce tu commit encima de lo remoto):

```bash
git add -A                            # 1. agrega tus cambios (los baselines ya NO están ignorados)
git commit -m "mensaje claro"         # 2. confirma
git pull                              # 3. integra lo remoto (rebasa tu commit encima)
git push                              # 4. sube
```

Hazlo una vez: `git config --global pull.rebase true`. El `git pull` de primero solo
sirve con el árbol limpio; si tienes ediciones pendientes, commitea antes.

> Nota: en mi equipo, la carpeta del proyecto YA es el clon (normalizado), así que se
> edita y commitea ahí directamente; no se usa rsync.

### Subir resultados de mediciones (atajo)

Tras medir en la Jetson o la RPi, en vez del ciclo manual usa el script portable:

```bash
bash scripts/sync_results.sh
# o con un mensaje propio:
bash scripts/sync_results.sh "Baseline V0 jetson-gpu R=5"
```

Hace `pull --rebase` + `add results/` + `commit` + `push`, con aviso si hay conflicto
(p.ej. RESULTS_LOG.md editado en dos sitios) o si no hay resultados nuevos.

### Qué se versiona y qué no

- **Sí:** código, guías, `results/*.json`, y **todos los `*.onnx`** vía Git LFS
  (un `.gitattributes` enruta `*.onnx`): baselines MobileNetV2 y ResNet-50, INT8,
  podados y destilados. ResNet-50 (~100MB) viene con el clon, ya no se pasa por archivo.
- **No:** `datasets/` (cada equipo descarga el suyo), `.venv/`, `__pycache__/`.

## Medición automatizada (un comando, desde la máquina anfitriona (host))

Para no orquestar a mano latencia + energía + sincronización en dos terminales,
`scripts/measure_remote.py` hace una condición de punta a punta por SSH y
`scripts/measure_jetson_model.sh` corre las dos condiciones de la Jetson de un tiro.
Comandos y opciones en el README ("Medición automatizada"). El host puede ser macOS,
Linux o Windows; para Windows, ver `docs/SETUP_HOST_WINDOWS.md`.

**Prerrequisitos (una vez):**
- SSH por llave host→equipo: `ssh-copy-id orlando@orlando-desktop.local`. Sin esto, el flujo se cuelga pidiendo contraseña.
- Relojes sincronizados por NTP en el host y el equipo medido (la ventana de energía depende de ello; Jetson: `sudo timedatectl set-ntp true`).

El orquestador **aborta en rojo** ante reloj desfasado, checksum que no cuadra, autotest del
medidor fallido, caída silenciosa de la GPU a CPU o ventana de energía sin cubrir. Los pasos
manuales de este runbook quedan como respaldo.

## Estado actual (junio 2026)

- Jetson: línea base V0 completa de MobileNetV2 en `jetson-gpu` y `jetson-cpu` (latencia,
  precisión y energía con medidor externo). GPU ~5x más rápida y ~4.3x menos energía a igual precisión.
- Director (CP2): confirma dos modelos (MobileNetV2 + ResNet-50) y el orden de técnicas
  (INT8 → poda estructurada → destilación). Ver `docs/DECISIONS.md` D9–D12.
- Pendiente: exportar ResNet-50 y su línea base; `rpi-cpu` (Luis); energía en rpi (shunt R010);
  Fase 2 / OE1 (optimización).
