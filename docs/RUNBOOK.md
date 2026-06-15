# Runbook — proceso end-to-end

Orden de ejecución de todo el flujo, con puntero a cada guía. Pensado para que
Orlando (Jetson) y Luis (RPi) trabajen igual y de forma trazable.

## Orden general

1. **Configurar el equipo**
   - Jetson Orin Nano → `docs/QUICKSTART_JETSON.md`
   - Raspberry Pi 5 → `docs/QUICKSTART_RPI.md`
2. **Modelo canónico** — ya viene en el repo (`models/cnn_baseline.onnx`).
   Verifica el checksum: `sha256sum models/cnn_baseline.onnx`
   → `609015cbb6ed30c7c456a2911a79bd2d303953e269a2d901da138dfcd56eb0dd`
3. **Línea base de LATENCIA** (3 condiciones) — comandos en `README.md`.
4. **Línea base de PRECISIÓN** — `docs/QUICKSTART_ACCURACY.md` (descarga ImageNet-V2,
   verifica preprocesamiento, corre el set completo en las 3 condiciones).
5. **ENERGÍA** — `docs/POWER_MEASUREMENT.md` (cuando llegue el INA226 + CP2112;
   validar con `--selftest` antes de confiar).
6. **Fase 2 / OE1** — aplicar técnicas de optimización y llenar la matriz (pendiente).

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

```bash
cd edge-ai-benchmark
git pull
# si hay conflicto por cambios locales y quieres alinear EXACTO con el remoto:
#   git fetch origin && git reset --hard origin/master
```

### Subir cambios (flujo estándar, trabajando dentro del clon)

```bash
git pull                              # 1. sincroniza antes
# ... editar / generar resultados ...
git add -A                            # 2. agrega cambios (respeta .gitignore)
git add -f models/cnn_baseline.onnx   # 3. solo si cambió el modelo (esta ignorado)
git commit -m "mensaje claro"         # 4. confirma
git push                              # 5. sube
```

> Nota (Orlando): si editas fuera del clon (en la carpeta del proyecto de Cowork),
> copia al clon antes de confirmar:
> `rsync -a --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude='datasets' "<carpeta_proyecto>/" ./`

### Qué se versiona y qué no

- **Sí:** código, guías, `results/*.json`, `RESULTS_LOG.md`, el modelo canónico (forzado).
- **No:** `datasets/` (cada equipo descarga el suyo), `.venv/`, `__pycache__/`,
  otros modelos en `models/`.

## Estado actual (junio 2026)

- Jetson: arranca desde NVMe; arnés validado en `jetson-gpu` y `jetson-cpu`;
  precisión validada (top-1 ~60.7% en subconjunto representativo de 2000).
- Pendiente: set completo de precisión en las 3 condiciones; latencia/precisión
  en `rpi-cpu` (Luis); energía (a la espera del INA226 + CP2112); Fase 2 / OE1.
