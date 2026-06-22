# Orquestar desde cualquier máquina anfitriona (host) — y cómo hacerlo en Windows

El banco de medición distingue dos roles:

- **Equipo medido** (Jetson Orin Nano, Raspberry Pi 5): corre la inferencia. Es Linux y los pasos de
  medición se ejecutan ahí tal cual.
- **Máquina anfitriona (host)**: desde donde se *orquesta* la campaña con `scripts/measure_remote.py`
  —se conecta por SSH al equipo medido, arranca el registro de energía local (medidor INA226 + puente
  CP2112, leído por **hidapi**, que es multiplataforma) y consolida los resultados—. El host puede ser
  **macOS, Linux o Windows**: nada del flujo es exclusivo del Mac.

Importante: **latencia y precisión no requieren host**. Se corren directamente en el equipo medido
(ver `QUICKSTART_*` y `GUIA_LUIS_RPI.md`). El host solo hace falta para *automatizar* la campaña y para
la **energía**, porque el medidor se conecta al host.

## Requisitos del host (cualquier sistema operativo)

- Python 3, el repositorio clonado y `pip install -r requirements.txt`.
- SSH **por llave** hacia el equipo medido (sin contraseña) y reloj sincronizado por NTP en ambos.
- git con **Git LFS** instalado (para traer los `.onnx`).
- Solo para energía: el medidor INA226 + CP2112 conectado al host (se lee por hidapi; ver
  `POWER_MEASUREMENT.md`).

## En Windows

### Opción A (recomendada): WSL2 con Ubuntu

Da un entorno Linux idéntico al de la documentación, así que **todos los comandos funcionan igual** y
no hay que adaptar nada.

1. Instalar WSL2 + Ubuntu. En **PowerShell como administrador**:
   ```powershell
   wsl --install
   ```
   Reinicia cuando lo pida, abre Ubuntu y crea tu usuario de Linux.

2. Dentro de Ubuntu, preparar el repositorio y el entorno:
   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-pip git git-lfs
   git lfs install
   git clone https://github.com/ODuqueDvise/edge-ai-benchmark.git ~/edge-ai-benchmark   # en ~/ (ext4), NO en /mnt/c
   cd ~/edge-ai-benchmark && python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. SSH por llave hacia el equipo medido (RPi o Jetson):
   ```bash
   ssh-keygen -t ed25519 -C "host"
   ssh-copy-id usuario@equipo        # una vez; sin esto el orquestador se cuelga pidiendo contraseña
   ```

4. **Latencia y precisión**: el orquestador corre tal cual desde WSL; no requieren hardware extra.

5. **Energía (el punto delicado)**: el medidor es un USB HID (CP2112) y WSL2 no expone el USB por
   defecto. Hay que adjuntarlo con **usbipd-win**:
   - En **Windows** (PowerShell admin): `winget install usbipd` (o desde `github.com/dorssel/usbipd-win`).
   - `usbipd list` → ubica el CP2112 y anota su `BUSID`.
   - `usbipd bind --busid <BUSID>` (una vez) y `usbipd attach --wsl --busid <BUSID>` (cada vez que lo conectes).
   - En WSL, `lsusb` debe listarlo y hidapi lo leerá.
   - [Probable] Pasar un dispositivo **HID** a WSL puede ser quisquilloso según la versión del kernel;
     si se resiste, usa la Opción B solo para el registro de energía.

### Opción B: Windows nativo (sin WSL)

Útil sobre todo si el paso del medidor USB a WSL se complica: en Windows nativo el CP2112 (HID)
funciona **sin driver** (documentado en `POWER_MEASUREMENT.md`).

- Instala **Python para Windows** (python.org), **Git para Windows** (incluye *Git Bash*, necesario
  para el wrapper `measure_jetson_model.sh`) y usa el **cliente OpenSSH** de Windows.
- `measure_remote.py` es Python puro y corre en Windows; ejecútalo desde *Git Bash* o PowerShell.
- El registro de energía (`ina226_cp2112_logger.py`) funciona nativo por hidapi.
- Contrapartida: más piezas sueltas y rutas/entornos distintos a la documentación Linux.

## Recomendación

Para latencia y precisión —y para empezar sin fricción— usa **WSL2**: el entorno es idéntico al de la
documentación. Para la energía, adjunta el medidor con **usbipd-win**; y si el HID en WSL se resiste,
registra la energía con **Windows nativo** (hidapi sin driver) y deja el resto en WSL.
