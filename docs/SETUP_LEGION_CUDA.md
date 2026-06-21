# Setup del Legion para entrenamiento (WSL2 + CUDA) — fase de poda

Guía paso a paso para dejar el Lenovo Legion 5 (RTX 3060 Laptop, 6 GB) listo para hacer
*fine-tuning* de los modelos podados con PyTorch sobre GPU. Es la preparación de la
**segunda técnica del OE1** (poda estructurada), que —a diferencia del INT8— necesita
reentrenar para recuperar la precisión. Síguela en orden; cada paso trae una verificación.

Tiempo aproximado: ~30–45 min (más la descarga de PyTorch).

## Antes de empezar
- Legion con Windows 11, **enchufado a la corriente** (entrenar en batería hace *throttling*).
- ~40–150 GB libres en disco (tienes 787 GB, de sobra).
- Importante: el soporte de CUDA en WSL viene del **driver de NVIDIA de Windows**. NO se
  instala ningún driver de NVIDIA dentro de Linux/WSL.

---

## 1. Confirmar y actualizar el driver de NVIDIA (en Windows)

Abre **PowerShell** (o cmd) y ejecuta:

```powershell
nvidia-smi
```

**Qué deberías ver:** una tabla con `NVIDIA GeForce RTX 3060 Laptop GPU` y, arriba, una
versión de driver. Si el driver es viejo (más de unos meses), actualízalo desde
**NVIDIA App / GeForce Experience** o `nvidia.com/drivers` antes de seguir: el CUDA de WSL
depende de que el driver de Windows sea reciente. En esa misma tabla, la columna de memoria
muestra `6144MiB` → confirma tus 6 GB de VRAM.

## 2. Instalar WSL2 + Ubuntu

En **PowerShell como administrador** (clic derecho → "Ejecutar como administrador"):

```powershell
wsl --install
```

Esto instala WSL2 y Ubuntu. **Reinicia** cuando lo pida. Al volver, se abre Ubuntu y te pide
crear un usuario y contraseña de Linux (anótalos). Luego, en PowerShell:

```powershell
wsl -l -v
```

**Qué deberías ver:** `Ubuntu` con `VERSION 2`. Si dijera VERSION 1, ejecuta
`wsl --set-version Ubuntu 2`.

## 3. Verificar que la GPU se ve dentro de WSL

Abre **Ubuntu** (desde el menú de inicio) y ejecuta:

```bash
nvidia-smi
```

**Qué deberías ver:** la MISMA tabla con la RTX 3060. Si aparece, el puente CUDA Windows→WSL
funciona y ya tienes GPU en Linux. (Si dice "command not found" o no detecta la GPU: el driver
de Windows está viejo o WSL no es la versión 2 → vuelve a los pasos 1 y 2. **No** instales un
driver dentro de Ubuntu.)

## 4. Entorno de Python

Dentro de Ubuntu:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git
python3 -m venv ~/prune-env
source ~/prune-env/bin/activate
python --version          # 3.10+ está bien
```

**Qué deberías ver:** el prompt cambia a `(prune-env)`. Trabaja siempre con el entorno
activado; si abres otra terminal, repite `source ~/prune-env/bin/activate`.

## 5. Instalar PyTorch con CUDA y verificarlo

```bash
pip install --upgrade pip
pip install torch torchvision
```

En Linux, el wheel por defecto ya trae CUDA. Verifica:

```bash
python -c "import torch; print(torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin GPU')"
```

**Qué deberías ver:**
```
2.x.x
CUDA disponible: True
NVIDIA GeForce RTX 3060 Laptop GPU
```

Si `CUDA disponible: False`, instala el wheel explícito desde el selector oficial de
`pytorch.org` (elige Linux + Pip + la versión de CUDA que liste tu `nvidia-smi`), por ejemplo:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.

## 6. Prueba de entrenamiento real (1 paso en GPU)

Confirma que no solo detecta la GPU, sino que **entrena** en ella:

```bash
python - << 'PY'
import torch, torchvision, time
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = torchvision.models.resnet50(weights=None).to(dev)
opt = torch.optim.SGD(m.parameters(), lr=0.01)
scaler = torch.cuda.amp.GradScaler()           # precisión mixta: clave para los 6 GB
x = torch.randn(16, 3, 224, 224, device=dev)   # lote 16 cabe holgado en 6 GB
y = torch.randint(0, 1000, (16,), device=dev)
t = time.time()
for _ in range(3):
    opt.zero_grad()
    with torch.cuda.amp.autocast():
        loss = torch.nn.functional.cross_entropy(m(x), y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
torch.cuda.synchronize()
print("OK: 3 pasos de ResNet-50 en %s en %.1f s | VRAM usada: %.0f MB"
      % (dev, time.time()-t, torch.cuda.max_memory_allocated()/1e6))
PY
```

**Qué deberías ver:** `OK: 3 pasos de ResNet-50 en cuda ...` con un uso de VRAM cómodo por
debajo de 6 GB. Si sale `cuda out of memory`, baja el lote (16 → 8) y repite.

---

## Notas para tu hardware (6 GB + portátil)

- **Precisión mixta (AMP) siempre.** Reduce a la mitad la memoria y acelera; ya está en la
  prueba del paso 6. Para el *fine-tuning* real, lotes de 32–48 con AMP caben; si no, usa
  **acumulación de gradiente** (varios lotes pequeños antes de cada `optimizer.step`).
- **Térmica.** Enchufado, modo de energía de Windows en "Máximo rendimiento" y el perfil de
  ventilador en **Lenovo Vantage → Rendimiento**. Sin eso, la GPU baja frecuencia bajo carga
  sostenida y el entrenamiento se alarga.
- **Disco / datos.** Guarda el dataset de entrenamiento **dentro** del sistema de archivos de
  WSL (tu carpeta `~` en Ubuntu, que es ext4), NO en `/mnt/c/...`: leer desde el disco de
  Windows a través de WSL es mucho más lento y mata la velocidad de entrenamiento.

## (Opcional) Dejar git listo para subir desde el Legion

Si más adelante vas a commitear modelos podados o resultados desde esta máquina:

```bash
git config --global user.name  "Orlando Duque Cantor"
git config --global user.email "TU_ID+ODuqueDvise@users.noreply.github.com"   # tu noreply de GitHub, evita GH007
git config --global pull.rebase true
ssh-keygen -t ed25519 -C "legion-orlando" && cat ~/.ssh/id_ed25519.pub          # pega la llave en GitHub > SSH keys
```

---

Cuando termines, avísame con el resultado del paso 6 (el tiempo y la VRAM usada). Con la
máquina lista, el siguiente paso es bajar el subconjunto de ImageNet de entrenamiento y armar
el flujo de poda estructurada (torch-pruning/DepGraph) + reentrenamiento.
