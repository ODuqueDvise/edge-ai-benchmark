#!/usr/bin/env python3
"""
Paso 0 de la fase de poda — verificacion del mapeo de clases.

Confirma que el indice de clase del subconjunto descargado (carpetas 0000-0999,
indice de Hugging Face) coincide con el orden de clases que producen los modelos
de torchvision. Si no coincidieran, el reentrenamiento de recuperacion aprenderia
con etiquetas cruzadas y todo el resultado seria invalido.

Metodo: evaluar un modelo preentrenado de torchvision sobre una muestra del
subconjunto, usando el nombre de la carpeta como etiqueta verdadera.
  - Si la exactitud sale alta  -> el mapeo COINCIDE, se puede entrenar.
  - Si sale cercana a lo aleatorio (~0.1%) -> el orden NO coincide, hay que reordenar.

Como son imagenes de entrenamiento (el modelo vio una distribucion parecida), con
el mapeo correcto la top-1 deberia ser muy alta (>80% en ResNet-50).

Uso (en el Legion, dentro de prune-env):
  python verify_class_mapping.py --model resnet50     --per-class 4
  python verify_class_mapping.py --model mobilenet_v2 --per-class 4
"""
import argparse, os, glob, random
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolera JPGs truncados por los cortes abruptos de la descarga
import torchvision.models as M

# .DEFAULT siempre apunta a los mejores pesos disponibles en tu version de torchvision
WEIGHTS = {
    "resnet50":     (M.resnet50,     M.ResNet50_Weights.DEFAULT),
    "mobilenet_v2": (M.mobilenet_v2, M.MobileNet_V2_Weights.DEFAULT),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(WEIGHTS))
    ap.add_argument("--data", default=os.path.expanduser("~/imagenet_train_subset"))
    ap.add_argument("--per-class", type=int, default=4, help="imagenes por clase a muestrear")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ctor, w = WEIGHTS[a.model]
    model = ctor(weights=w).eval().to(dev)
    tf = w.transforms()           # preprocesamiento EXACTO del modelo (resize/crop/normalize)

    data = os.path.expanduser(a.data)
    classes = sorted(d for d in os.listdir(data) if d.isdigit())
    samples = []
    rng = random.Random(0)
    for c in classes:
        fs = glob.glob(os.path.join(data, c, "*.jpg"))
        rng.shuffle(fs)
        for f in fs[:a.per_class]:
            samples.append((f, int(c)))
    print("modelo: %s | muestras: %d en %d clases | dispositivo: %s"
          % (a.model, len(samples), len(classes), dev), flush=True)

    correct = top5 = total = 0
    batch_imgs, batch_labs = [], []

    @torch.no_grad()
    def flush():
        nonlocal correct, top5, total
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(dev)
        out = model(x)
        p5 = out.topk(5, dim=1).indices.cpu()
        y = torch.tensor(batch_labs)
        correct += (p5[:, 0] == y).sum().item()
        top5    += (p5 == y[:, None]).any(dim=1).sum().item()
        total   += len(batch_labs)
        batch_imgs.clear(); batch_labs.clear()

    for f, lab in samples:
        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            continue                      # salta (y delata) cualquier JPG corrupto
        batch_imgs.append(tf(img)); batch_labs.append(lab)
        if len(batch_imgs) >= a.batch:
            flush()
    flush()

    acc1 = 100.0 * correct / total if total else 0.0
    acc5 = 100.0 * top5    / total if total else 0.0
    print("top-1: %.1f%% | top-5: %.1f%% (n=%d)" % (acc1, acc5, total), flush=True)
    if acc1 >= 50:
        print("VEREDICTO: el mapeo COINCIDE. Indice de carpeta = clase de torchvision. Se puede entrenar.")
    elif acc1 <= 5:
        print("VEREDICTO: el mapeo NO coincide (exactitud ~aleatoria). Hay que reordenar las etiquetas antes de entrenar.")
    else:
        print("VEREDICTO: ambiguo (%.1f%%). Sube --per-class y revisa el preprocesamiento." % acc1)

if __name__ == "__main__":
    main()
