#!/usr/bin/env python3
"""
Poda estructurada (DepGraph) + reentrenamiento de recuperacion + export a ONNX.
Segunda tecnica del OE1 (EXP-07/08 MobileNetV2, EXP-19/20 ResNet-50).

Flujo, en una sola corrida por modelo (lanzar con nohup en el Legion):
  1. Carga el modelo preentrenado de torchvision (mismos pesos DEFAULT que la base).
  2. Poda estructurada de canales con torch-pruning (importancia L1, poda global),
     iterando hasta alcanzar una fraccion objetivo de MACs. NO toca el clasificador
     (mantiene 1000 salidas). Reporta MACs/parametros antes y despues.
  3. Reentrenamiento de recuperacion sobre el subconjunto de ImageNet (AMP, SGD+coseno,
     label smoothing). Checkpoint por epoca (modelo podado completo) -> resume si se corta.
  4. Export a ONNX FP32 con los MISMOS ajustes que la linea base (opset 18, un solo
     archivo, input/output, SHA-256), para que la comparacion contra V0 sea limpia.

Requiere (en el Legion, prune-env): torch, torchvision, torch-pruning, onnx, onnxruntime, pillow.

--target-macs = fraccion de MACs a CONSERVAR. 0.5 -> reducir 50% (ResNet). 0.7 -> reducir 30% (MobileNet).

Ejemplos:
  python scripts/prune_finetune.py --model resnet50 --target-macs 0.5 \
      --data ~/imagenet_train_subset --epochs 15 \
      --out models/resnet50_pruned.onnx --ckpt ~/ck_resnet50.pth

  python scripts/prune_finetune.py --model mobilenet_v2 --target-macs 0.7 \
      --data ~/imagenet_train_subset --epochs 15 \
      --out models/cnn_pruned.onnx --ckpt ~/ck_mobilenet_v2.pth

  # reanudar tras una caida: agrega --resume (mismo --ckpt)
"""
import argparse, os, time, random, hashlib
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch_pruning as tp

WEIGHTS = {
    "resnet50":     (M.resnet50,     M.ResNet50_Weights.DEFAULT),
    "mobilenet_v2": (M.mobilenet_v2, M.MobileNet_V2_Weights.DEFAULT),
}
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]


def log(*a): print(*a, flush=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class RobustImageFolder(torchvision.datasets.ImageFolder):
    # ante un JPG corrupto (cortes abruptos de la descarga), reusa otro indice al azar
    def __getitem__(self, i):
        try:
            return super().__getitem__(i)
        except Exception:
            return super().__getitem__(random.randrange(len(self)))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        pred = model(x).argmax(1).cpu()
        correct += (pred == y).sum().item(); total += y.numel()
    return 100.0 * correct / max(total, 1)


def build_loaders(data, batch, workers):
    train_tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    val_tf   = T.Compose([T.Resize(256), T.CenterCrop(224),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    train_ds = RobustImageFolder(data, transform=train_tf)
    val_ds   = RobustImageFolder(data, transform=val_tf)
    n = len(train_ds)
    idx = list(range(n)); random.Random(0).shuffle(idx)
    cut = int(0.95 * n)
    tr = Subset(train_ds, idx[:cut]); va = Subset(val_ds, idx[cut:])
    tl = DataLoader(tr, batch_size=batch, shuffle=True,  num_workers=workers,
                    pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    vl = DataLoader(va, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    log("dataset: %d imagenes (%d train / %d val) en %d clases"
        % (n, len(tr), len(va), len(train_ds.classes)))
    return tl, vl


def prune_to_target(model, target_macs_frac, dev, steps=16):
    model.to(dev).eval()
    example = torch.randn(1, 3, 224, 224, device=dev)
    ignored = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 1000]
    imp = tp.importance.MagnitudeImportance(p=1)                      # L1
    base_macs, base_p = tp.utils.count_ops_and_params(model, example)
    pruner = tp.pruner.MetaPruner(model, example, importance=imp,
                                  pruning_ratio=0.95, ignored_layers=ignored,
                                  global_pruning=True, iterative_steps=steps)
    macs = base_macs; p = base_p; used = 0
    for i in range(steps):
        pruner.step(); used = i + 1
        macs, p = tp.utils.count_ops_and_params(model, example)
        if macs <= target_macs_frac * base_macs:
            break
    log("poda: MACs %.2fG -> %.2fG (%.0f%% del original) | params %.2fM -> %.2fM | pasos=%d"
        % (base_macs / 1e9, macs / 1e9, 100 * macs / base_macs, base_p / 1e6, p / 1e6, used))
    return model


def export_onnx(model, out, dev):
    model.eval().to(dev)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224, device=dev)
    torch.onnx.export(model, dummy, out, opset_version=18,
                      input_names=["input"], output_names=["output"])
    import onnx
    m = onnx.load(out); onnx.save_model(m, out, save_as_external_data=False)
    if os.path.exists(out + ".data"): os.remove(out + ".data")
    log("ONNX exportado:", out, "(%.1f MB)" % (os.path.getsize(out) / 1e6))
    log("SHA-256:", sha256(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(WEIGHTS))
    ap.add_argument("--data", default=os.path.expanduser("~/imagenet_train_subset"))
    ap.add_argument("--target-macs", type=float, required=True,
                    help="fraccion de MACs a CONSERVAR (0.5 = reducir 50%%)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True, help="ruta ONNX de salida")
    ap.add_argument("--ckpt", required=True, help="ruta del checkpoint (para resume)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("dispositivo:", dev, "| modelo:", a.model, "| conservar MACs:", a.target_macs)
    tl, vl = build_loaders(os.path.expanduser(a.data), a.batch, a.workers)

    ck = None
    start_epoch, best = 0, -1.0
    if a.resume and os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        model = ck["model"].to(dev)
        start_epoch = ck["epoch"] + 1; best = ck.get("best", -1.0)
        log("reanudando desde epoca %d (mejor val=%.2f%%)" % (start_epoch, best))
    else:
        ctor, w = WEIGHTS[a.model]
        model = prune_to_target(ctor(weights=w), a.target_macs, dev)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler()
    if ck is not None:
        try:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
        except Exception as e:
            log("[aviso] no se restauro optimizador/scheduler:", e)

    for epoch in range(start_epoch, a.epochs):
        model.train(); t0 = time.time(); seen = 0
        for bi, (x, y) in enumerate(tl):
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            seen += y.numel()
            if bi % 100 == 0:
                log("  epoca %d  paso %d/%d  loss %.3f  %.0f img/s"
                    % (epoch, bi, len(tl), loss.item(), seen / max(time.time() - t0, 1e-9)))
        sched.step()
        acc = evaluate(model, vl, dev)
        log("epoca %d/%d  val top-1 %.2f%%  lr %.4g  (%.1f min)"
            % (epoch, a.epochs - 1, acc, sched.get_last_lr()[0], (time.time() - t0) / 60))
        torch.save({"model": model, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": max(best, acc)}, a.ckpt)
        if acc > best:
            best = acc
            torch.save({"model": model, "epoch": epoch, "best": best}, a.ckpt + ".best")
            log("  nuevo mejor (%.2f%%) -> %s" % (best, a.ckpt + ".best"))

    bestck = a.ckpt + ".best"
    if os.path.exists(bestck):
        model = torch.load(bestck, map_location=dev, weights_only=False)["model"].to(dev)
        log("exportando el mejor checkpoint (val=%.2f%%)" % best)
    export_onnx(model, a.out, dev)
    log("LISTO. ONNX en %s. Siguiente: copiarlo a la Jetson y medir (EXP-07/08 o 19/20)." % a.out)


if __name__ == "__main__":
    main()
