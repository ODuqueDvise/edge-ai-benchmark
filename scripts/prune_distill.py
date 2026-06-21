#!/usr/bin/env python3
"""
Recuperación de la poda por DESTILACIÓN (KD) — alternativa al fine-tuning normal de
`prune_finetune.py`, para corregir el sobre-ajuste con datos de recuperación limitados (D16).

Idea: el modelo SIN podar (maestro, congelado) genera objetivos suaves; el modelo podado
(estudiante) se entrena con una mezcla de pérdida de destilación (KL sobre logits suavizados)
y entropía cruzada dura. Los objetivos suaves del maestro aportan mucha más información por
imagen que las etiquetas duras, así que el estudiante recupera mejor desde el mismo subconjunto
de 100/clase. Dos correcciones frente a `prune_finetune.py`:
  - maestro + KD (regulariza, reduce el sobre-ajuste);
  - se exporta la ULTIMA época, NO la "mejor" por validación de entrenamiento (que premiaba al
    modelo más sobre-ajustado). La precisión real se mide en ImageNet-V2 sobre la Jetson.

Estructura de poda idéntica a `prune_finetune.py` (torch-pruning, L1 global, opset 18, FP32).
Requiere (Legion, prune-env): torch, torchvision, torch-pruning, onnx, onnxscript, onnxruntime, pillow.

Ejemplos:
  python scripts/prune_distill.py --model resnet50 --target-macs 0.5 \
      --data ~/imagenet_train_subset --epochs 15 --temp 4 --alpha 0.8 \
      --out models/resnet50_pruned_kd.onnx --ckpt ~/ck_resnet50_kd.pth

  python scripts/prune_distill.py --model mobilenet_v2 --target-macs 0.7 \
      --data ~/imagenet_train_subset --epochs 15 --temp 4 --alpha 0.8 \
      --out models/cnn_pruned_kd.onnx --ckpt ~/ck_mobilenet_v2_kd.pth

  # reanudar: agrega --resume (mismo --ckpt). Si hay OOM por el maestro+estudiante: --batch 32.
"""
import argparse, os, time, random, hashlib
import torch, torch.nn as nn, torch.nn.functional as F
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
    def __getitem__(self, i):
        try:
            return super().__getitem__(i)
        except Exception:
            return super().__getitem__(random.randrange(len(self)))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); correct = total = 0
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        pred = model(x).argmax(1).cpu()
        correct += (pred == y).sum().item(); total += y.numel()
    return 100.0 * correct / max(total, 1)


def build_loaders(data, batch, workers):
    train_tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    val_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    tr_ds = RobustImageFolder(data, transform=train_tf)
    va_ds = RobustImageFolder(data, transform=val_tf)
    n = len(tr_ds); idx = list(range(n)); random.Random(0).shuffle(idx); cut = int(0.95 * n)
    tr = Subset(tr_ds, idx[:cut]); va = Subset(va_ds, idx[cut:])
    tl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    vl = DataLoader(va, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    log("dataset: %d (%d train / %d val) en %d clases" % (n, len(tr), len(va), len(tr_ds.classes)))
    return tl, vl


def prune_to_target(model, target, dev, steps=16):
    model.to(dev).eval()
    ex = torch.randn(1, 3, 224, 224, device=dev)
    ignored = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 1000]
    imp = tp.importance.MagnitudeImportance(p=1)
    base_macs, base_p = tp.utils.count_ops_and_params(model, ex)
    pr = tp.pruner.MetaPruner(model, ex, importance=imp, pruning_ratio=0.95,
                              ignored_layers=ignored, global_pruning=True, iterative_steps=steps)
    macs = base_macs; p = base_p; used = 0
    for i in range(steps):
        pr.step(); used = i + 1
        macs, p = tp.utils.count_ops_and_params(model, ex)
        if macs <= target * base_macs:
            break
    log("poda: MACs %.2fG->%.2fG (%.0f%% del original) | params %.2fM->%.2fM | pasos=%d"
        % (base_macs / 1e9, macs / 1e9, 100 * macs / base_macs, base_p / 1e6, p / 1e6, used))
    return model


def export_onnx(model, out, dev):
    model.eval().to(dev); os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.onnx.export(model, torch.randn(1, 3, 224, 224, device=dev), out, opset_version=18,
                      input_names=["input"], output_names=["output"])
    import onnx
    m = onnx.load(out); onnx.save_model(m, out, save_as_external_data=False)
    if os.path.exists(out + ".data"): os.remove(out + ".data")
    log("ONNX exportado:", out, "(%.1f MB)" % (os.path.getsize(out) / 1e6)); log("SHA-256:", sha256(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(WEIGHTS))
    ap.add_argument("--data", default=os.path.expanduser("~/imagenet_train_subset"))
    ap.add_argument("--target-macs", type=float, required=True, help="fracción de MACs a conservar")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temp", type=float, default=4.0, help="temperatura de destilación (T)")
    ap.add_argument("--alpha", type=float, default=0.8, help="peso del término KD (resto: CE dura)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("dispositivo:", dev, "| modelo:", a.model, "| conservar MACs:", a.target_macs,
        "| T:", a.temp, "| alpha:", a.alpha)
    tl, vl = build_loaders(os.path.expanduser(a.data), a.batch, a.workers)

    ctor, w = WEIGHTS[a.model]
    teacher = ctor(weights=w).eval().to(dev)          # maestro: SIN podar, congelado
    for p in teacher.parameters():
        p.requires_grad_(False)

    ck = None; start_epoch = 0
    if a.resume and os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        student = ck["model"].to(dev); start_epoch = ck["epoch"] + 1
        log("reanudando KD desde epoca %d" % start_epoch)
    else:
        student = prune_to_target(ctor(weights=w), a.target_macs, dev)   # estudiante: podado fresco

    opt = torch.optim.SGD(student.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler()
    if ck is not None:
        try:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        except Exception as e:
            log("[aviso] no se restauró opt/sched:", e)

    Tt = a.temp; alpha = a.alpha
    for epoch in range(start_epoch, a.epochs):
        student.train(); t0 = time.time(); seen = 0
        for bi, (x, y) in enumerate(tl):
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            with torch.no_grad():
                t_logits = teacher(x)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                s_logits = student(x)
                ce = F.cross_entropy(s_logits, y)
                kd = F.kl_div(F.log_softmax(s_logits / Tt, dim=1),
                              F.softmax(t_logits / Tt, dim=1),
                              reduction="batchmean") * (Tt * Tt)
                loss = alpha * kd + (1 - alpha) * ce
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            seen += y.numel()
            if bi % 100 == 0:
                log("  epoca %d  paso %d/%d  loss %.3f (kd %.3f ce %.3f)  %.0f img/s"
                    % (epoch, bi, len(tl), loss.item(), kd.item(), ce.item(), seen / max(time.time() - t0, 1e-9)))
        sched.step()
        acc = evaluate(student, vl, dev)
        log("epoca %d/%d  val top-1 (train-dist, solo monitoreo) %.2f%%  lr %.4g  (%.1f min)"
            % (epoch, a.epochs - 1, acc, sched.get_last_lr()[0], (time.time() - t0) / 60))
        torch.save({"model": student, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch}, a.ckpt)

    log("exportando el modelo de la ULTIMA época (no se selecciona por val de train; la precisión real va en V2 sobre la Jetson)")
    export_onnx(student, a.out, dev)
    log("LISTO. ONNX en %s. Copiar a la Jetson y medir con el mismo procedimiento de los podados." % a.out)


if __name__ == "__main__":
    main()
