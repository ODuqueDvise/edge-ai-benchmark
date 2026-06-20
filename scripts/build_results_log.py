#!/usr/bin/env python3
"""Genera results/RESULTS_LOG.md desde los results/*.json (sin entrada manual).

Fuentes: *_run JSON (latencia), acc_*.json (precision), energy_*.json (energia
externa del medidor). Solo stdlib. Ejecutar tras `git pull`.
"""
import glob, json, os, statistics
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "RESULTS_LOG.md")
# Etiquetas explicitas por prefijo sha (modelos canonicos medidos sin model.name).
MODEL_LABELS = {"609015cb": "MobileNetV2 (base)"}
# sha8 -> nombre de archivo del modelo (metadata.model.name); poblado al cargar.
NAMES = {}


def lab(s):
    return MODEL_LABELS.get(s) or NAMES.get(s) or s


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else 0.0


def sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def load():
    lat, acc, ener = defaultdict(list), [], defaultdict(list)
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        md = d.get("metadata", {})
        dev = md.get("device_tag", "?")
        sha = (md.get("model", {}).get("sha256") or "????????")[:8]
        nm = md.get("model", {}).get("name")
        if nm and sha not in NAMES:
            NAMES[sha] = nm
        if "latency_summary" in d:
            lat[(dev, sha, md.get("iters", d["latency_summary"].get("n")))].append(d["latency_summary"])
        if "accuracy" in d:
            a = d["accuracy"]
            acc.append((dev, sha, a.get("n_images"), a.get("top1") or 0, a.get("top5") or 0))
        if "energy" in d:
            ener[(dev, sha)].append(d["energy"])
    return lat, acc, ener


def main():
    lat, acc, ener = load()
    o = []
    o += ["# Registro de resultados (GENERADO)", "",
          "> Generado por `scripts/build_results_log.py` desde `results/*.json`. **No editar a mano.**",
          "> Re-generar tras nuevas corridas (idealmente despues de `git pull`).",
          "> Constantes congeladas: warmup 100, iters 2000, R 5, MAXN (Jetson) / governor performance (RPi), entrada 1,3,224,224.",
          "", "## Latencia", "",
          "| Condicion | Modelo | R | p50 media±desv (ms) | p95 (ms) | p99 (ms) | thr (ips) |",
          "|---|---|---|---|---|---|---|"]
    for (dev, sha, _it), runs in sorted(lat.items()):
        p50 = [r["p50_ms"] for r in runs]
        o.append("| %s | %s | %d | %.3f ± %.3f | %.3f | %.3f | %.1f |" % (
            dev, lab(sha), len(runs), mean(p50), sd(p50),
            mean([r["p95_ms"] for r in runs]), mean([r["p99_ms"] for r in runs]),
            mean([r.get("throughput_ips") for r in runs])))
    o += ["", "## Precision (ImageNet-V2)", "",
          "| Condicion | Modelo | n img | top-1 | top-5 |", "|---|---|---|---|---|"]
    for dev, sha, n, t1, t5 in sorted(acc):
        o.append("| %s | %s | %s | %.4f | %.4f |" % (dev, lab(sha), n, t1, t5))
    o += ["", "## Energia (medidor externo)", "",
          "| Condicion | Modelo | corridas | Pot. media (W) | Energia/inf total (mJ) | Energia/inf neta (mJ) |",
          "|---|---|---|---|---|---|"]
    for (dev, sha), es in sorted(ener.items()):
        net = [x.get("per_inf_net_mj") for x in es]
        net_s = ("%.3f" % mean(net)) if any(v is not None for v in net) else "—"
        o.append("| %s | %s | %d | %.2f | %.3f | %s |" % (
            dev, lab(sha), len(es), mean([x.get("avg_power_w") for x in es]),
            mean([x.get("per_inf_total_mj") for x in es]), net_s))
    open(OUT, "w").write("\n".join(o) + "\n")
    print("Escrito %s (%d latencia, %d precision, %d energia)" %
          (OUT, sum(len(v) for v in lat.values()), len(acc), sum(len(v) for v in ener.values())))


if __name__ == "__main__":
    main()
