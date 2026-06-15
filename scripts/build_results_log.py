#!/usr/bin/env python3
"""Genera results/RESULTS_LOG.md desde los results/*.json (sin entrada manual).

Los JSON crudos son la fuente de verdad; el log se DERIVA de ellos. Agrega las R
corridas de latencia (media +/- desv del p50) por condicion y modelo, lista la
precision y la potencia media (referencia interna). Sin dependencias externas
(solo stdlib), asi corre dentro o fuera del venv. Ejecutar tras `git pull`.
"""
import glob, json, os, statistics
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "RESULTS_LOG.md")

MODEL_LABELS = {"609015cb": "V0 base (MobileNetV2)"}   # checksum(8) -> variante


def lab(s):
    return MODEL_LABELS.get(s, s)


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def sd(xs):
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
        if "latency_summary" in d:
            s = d["latency_summary"]
            lat[(dev, sha, md.get("iters", s.get("n")))].append(s)
            p = d.get("power_internal_crossref", {}) or {}
            if p.get("avg_power_w"):
                ener[(dev, sha)].append(p["avg_power_w"])
        if "accuracy" in d:
            a = d["accuracy"]
            acc.append((dev, sha, a.get("n_images"), a.get("top1") or 0, a.get("top5") or 0))
    return lat, acc, ener


def main():
    lat, acc, ener = load()
    o = []
    o += ["# Registro de resultados (GENERADO)", "",
          "> Generado por `scripts/build_results_log.py` desde `results/*.json`. **No editar a mano.**",
          "> Re-generar tras nuevas corridas (idealmente despues de `git pull`, para incluir ambos equipos).",
          "> Constantes congeladas: warmup 100, iters 2000, R 5, MAXN (Jetson) / governor performance (RPi), entrada 1,3,224,224.",
          "", "## Latencia", "",
          "| Condicion | Modelo | R | p50 media±desv (ms) | p95 media (ms) | p99 media (ms) | thr media (ips) |",
          "|---|---|---|---|---|---|---|"]
    for (dev, sha, _it), runs in sorted(lat.items()):
        p50 = [r["p50_ms"] for r in runs]
        thrs = [r["throughput_ips"] for r in runs if r.get("throughput_ips")]
        o.append("| %s | %s | %d | %.3f ± %.3f | %.3f | %.3f | %.1f |" % (
            dev, lab(sha), len(runs), mean(p50), sd(p50),
            mean([r["p95_ms"] for r in runs]), mean([r["p99_ms"] for r in runs]), mean(thrs)))
    o += ["", "## Precision (ImageNet-V2)", "",
          "| Condicion | Modelo | n img | top-1 | top-5 |", "|---|---|---|---|---|"]
    for dev, sha, n, t1, t5 in sorted(acc):
        o.append("| %s | %s | %s | %.4f | %.4f |" % (dev, lab(sha), n, t1, t5))
    o += ["", "## Energia (potencia media, referencia interna; medidor externo pendiente)", "",
          "| Condicion | Modelo | pot. media (W, ref. interna) |", "|---|---|---|"]
    for (dev, sha), ws in sorted(ener.items()):
        o.append("| %s | %s | %.2f |" % (dev, lab(sha), mean(ws)))
    open(OUT, "w").write("\n".join(o) + "\n")
    print("Escrito %s (%d corridas latencia, %d precision)" % (OUT, sum(len(v) for v in lat.values()), len(acc)))


if __name__ == "__main__":
    main()
