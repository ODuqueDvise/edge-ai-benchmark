#!/usr/bin/env python3
"""
Análisis OE3 — comparación GPU vs CPU a través de las técnicas de optimización.

Método (directriz del director, DECISIONS D11):
  - Tendencia central en escala logarítmica (media geométrica de la latencia).
  - Conclusiones por TAMAÑO DE EFECTO e INTERVALOS DE CONFIANZA, no p-valores
    (con miles de inferencias casi todo sale "significativo").
  - La cola se reporta con p50, p95 y p99.
  - Aligned Rank Transform (factorial): se exporta el CSV ordenado por corrida para
    correrlo en R/ARTool (results/oe3_tidy_runs.csv); aquí el peso está en efecto + IC.

Lee results/*.json (latencia con 2000 muestras crudas × R corridas, precisión, energía),
limpia los runs de prueba/gate, y escribe results/OE3_ANALISIS.md + results/oe3_tidy_runs.csv.
Unidad de réplica para los IC: la corrida independiente (R), no la inferencia individual.
"""
import json, glob, os, math, csv
from collections import defaultdict
import numpy as np
from scipy import stats

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SHA_MODEL = {"609015cb": "MobileNetV2", "05e5bc14": "ResNet-50", "c1eac3d6": "MobileNetV2",
             "2161a04a": "ResNet-50", "940aefb8": "ResNet-50", "7be5303c": "MobileNetV2",
             "efffe63b": "ResNet-50", "8a6cdf8c": "MobileNetV2"}
TECH = ["V0", "INT8", "Poda", "Poda+KD"]
DEV = ["jetson-gpu", "jetson-cpu"]
DEVLAB = {"jetson-gpu": "GPU", "jetson-cpu": "CPU"}


def classify(md):
    name = (md.get("model", {}).get("name") or "").lower()
    sha = (md.get("model", {}).get("sha256") or "")[:8]
    var = (md.get("variant") or "").lower()
    model = SHA_MODEL.get(sha) or ("ResNet-50" if "resnet" in name else
            "MobileNetV2" if ("cnn" in name or "mobile" in name) else "?")
    if "pruned_kd" in name: tech = "Poda+KD"
    elif "pruned" in name:  tech = "Poda"
    elif "int8" in name or var == "int8" or sha in ("c1eac3d6", "2161a04a"): tech = "INT8"
    else: tech = "V0"
    return model, tech, name


def load():
    lat = defaultdict(list); acc = {}; ene = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        try: d = json.load(open(f))
        except Exception: continue
        md = d.get("metadata", {}); dev = md.get("device_tag", "?")
        model, tech, name = classify(md)
        if "latency_summary" in d:
            if (d["latency_summary"].get("n") or 0) < 2000: continue           # excluye prueba/gate
            if dev == "jetson-gpu" and name.endswith("_int8"): continue        # QDQ-en-GPU fallido
            raw = d.get("latency_raw_ms")
            if raw: lat[(model, tech, dev)].append((md.get("timestamp_utc", ""), np.asarray(raw, float)))
        if "accuracy" in d:
            a = d["accuracy"]; acc[(model, tech, dev)] = (a.get("top1"), a.get("top5"))
        if "energy" in d:
            ene[(model, tech, dev)] = d["energy"].get("per_inf_net_mj")
    lat5 = {k: [r for _, r in sorted(v, key=lambda x: x[0])[-5:]] for k, v in lat.items()}  # R=5 estricto: 5 más recientes
    return lat5, acc, ene


def geomean(x): return float(np.exp(np.mean(np.log(x))))


def summarize(runs):
    pooled = np.concatenate(runs)
    run_log = np.log([geomean(r) for r in runs])
    return dict(R=len(runs), gm=geomean(pooled),
                p50=float(np.percentile(pooled, 50)), p95=float(np.percentile(pooled, 95)),
                p99=float(np.percentile(pooled, 99)), run_log=run_log)


def ratio_ci(num_log, den_log):
    diff = num_log.mean() - den_log.mean()
    se = math.sqrt(num_log.var(ddof=1)/len(num_log) + den_log.var(ddof=1)/len(den_log))
    df = max(len(num_log) + len(den_log) - 2, 1)
    t = float(stats.t.ppf(0.975, df))
    return math.exp(diff), math.exp(diff - t*se), math.exp(diff + t*se)


def make_figure(S):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    models = ["ResNet-50", "MobileNetV2"]; techs = ["V0", "INT8", "Poda"]
    gaps = {}
    for m in models:
        for t in techs:
            g = S.get((m, t, "jetson-gpu")); c = S.get((m, t, "jetson-cpu"))
            if g and c: gaps[(m, t)] = math.exp(c["run_log"].mean() - g["run_log"].mean())
    x = np.arange(len(models)); width = 0.26
    gray = {"V0": "0.55", "INT8": "0.80", "Poda": "0.25"}
    hatch = {"V0": "", "INT8": "//", "Poda": ".."}
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for i, t in enumerate(techs):
        vals = [gaps.get((m, t), 0) for m in models]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=t, color=gray[t],
                      edgecolor="black", hatch=hatch[t], linewidth=0.8)
        for b, v in zip(bars, vals):
            if v: ax.text(b.get_x() + b.get_width() / 2, v + 0.12, "%.1f×" % v,
                          ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Brecha de latencia GPU↔CPU (×)")
    ax.legend(title="Técnica", frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, (max(gaps.values()) if gaps else 1) * 1.18)
    fig.tight_layout()
    png = os.path.join(RESULTS, "oe3_brecha_gpu_cpu.png")
    fig.savefig(png, dpi=200); fig.savefig(os.path.join(RESULTS, "oe3_brecha_gpu_cpu.pdf"))
    return png


def main():
    lat, acc, ene = load()
    S = {k: summarize(v) for k, v in lat.items()}
    out = []
    def w(s=""): out.append(s); print(s)

    w("# Análisis OE3 — GPU vs CPU a través de las técnicas (Jetson)\n")
    w("> Tendencia central = media geométrica (escala log). IC al 95% sobre las R corridas")
    w("> independientes (t de Student). Cola = percentiles de las muestras crudas combinadas.")
    w("> rpi-cpu pendiente (Luis). Generado por `scripts/analyze_oe3.py`.\n")

    w("## 1. Resumen por condición\n")
    w("| Modelo | Técnica | Disp. | R | Media geom. (ms) | p50 | p95 | p99 | top-1 | E. neta (mJ) |")
    w("|---|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for model in ["ResNet-50", "MobileNetV2"]:
        for tech in TECH:
            for dev in DEV:
                s = S.get((model, tech, dev))
                if not s: continue
                t1 = acc.get((model, tech, dev), (None, None))[0]
                ej = ene.get((model, tech, dev))
                w("| %s | %s | %s | %d | %.3f | %.3f | %.3f | %.3f | %s | %s |" % (
                    model, tech, DEVLAB[dev], s["R"], s["gm"], s["p50"], s["p95"], s["p99"],
                    "%.3f" % t1 if t1 else "—", "%.1f" % ej if ej else "—"))

    w("\n## 2. Brecha GPU↔CPU por técnica (tamaño de efecto e IC95)\n")
    w("> Razón = latencia(CPU) / latencia(GPU). Cuánto más rápida es la GPU.\n")
    w("| Modelo | Técnica | Brecha GPU↔CPU | IC95 |")
    w("|---|---|--:|--:|")
    for model in ["ResNet-50", "MobileNetV2"]:
        for tech in ["V0", "INT8", "Poda"]:
            g = S.get((model, tech, "jetson-gpu")); c = S.get((model, tech, "jetson-cpu"))
            if not (g and c): continue
            r, lo, hi = ratio_ci(c["run_log"], g["run_log"])
            w("| %s | %s | %.2f× | [%.2f, %.2f] |" % (model, tech, r, lo, hi))

    w("\n## 3. Efecto de cada técnica sobre la latencia (vs V0), por dispositivo\n")
    w("> Razón = latencia(V0) / latencia(técnica). >1 acelera; <1 frena.\n")
    w("| Modelo | Disp. | Técnica | Aceleración vs V0 | IC95 |")
    w("|---|---|---|--:|--:|")
    for model in ["ResNet-50", "MobileNetV2"]:
        for dev in DEV:
            base = S.get((model, "V0", dev))
            if not base: continue
            for tech in ["INT8", "Poda"]:
                s = S.get((model, tech, dev))
                if not s: continue
                r, lo, hi = ratio_ci(base["run_log"], s["run_log"])
                w("| %s | %s | %s | %.2f× | [%.2f, %.2f] |" % (model, DEVLAB[dev], tech, r, lo, hi))

    w("\n## 4. Precisión (ImageNet-V2, top-1) por técnica\n")
    w("| Modelo | V0 | INT8 | Poda (FT) | Poda+KD |")
    w("|---|--:|--:|--:|--:|")
    for model in ["ResNet-50", "MobileNetV2"]:
        def a(t):
            for dev in DEV:
                v = acc.get((model, t, dev))
                if v and v[0]: return "%.3f" % v[0]
            return "—"
        w("| %s | %s | %s | %s | %s |" % (model, a("V0"), a("INT8"), a("Poda"), a("Poda+KD")))

    # CSV ordenado por corrida (para ART en R/ARTool)
    csvp = os.path.join(RESULTS, "oe3_tidy_runs.csv")
    with open(csvp, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["modelo", "tecnica", "dispositivo", "corrida", "geomean_ms", "log_geomean_ms"])
        for (model, tech, dev), s in S.items():
            for i, lg in enumerate(s["run_log"]):
                wr.writerow([model, tech, DEVLAB[dev], i + 1, "%.5f" % math.exp(lg), "%.6f" % lg])
    w("\n---\nCSV ordenado por corrida para ART/R: `results/oe3_tidy_runs.csv` (%d filas)." %
      sum(len(s["run_log"]) for s in S.values()))

    try:
        png = make_figure(S)
        w("Figura de la brecha GPU↔CPU: `results/%s` (+ .pdf)." % os.path.basename(png))
    except Exception as e:
        w("[figura omitida: %s]" % e)

    with open(os.path.join(RESULTS, "OE3_ANALISIS.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n[escrito] results/OE3_ANALISIS.md")


if __name__ == "__main__":
    main()
