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
import json, glob, os, math, csv, datetime as dt
from collections import defaultdict
import numpy as np
from scipy import stats

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SHA_MODEL = {"609015cb": "MobileNetV2", "05e5bc14": "ResNet-50", "c1eac3d6": "MobileNetV2",
             "2161a04a": "ResNet-50", "940aefb8": "ResNet-50", "7be5303c": "MobileNetV2",
             "efffe63b": "ResNet-50", "8a6cdf8c": "MobileNetV2"}
TECH = ["V0", "INT8", "Poda", "Poda+KD"]
DEV = ["jetson-gpu", "jetson-cpu", "rpi-cpu"]
DEVLAB = {"jetson-gpu": "GPU", "jetson-cpu": "CPU", "rpi-cpu": "RPi-CPU"}
# Campaña oficial de latencia rpi-cpu: fría, auditada y de ENTORNO ÚNICO sobre este kernel
# (BITACORA 14/19 jul 2026). Las corridas rpi con kernel 1009 (22 jun / 7 jul) quedan fuera:
# el salto 1009→1014 movió el ResNet-50 sin podar +8-10% y no se mezclan entornos en razones.
RPI_RELEASE = "7.0.0-1014-raspi"
# Separación entre campañas (D21). Dentro de una misma sesión de medición las corridas van
# seguidas (10 s a 2 min entre ellas); un hueco mayor a esto indica OTRA sesión, con otro
# estado del equipo. La política de ENTORNO ÚNICO POR COLUMNA se aplica ahora también a la
# Jetson, no solo a la RPi: una condición se estima con UNA campaña, nunca con una mezcla.
SESSION_GAP_S = 30 * 60


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
            if dev == "rpi-cpu" and md.get("platform", {}).get("release") != RPI_RELEASE:
                continue                                                       # entorno único (kernel 1014)
            raw = d.get("latency_raw_ms")
            if raw: lat[(model, tech, dev)].append((md.get("timestamp_utc", ""), np.asarray(raw, float)))
        if "accuracy" in d:
            a = d["accuracy"]; acc[(model, tech, dev)] = (a.get("top1"), a.get("top5"))
        if "energy" in d:
            ene[(model, tech, dev)] = d["energy"].get("per_inf_net_mj")
    # Jetson: UNA campaña por condición, R=5 (ver pick_campaign). RPi: TODAS las corridas del
    # kernel oficial — la multimodalidad entre procesos (BITACORA 14 jul) se absorbe con R alto.
    kept = {}; camp = {}
    for k, v in lat.items():
        if k[2] == "rpi-cpu":
            kept[k] = [r for _, r in sorted(v, key=lambda x: x[0])]
            camp[k] = "kernel %s (todas las corridas oficiales)" % RPI_RELEASE
        else:
            kept[k], camp[k] = pick_campaign(v)
    return kept, acc, ene, camp


def pick_campaign(v):
    """Selecciona UNA campaña de medición por condición — política de ENTORNO ÚNICO (D21).

    Agrupa las corridas por cercanía temporal (hueco > SESSION_GAP_S ⇒ sesión distinta) y
    devuelve el grupo más numeroso, desempatando por el más reciente, recortado a 5 corridas.

    Motivo: la regla anterior ("las 5 más recientes") podía mezclar sesiones. En MobileNetV2
    sin optimizar tomaba cuatro corridas de la campaña del 15 jun más una recomprobación
    suelta del 20 jun, y descartaba una del 15 para hacerle sitio. Esa corrida aislada
    (gm 2.908 ms frente a 2.460-2.482 de la campaña) subía el CV de 0.35 % a 7.66 % y era la
    causa completa del IC anómalo [4.58, 5.33] de esa condición. El criterio selecciona
    CAMPAÑAS, no valores: no mira la latencia, solo la estructura temporal.

    Devuelve (lista de arrays de latencia, etiqueta de la campaña elegida).
    """
    ts = [dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t, _ in v]
    order = sorted(range(len(v)), key=lambda i: ts[i])
    groups = [[order[0]]]
    for prev, cur in zip(order, order[1:]):
        if (ts[cur] - ts[prev]).total_seconds() > SESSION_GAP_S:
            groups.append([])
        groups[-1].append(cur)
    best = max(groups, key=lambda g: (len(g), ts[g[-1]]))[-5:]
    label = "%s UTC" % ts[best[0]].strftime("%Y-%m-%d %H:%M")
    if len(groups) > 1:
        label += " (%d campañas disponibles; se descartan %d corrida(s) de otras sesiones)" % (
            len(groups), len(v) - len(best))
    return [v[i][1] for i in best], label


def geomean(x): return float(np.exp(np.mean(np.log(x))))


def summarize(runs):
    """Resume una condición. Distingue explícitamente las dos familias de estadístico:

      gm_runs  — TENDENCIA CENTRAL: media geométrica de las medias geométricas por corrida.
                 Es la única base de las razones y de sus IC (unidad de réplica = corrida).
      p50/95/99 — DISTRIBUCIÓN: percentiles sobre las muestras crudas combinadas. Describen
                 la cola; NO reproducen las razones (la asimetría hace gm ≠ p50).
    """
    pooled = np.concatenate(runs)
    per_run = np.array([geomean(r) for r in runs])
    run_log = np.log(per_run)
    return dict(R=len(runs), gm_runs=float(np.exp(run_log.mean())), gm_pooled=geomean(pooled),
                cv_runs=float(per_run.std(ddof=1) / per_run.mean() * 100) if len(runs) > 1 else 0.0,
                p50=float(np.percentile(pooled, 50)), p95=float(np.percentile(pooled, 95)),
                p99=float(np.percentile(pooled, 99)), run_log=run_log)


def ratio_ci(num_log, den_log):
    diff = num_log.mean() - den_log.mean()
    se = math.sqrt(num_log.var(ddof=1)/len(num_log) + den_log.var(ddof=1)/len(den_log))
    df = max(len(num_log) + len(den_log) - 2, 1)
    t = float(stats.t.ppf(0.975, df))
    return math.exp(diff), math.exp(diff - t*se), math.exp(diff + t*se)


def _fit_rss(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(((y - X @ coef) ** 2).sum())


def art_anova(rows):
    """ART factorial 2 vías (Wobbrock 2011) sobre (dispositivo, técnica, log-geomean por corrida).

    Alinea por efecto (residuos + estimación del efecto con medias de celda), rankea y corre
    ANOVA Tipo III (codificación de efectos) sobre los rangos. Devuelve por efecto:
    (nombre, gl1, gl2, F, p, eta² parcial). Soporta celdas desbalanceadas (R distinto por celda).
    """
    A = sorted({r[0] for r in rows}); B = sorted({r[1] for r in rows})
    y = np.array([r[2] for r in rows], float)
    ai = np.array([A.index(r[0]) for r in rows]); bi = np.array([B.index(r[1]) for r in rows])

    def eff(idx, k):                      # codificación de efectos (suma cero)
        M = np.zeros((len(idx), k - 1))
        for j in range(k - 1):
            M[idx == j, j] = 1.0
        M[idx == k - 1, :] = -1.0
        return M

    XA = eff(ai, len(A)); XB = eff(bi, len(B))
    XAB = np.hstack([(XA[:, i] * XB[:, j])[:, None]
                     for i in range(XA.shape[1]) for j in range(XB.shape[1])])
    one = np.ones((len(y), 1))
    cell = {(a, b): y[(ai == a) & (bi == b)].mean() for a in range(len(A)) for b in range(len(B))}
    mu = float(np.mean(list(cell.values())))
    Am = {a: np.mean([cell[(a, b)] for b in range(len(B))]) for a in range(len(A))}
    Bm = {b: np.mean([cell[(a, b)] for a in range(len(A))]) for b in range(len(B))}
    resid = y - np.array([cell[(a, b)] for a, b in zip(ai, bi)])
    aligned = {
        "dispositivo": resid + np.array([Am[a] - mu for a in ai]),
        "técnica":     resid + np.array([Bm[b] - mu for b in bi]),
        "disp×téc":    resid + np.array([cell[(a, b)] - Am[a] - Bm[b] + mu for a, b in zip(ai, bi)]),
    }
    dfx = {"dispositivo": len(A) - 1, "técnica": len(B) - 1, "disp×téc": (len(A) - 1) * (len(B) - 1)}
    Xfull = np.hstack([one, XA, XB, XAB])
    out = []
    for name, yal in aligned.items():
        rk = stats.rankdata(yal)
        Xred = {"dispositivo": np.hstack([one, XB, XAB]),
                "técnica":     np.hstack([one, XA, XAB]),
                "disp×téc":    np.hstack([one, XA, XB])}[name]
        rss_f = _fit_rss(Xfull, rk); rss_r = _fit_rss(Xred, rk)
        df1 = dfx[name]; df2 = len(y) - Xfull.shape[1]
        F = ((rss_r - rss_f) / df1) / (rss_f / df2)
        p = float(stats.f.sf(F, df1, df2))
        eta = (rss_r - rss_f) / rss_r if rss_r > 0 else float("nan")
        out.append((name, df1, df2, F, p, eta))
    return out


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


def make_figure_despliegue(S):
    """Brecha de despliegue: acelerador embebido (Jetson-GPU) vs CPU de placa (RPi 5)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    models = ["ResNet-50", "MobileNetV2"]; techs = ["V0", "INT8", "Poda"]
    gaps = {}
    for m in models:
        for t in techs:
            g = S.get((m, t, "jetson-gpu")); c = S.get((m, t, "rpi-cpu"))
            if g and c: gaps[(m, t)] = math.exp(c["run_log"].mean() - g["run_log"].mean())
    if not gaps: return None
    x = np.arange(len(models)); width = 0.26
    gray = {"V0": "0.55", "INT8": "0.80", "Poda": "0.25"}
    hatch = {"V0": "", "INT8": "//", "Poda": ".."}
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for i, t in enumerate(techs):
        vals = [gaps.get((m, t), 0) for m in models]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=t, color=gray[t],
                      edgecolor="black", hatch=hatch[t], linewidth=0.8)
        for b, v in zip(bars, vals):
            if v: ax.text(b.get_x() + b.get_width() / 2, v + 0.35, "%.1f×" % v,
                          ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Brecha de latencia Jetson-GPU↔RPi-CPU (×)")
    ax.legend(title="Técnica", frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(gaps.values()) * 1.18)
    fig.tight_layout()
    png = os.path.join(RESULTS, "oe3_brecha_despliegue.png")
    fig.savefig(png, dpi=200); fig.savefig(os.path.join(RESULTS, "oe3_brecha_despliegue.pdf"))
    return png


def main():
    lat, acc, ene, camp = load()
    S = {k: summarize(v) for k, v in lat.items()}
    out = []
    def w(s=""): out.append(s); print(s)

    w("# Análisis OE3 — GPU vs CPU a través de las técnicas (Jetson y RPi 5)\n")
    w("> **Cadena de cálculo (una sola, para todo el reporte).** Cada corrida produce 2000")
    w("> latencias crudas. (1) Por corrida se calcula su media geométrica. (2) La tendencia")
    w("> central de una condición es la media geométrica de esas R medias por corrida —")
    w("> equivalentemente, exp(media de los log). (3) Toda RAZÓN entre dos condiciones es el")
    w("> cociente de esas tendencias centrales, y su IC95 sale de la t de Student sobre la")
    w("> diferencia de log-medias, con la CORRIDA como unidad de réplica (nunca la inferencia")
    w("> individual). (4) Los percentiles p50/p95/p99 describen la DISTRIBUCIÓN y se calculan")
    w("> sobre las muestras crudas combinadas; NO son la base de ninguna razón. Un cociente de")
    w("> p50 no reproduce la brecha y no debe usarse para reconstruirla.")
    w("> Directriz de origen: DECISIONS D11 (log para tendencia central; cola con p50/p95/p99;")
    w("> conclusiones por tamaño de efecto e IC, no por p-valores).")
    w("> **Selección de corridas:** una sola campaña por condición (D21). rpi-cpu: campaña")
    w("> oficial fría y auditada de entorno único (kernel %s), con R alto para" % RPI_RELEASE)
    w("> absorber la multimodalidad entre procesos (BITACORA 14/19 jul 2026).")
    w("> Generado por `scripts/analyze_oe3.py`.\n")

    w("## 1. Resumen por condición\n")
    w("> La media geométrica es la tendencia central; p50/p95/p99 describen la distribución.\n")
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
                    model, tech, DEVLAB[dev], s["R"], s["gm_runs"], s["p50"], s["p95"], s["p99"],
                    "%.3f" % t1 if t1 else "—", "%.1f" % ej if ej else "—"))

    w("\n### 1b. Campaña usada y dispersión entre corridas (auditoría de selección)\n")
    w("> CV = desviación estándar relativa de las medias geométricas POR CORRIDA. Es la")
    w("> dispersión que alimenta los IC; valores <1 % indican una campaña homogénea.\n")
    w("| Modelo | Técnica | Disp. | R | CV entre corridas | Campaña |")
    w("|---|---|---|--:|--:|---|")
    for model in ["ResNet-50", "MobileNetV2"]:
        for tech in TECH:
            for dev in DEV:
                s = S.get((model, tech, dev))
                if not s: continue
                w("| %s | %s | %s | %d | %s | %s |" % (
                    model, tech, DEVLAB[dev], s["R"],
                    "%.2f %%" % s["cv_runs"] if s["R"] > 1 else "—",
                    camp.get((model, tech, dev), "—")))

    w("\n## 2. Brecha GPU↔CPU por técnica (tamaño de efecto e IC95)\n")
    w("> Razón = latencia(CPU) / latencia(GPU). Cuánto más rápida es la GPU.")
    w("> Las dos columnas de medias geométricas son las de la sección 1: la brecha es su")
    w("> cociente exacto, de modo que cualquiera puede reconstruirla desde esta tabla.\n")
    w("| Modelo | Técnica | gm CPU (ms) | gm GPU (ms) | Brecha GPU↔CPU | IC95 |")
    w("|---|---|--:|--:|--:|--:|")
    for model in ["ResNet-50", "MobileNetV2"]:
        for tech in ["V0", "INT8", "Poda"]:
            g = S.get((model, tech, "jetson-gpu")); c = S.get((model, tech, "jetson-cpu"))
            if not (g and c): continue
            r, lo, hi = ratio_ci(c["run_log"], g["run_log"])
            w("| %s | %s | %.3f | %.3f | %.2f× | [%.2f, %.2f] |" % (
                model, tech, c["gm_runs"], g["gm_runs"], r, lo, hi))

    w("\n### 2b. Brecha de despliegue Jetson-GPU ↔ RPi-CPU (tamaño de efecto e IC95)\n")
    w("> Razón = latencia(RPi-CPU) / latencia(Jetson-GPU): acelerador embebido frente a la CPU")
    w("> de placa. Referencia de despliegue entre plataformas (3ª columna de OE3).\n")
    w("| Modelo | Técnica | Brecha Jetson-GPU↔RPi-CPU | IC95 |")
    w("|---|---|--:|--:|")
    for model in ["ResNet-50", "MobileNetV2"]:
        for tech in ["V0", "INT8", "Poda"]:
            g = S.get((model, tech, "jetson-gpu")); c = S.get((model, tech, "rpi-cpu"))
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

    w("\n## 5. ART factorial (chequeo de robustez; la inferencia principal es efecto + IC)\n")
    w("> Aligned Rank Transform sobre el log de la media geométrica por corrida; factores")
    w("> dispositivo × técnica (V0/INT8/Poda). La interacción dispositivo×técnica ES el")
    w("> hallazgo de OE3: la técnica mueve la brecha. p reportado como confirmación no")
    w("> paramétrica, no como criterio de decisión (D11). Verificable en R/ARTool con el CSV.\n")
    w("| Modelo | Diseño | Efecto | F | gl | p | η²p |")
    w("|---|---|---|--:|--:|--:|--:|")
    for model in ["ResNet-50", "MobileNetV2"]:
        for des, devs in [("Jetson: GPU vs CPU", ["jetson-gpu", "jetson-cpu"]),
                          ("3 dispositivos", DEV)]:
            rows = []
            for dev in devs:
                for tech in ["V0", "INT8", "Poda"]:
                    s = S.get((model, tech, dev))
                    if s: rows += [(DEVLAB[dev], tech, float(lg)) for lg in s["run_log"]]
            if len({r[0] for r in rows}) < 2: continue
            for name, df1, df2, F, p, eta in art_anova(rows):
                w("| %s | %s | %s | %.1f | %d, %d | %s | %.2f |" % (
                    model, des, name, F, df1, df2,
                    "<0.001" if p < 0.001 else "%.3f" % p, eta))

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
        png2 = make_figure_despliegue(S)
        if png2:
            w("Figura de la brecha de despliegue: `results/%s` (+ .pdf)." % os.path.basename(png2))
    except Exception as e:
        w("[figura omitida: %s]" % e)

    with open(os.path.join(RESULTS, "OE3_ANALISIS.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n[escrito] results/OE3_ANALISIS.md")


if __name__ == "__main__":
    main()
