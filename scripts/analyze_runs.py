#!/usr/bin/env python3
"""Resume varias corridas de latencia para decidir las constantes del protocolo (M, R).

Lee los JSON de latencia que le pases y reporta percentiles por corrida y, sobre
todo, la VARIABILIDAD ENTRE corridas (el coeficiente de variación del p50). Eso es
lo que te dice cuantas ejecuciones independientes (R) necesitas.

Uso:
  python scripts/analyze_runs.py "results/jetson-gpu_*tensorrt*.json"
  python scripts/analyze_runs.py "results/jetson-cpu_*cpu*.json"
"""
import argparse, glob, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="patrones, p.ej. 'results/jetson-gpu_*.json'")
    a = ap.parse_args()

    files = []
    for g in a.globs:
        files += sorted(glob.glob(g))
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
            s = d.get("latency_summary")
            if s:
                rows.append((f.split("/")[-1], s["p50_ms"], s["p95_ms"], s["p99_ms"], s["mean_ms"], s["n"]))
        except Exception as e:
            print("omito", f, e)
    if not rows:
        print("No hay corridas de latencia con esos patrones.")
        return

    print("%-50s %8s %8s %8s %8s %7s" % ("archivo", "p50", "p95", "p99", "media", "n"))
    for name, p50, p95, p99, m, n in rows:
        print("%-50s %8.3f %8.3f %8.3f %8.3f %7d" % (name, p50, p95, p99, m, n))

    p50s = np.array([r[1] for r in rows], dtype=float)
    if len(p50s) > 1:
        cv = 100 * p50s.std(ddof=1) / p50s.mean()
        print("\nEntre %d corridas -> p50: media %.3f ms, desv %.3f ms, CV %.2f%%"
              % (len(p50s), p50s.mean(), p50s.std(ddof=1), cv))
        print("Regla practica: CV bajo (~<2-3%) -> R=3-5 basta para reportar media +/- desv.")
        print("CV alto -> sube R, revisa termico/throttling, o fija mejor las frecuencias.")


if __name__ == "__main__":
    main()
