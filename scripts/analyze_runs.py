#!/usr/bin/env python3
"""Resume corridas de latencia y mide la variabilidad ENTRE corridas para fijar R.

AGRUPA por (modelo, nº de iteraciones): el CV solo tiene sentido entre corridas
comparables. Asi no mezcla modelos distintos (p.ej. el smoke model) ni
configuraciones distintas (1000 vs 2000 iters).

Uso:
  python scripts/analyze_runs.py "results/jetson-gpu_*tensorrt*.json"
"""
import argparse, glob, json
from collections import defaultdict
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="patrones, p.ej. 'results/jetson-gpu_*.json'")
    a = ap.parse_args()

    files = []
    for g in a.globs:
        files += sorted(glob.glob(g))

    groups = defaultdict(list)
    for f in files:
        try:
            d = json.load(open(f))
            s = d.get("latency_summary")
            if not s:
                continue
            md = d.get("metadata", {})
            sha = (md.get("model", {}).get("sha256") or "????????")[:8]
            iters = md.get("iters", s.get("n"))
            dev = md.get("device_tag", "?")
            groups[(dev, sha, iters)].append(
                (f.split("/")[-1], s["p50_ms"], s["p95_ms"], s["p99_ms"], s["mean_ms"], s["n"]))
        except Exception as e:
            print("omito", f, e)

    if not groups:
        print("No hay corridas de latencia con esos patrones.")
        return

    for (dev, sha, iters), rows in groups.items():
        print("\n=== %s | modelo %s | iters %s | %d corrida(s) ===" % (dev, sha, iters, len(rows)))
        print("%-50s %8s %8s %8s %8s" % ("archivo", "p50", "p95", "p99", "media"))
        for name, p50, p95, p99, m, n in rows:
            print("%-50s %8.3f %8.3f %8.3f %8.3f" % (name, p50, p95, p99, m))
        p50s = np.array([r[1] for r in rows], dtype=float)
        if len(p50s) > 1:
            cv = 100 * p50s.std(ddof=1) / p50s.mean()
            print("  -> p50 entre corridas: media %.3f ms, desv %.3f ms, CV %.2f%%" %
                  (p50s.mean(), p50s.std(ddof=1), cv))
            print("     CV ~<2-3%% -> R=3-5 basta. CV alto -> sube R o revisa termico.")


if __name__ == "__main__":
    main()
