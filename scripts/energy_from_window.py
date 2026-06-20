#!/usr/bin/env python3
"""Calcula la energia sobre la ventana de una corrida y la GUARDA en results/energy_*.json.

Toma el CSV del logger INA226 y un JSON de resultados (con metadata.window),
integra la potencia en esa ventana, reporta energia total y neta, y escribe
results/energy_<device>_<stamp>.json para que build_results_log.py la incluya solo.

Relojes del host de registro y del equipo deben estar sincronizados (NTP).

  python scripts/energy_from_window.py --log power.csv --result results/jetson-gpu_....json --idle-watts 7.8
"""
import argparse, csv, json, os, time


def load_log(path):
    ts, pw = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["epoch_s"]))
            pw.append(float(row["power_W"]))
    return ts, pw


def integrate(ts, pw, t0, t1):
    e, seg = 0.0, 0
    for i in range(1, len(ts)):
        a, b = ts[i - 1], ts[i]
        lo, hi = max(a, t0), min(b, t1)
        if hi <= lo:
            continue
        span = b - a
        pa = pw[i - 1] + (pw[i] - pw[i - 1]) * ((lo - a) / span if span else 0)
        pb = pw[i - 1] + (pw[i] - pw[i - 1]) * ((hi - a) / span if span else 0)
        e += 0.5 * (pa + pb) * (hi - lo)
        seg += 1
    return e, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--idle-watts", type=float, default=None, help="potencia en reposo a restar (W)")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    R = json.load(open(a.result))
    md = R.get("metadata", {})
    win = md["window"]
    t0, t1 = win["start_epoch_s"], win["end_epoch_s"]
    iters = md.get("iters") or R.get("latency_summary", {}).get("n")
    dev = md.get("device_tag", "?")
    sha = md.get("model", {}).get("sha256")
    mname = md.get("model", {}).get("name")
    var = md.get("variant")

    ts, pw = load_log(a.log)
    e, seg = integrate(ts, pw, t0, t1)
    if seg == 0:
        print("ERROR: el log no cubre la ventana. Revisa relojes NTP y que el logger corriera durante la corrida.")
        return
    dur = t1 - t0
    avg = e / dur
    per_total = (e / iters * 1000) if iters else None

    out = {"metadata": {"device_tag": dev, "model": {"name": mname, "sha256": sha}, "variant": var,
                        "source_result": os.path.basename(a.result),
                        "idle_watts": a.idle_watts, "iters": iters,
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "energy": {"window_s": dur, "total_j": e, "avg_power_w": avg,
                      "per_inf_total_mj": per_total}}

    print("Ventana: %.2f s | energia total: %.3f J | potencia media: %.3f W" % (dur, e, avg))
    print("Energia por inferencia (total): %.3f mJ  (n=%s)" % (per_total, iters))
    if a.idle_watts is not None:
        e_net = e - a.idle_watts * dur
        per_net = (e_net / iters * 1000) if iters else None
        out["energy"]["net_j"] = e_net
        out["energy"]["per_inf_net_mj"] = per_net
        print("Energia neta (resta reposo %.2f W): %.3f J | por inferencia: %.3f mJ" % (a.idle_watts, e_net, per_net))

    os.makedirs(a.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    model_tag = mname or ((sha or "model")[:8])
    outp = os.path.join(a.out_dir, "energy_%s_%s_%s.json" % (dev, model_tag, stamp))
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print("Escrito:", outp)


if __name__ == "__main__":
    main()
