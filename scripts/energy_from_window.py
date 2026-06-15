#!/usr/bin/env python3
"""Calcula la energia sobre la ventana de una corrida del arnes.

Toma el CSV del logger INA226 y un JSON de resultados (con
metadata.window.start_epoch_s/end_epoch_s), integra la potencia en esa ventana
(trapezoidal) y reporta energia total, energia por inferencia y potencia media.
Opcional: resta una linea base de reposo.

IMPORTANTE: los relojes del host de registro y del equipo bajo prueba deben
estar sincronizados (NTP), o las ventanas no coincidiran.

  python scripts/energy_from_window.py --log power_log.csv --result results/jetson-gpu_....json
  python scripts/energy_from_window.py --log power_log.csv --result ... --idle-watts 5.1
"""
import argparse, csv, json


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
    a = ap.parse_args()

    R = json.load(open(a.result))
    win = R["metadata"]["window"]
    t0, t1 = win["start_epoch_s"], win["end_epoch_s"]
    iters = R["metadata"].get("iters") or R["latency_summary"]["n"]

    ts, pw = load_log(a.log)
    e, seg = integrate(ts, pw, t0, t1)
    if seg == 0:
        print("ERROR: el log no cubre la ventana de la corrida.")
        print("Revisa: relojes sincronizados (NTP) y que el logger corriera durante la medicion.")
        return
    dur = t1 - t0
    print("Ventana: %.2f s | energia total: %.3f J | potencia media: %.3f W" % (dur, e, e / dur))
    print("Energia por inferencia (total): %.3f mJ  (n=%d)" % (e / iters * 1000, iters))
    if a.idle_watts is not None:
        e_net = e - a.idle_watts * dur
        print("Energia neta (resta reposo %.2f W): %.3f J | por inferencia: %.3f mJ"
              % (a.idle_watts, e_net, e_net / iters * 1000))


if __name__ == "__main__":
    main()
