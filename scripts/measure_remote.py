#!/usr/bin/env python3
"""Orquestador de medición remota (corre en el Mac, UN comando por condición).

Unifica en un solo terminal: chequeos -> logger de energía local -> latencia por
SSH (R corridas) -> la Jetson commitea sus JSON -> el Mac hace pull -> guardia de
proveedor -> energía local -> commit del energy_*.json.

Filosofía: automatiza la plomería, NO el criterio. Aborta en rojo ante reloj
desfasado, checksum que no cuadra, autotest del medidor fallido, ventana de energía
sin cubrir o caída silenciosa de la GPU a CPU. Un final verde significa "medido
bien", no "corrió sin reventar".

Requisitos: SSH por llave al host (ssh-copy-id, sin contraseña), relojes
sincronizados por NTP, y el medidor INA226+CP2112 conectado al Mac (si se mide
energía). Los JSON de dispositivo viajan por git (la Jetson commitea, el Mac hace
pull): NO se usa rsync, así que no hay colisión de archivos sin trackear.

Ejemplos:
  # jetson-gpu con energía (shunt R100 = 0.1 ohm) y verificación de modelo
  python3 scripts/measure_remote.py --host orlando@orlando-desktop.local \
      --device-tag jetson-gpu --provider tensorrt --model models/resnet50_baseline.onnx \
      --shunt 0.1 --expect-sha 05e5bc14444e89b9b47b36c663bc40e061db8d20389d833dcde3c7da667290dc

  # ver el plan sin ejecutar nada
  python3 scripts/measure_remote.py --host ... --device-tag jetson-cpu --provider cpu \
      --model models/resnet50_baseline.onnx --dry-run
"""
from __future__ import annotations
import argparse, glob, json, os, signal, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def die(msg):
    print("\n[ABORTA] " + msg, file=sys.stderr)
    sys.exit(1)


def info(msg):
    print("[*] " + msg)


class Runner:
    def __init__(self, dry):
        self.dry = dry

    def local(self, cmd, capture=False, check=True):
        if self.dry:
            print("    (dry) local:", " ".join(cmd))
            return ""
        r = subprocess.run(cmd, cwd=REPO, text=True, capture_output=capture)
        if check and r.returncode != 0:
            die("falló: " + " ".join(cmd) + (("\n" + (r.stderr or "")) if capture else ""))
        return (r.stdout or "") if capture else ""

    def ssh(self, host, remote_cmd, capture=False, check=True):
        if self.dry:
            print("    (dry) ssh:", remote_cmd)
            return ""
        r = subprocess.run(["ssh", *SSH_OPTS, host, remote_cmd], text=True, capture_output=capture)
        if check and r.returncode != 0:
            die("ssh falló (" + remote_cmd[:60] + "...)" + (("\n" + (r.stderr or "")) if capture else ""))
        return (r.stdout or "") if capture else ""


def parse_args():
    p = argparse.ArgumentParser(description="Orquestador de medición remota (una condición).")
    p.add_argument("--host", required=True, help="usuario@host (SSH por llave, sin contraseña)")
    p.add_argument("--remote-repo", default="edge-ai-benchmark", help="ruta del repo en el host (def: edge-ai-benchmark)")
    p.add_argument("--device-tag", required=True, help="jetson-gpu | jetson-cpu | rpi-cpu")
    p.add_argument("--provider", required=True, help="tensorrt | cuda | cpu")
    p.add_argument("--model", required=True, help="ruta del .onnx (igual en ambas máquinas), p.ej. models/resnet50_baseline.onnx")
    p.add_argument("--expect-sha", default=None, help="SHA-256 esperado del modelo remoto (verifica antes de medir)")
    p.add_argument("--shunt", type=float, default=None, help="ohmios del shunt (p.ej. 0.1). Si se omite, NO se mide energía.")
    p.add_argument("--idle-watts", type=float, default=7.8, help="potencia en reposo a restar para energía neta")
    p.add_argument("--reps", type=int, default=5, help="R: corridas de latencia (def 5)")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--input-shape", default="1,3,224,224")
    p.add_argument("--power-mode", default="MAXN")
    p.add_argument("--addr", default="0x40", help="dirección I2C del INA226")
    p.add_argument("--interval", type=float, default=0.05, help="periodo de muestreo del logger (s)")
    p.add_argument("--accuracy", action="store_true", help="además, corre precisión (set completo; lenta, sin logger)")
    p.add_argument("--dataset", default="datasets/imagenetv2-matched-frequency-format-val")
    p.add_argument("--max-clock-skew", type=float, default=0.5, help="desfase de reloj Mac↔host tolerado (s)")
    p.add_argument("--activate", default=". .venv/bin/activate", help="comando para activar el venv en el host")
    p.add_argument("--no-commit", action="store_true", help="no commitea (para pruebas)")
    p.add_argument("--dry-run", action="store_true", help="imprime el plan sin ejecutar nada")
    return p.parse_args()


def main():
    a = parse_args()
    r = Runner(a.dry_run)
    stem = os.path.splitext(os.path.basename(a.model))[0]
    gpu_expected = a.device_tag.endswith("-gpu") or a.provider in ("tensorrt", "cuda")
    measure_energy = a.shunt is not None
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    csv = "results/power_%s_%s_%s.csv" % (a.device_tag, stem, stamp)

    print("=" * 70)
    print("Condición: %s / %s  | modelo: %s  | energía: %s" % (
        a.device_tag, a.provider, stem, "sí (shunt %s Ω)" % a.shunt if measure_energy else "NO"))
    print("=" * 70)

    # ---------- 1. Preflight (aborta en rojo) ----------
    info("Preflight: alcanzabilidad SSH (por llave)…")
    if not a.dry_run:
        probe = subprocess.run(["ssh", *SSH_OPTS, a.host, "echo ok"], text=True, capture_output=True)
        if probe.returncode != 0 or "ok" not in probe.stdout:
            die("no se pudo entrar por SSH sin contraseña a %s.\n"
                "Configura la llave una vez:  ssh-copy-id %s\n%s"
                % (a.host, a.host, probe.stderr or ""))

    info("Preflight: desfase de reloj Mac↔host (NTP)…")
    if not a.dry_run:
        t0 = time.time()
        out = r.ssh(a.host, "date +%s.%N", capture=True)
        t1 = time.time()
        try:
            remote = float(out.strip().split()[0])
        except Exception:
            die("no pude leer la hora del host: %r" % out)
        skew = (t0 + t1) / 2 - remote
        if abs(skew) > a.max_clock_skew:
            die("reloj Mac↔host desfasado %.2f s (RTT %.2f s). La ventana de energía no alinea.\n"
                "Sincroniza NTP en ambos (Jetson: sudo timedatectl set-ntp true) y reintenta." % (skew, t1 - t0))
        info("  desfase %.3f s (RTT %.3f s) — OK" % (skew, t1 - t0))

    if a.expect_sha:
        info("Preflight: checksum del modelo remoto…")
        if not a.dry_run:
            out = r.ssh(a.host, "cd %s && sha256sum %s" % (a.remote_repo, a.model), capture=True)
            remote_sha = (out.strip().split() or [""])[0]
            if remote_sha != a.expect_sha:
                die("el modelo remoto NO coincide.\n  esperado: %s\n  remoto:   %s\n"
                    "No midas: copia el archivo correcto a la Jetson." % (a.expect_sha, remote_sha))
            info("  checksum OK")

    if measure_energy:
        info("Preflight: autotest del INA226 (medidor local)…")
        if not a.dry_run:
            out = r.local([sys.executable, "scripts/ina226_cp2112_logger.py", "--selftest", "--addr", a.addr],
                          capture=True, check=False)
            print(out.strip())
            if "-> OK" not in out:
                die("el autotest del INA226 no dio OK. Revisa cableado, dirección I2C y shunt.")

    # ---------- 2-4. Logger local + latencia remota (R) ----------
    proc = None
    try:
        if measure_energy:
            info("Arrancando logger de energía -> %s" % csv)
            if not a.dry_run:
                proc = subprocess.Popen(
                    [sys.executable, "scripts/ina226_cp2112_logger.py", "--rshunt", str(a.shunt),
                     "--addr", a.addr, "--interval", str(a.interval), "--out", csv], cwd=REPO)
                time.sleep(2.0)
                if proc.poll() is not None:
                    die("el logger murió al arrancar (autotest/HID). Revisa el medidor.")
            else:
                print("    (dry) logger:", "ina226_cp2112_logger.py --rshunt %s --out %s" % (a.shunt, csv))

        info("Latencia remota: %d corrida(s) en %s…" % (a.reps, a.device_tag))
        lat_cmd = ("cd %s && %s && for i in $(seq 1 %d); do "
                   "python -m bench.run_benchmark --model %s --backend ort --provider %s "
                   "--device-tag %s --input-shape %s --warmup %d --iters %d --power-mode %s || exit 1; done"
                   % (a.remote_repo, a.activate, a.reps, a.model, a.provider, a.device_tag,
                      a.input_shape, a.warmup, a.iters, a.power_mode))
        r.ssh(a.host, lat_cmd)
    finally:
        if proc is not None:
            info("Deteniendo logger…")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.terminate()

    # ---------- 5b. Precisión (opcional, sin logger) ----------
    if a.accuracy:
        info("Precisión remota (set completo; puede tardar)…")
        acc_cmd = ("cd %s && %s && python -m bench.run_accuracy --model %s --backend ort "
                   "--provider %s --device-tag %s --dataset %s"
                   % (a.remote_repo, a.activate, a.model, a.provider, a.device_tag, a.dataset))
        r.ssh(a.host, acc_cmd)

    # ---------- 6-7. La Jetson commitea sus JSON; el Mac hace pull ----------
    info("La Jetson sube sus resultados (git)…")
    r.ssh(a.host, "cd %s && bash scripts/sync_results.sh 'auto %s %s'" % (a.remote_repo, a.device_tag, stem))
    info("El Mac trae los resultados (git pull)…")
    r.local(["git", "pull", "--rebase", "--autostash"])

    if a.dry_run:
        print("\n(dry) Plan completo. No se ejecutó nada.")
        return

    # ---------- 8. Guardia de proveedor ----------
    pat = "results/%s_%s_ort_%s_*.json" % (a.device_tag, stem, a.provider)
    cands = sorted(glob.glob(os.path.join(REPO, pat)))
    if not cands:
        die("no encuentro el JSON de latencia tras el pull (patrón %s)." % pat)
    latest = cands[-1]
    md = json.load(open(latest)).get("metadata", {})
    aps = md.get("active_providers") or []
    info("Proveedores activos: %s" % (aps or "?"))
    if gpu_expected and not any(("Tensorrt" in p or "CUDA" in p) for p in aps):
        die("se esperaba GPU pero los proveedores activos no incluyen TensorRT/CUDA: %s\n"
            "Cayó a CPU en silencio. No se registra como GPU." % aps)

    # ---------- 9. Energía local sobre la ventana ----------
    if measure_energy:
        info("Calculando energía sobre la ventana…")
        out = r.local([sys.executable, "scripts/energy_from_window.py", "--log", csv,
                       "--result", os.path.relpath(latest, REPO), "--idle-watts", str(a.idle_watts)],
                      capture=True)
        print(out.strip())
        if "ERROR" in out or "no cubre" in out:
            die("la ventana de energía no quedó cubierta por el CSV (relojes/logger). El número NO es válido.")

    # ---------- 10. Commit (Mac sube energy_*.json + CSV) ----------
    if measure_energy and not a.no_commit:
        info("Subiendo energía (git)…")
        r.local(["bash", "scripts/sync_results.sh", "auto energía %s %s" % (a.device_tag, stem)])

    # ---------- Resumen ----------
    s = json.load(open(latest)).get("latency_summary", {})
    print("\n" + "=" * 70)
    print("OK %s / %s  | p50 %.3f ms  p95 %.3f ms  p99 %.3f ms  (R=%d JSON: %s)"
          % (a.device_tag, a.provider, s.get("p50_ms", 0), s.get("p95_ms", 0),
             s.get("p99_ms", 0), len(cands), os.path.basename(latest)))
    print("=" * 70)


if __name__ == "__main__":
    main()
