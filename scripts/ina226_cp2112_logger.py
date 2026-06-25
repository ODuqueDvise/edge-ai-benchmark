#!/usr/bin/env python3
"""Logger de potencia INA226 a traves de un puente USB-I2C CP2112 (multiplataforma).

Corre en Mac y Windows por igual (CP2112 es clase HID; se accede con hidapi).
Mismo formato de CSV que ina226_logger.py, asi que energy_from_window.py no cambia.

El protocolo HID del CP2112 esta basado en una implementacion de referencia
(artizirk/cp2112, a su vez basada en pymlab) y en la nota AN495/AN496 de Silicon
Labs. NO esta probado contra hardware en este entorno: por eso trae un AUTOTEST
que lee el ID de fabricante/dispositivo del INA226 antes de registrar. Si el
autotest no coincide, NO confies en los datos: revisa cableado, direccion I2C y
el valor del shunt.

Requisitos (en el host de registro, NO el equipo bajo prueba):
  pip install hidapi
  # macOS: si pide permisos de HID, autorizar. Windows: funciona sin driver.

Uso:
  python scripts/ina226_cp2112_logger.py --rshunt 0.002 --addr 0x40 --interval 0.05 --out power_log.csv
  python scripts/ina226_cp2112_logger.py --selftest            # solo verifica el enlace
  # La direccion I2C se autodetecta en 0x40-0x4F; --addr solo fija la preferencia.
"""
import argparse, time, csv

CP2112_VID = 0x10C4
CP2112_PID = 0xEA90

# Registros INA226
REG_CONFIG = 0x00
REG_SHUNT = 0x01
REG_BUS = 0x02
REG_MFR_ID = 0xFE     # esperado 0x5449 ('TI')
REG_DIE_ID = 0xFF     # esperado 0x2260
CONFIG_AVG16 = 0x4727  # AVG=16, conversion 1.1ms, shunt+bus continuo
SHUNT_LSB = 2.5e-6     # V/bit
BUS_LSB = 1.25e-3      # V/bit


class CP2112:
    """Transporte minimo CP2112 sobre hidapi (suficiente para leer/escribir INA226)."""

    def __init__(self, serial=None):
        import hid
        self.h = hid.device()
        self.h.open(CP2112_VID, CP2112_PID, serial)
        # SMBus config (AN496): reloj 100 kHz, dir. propia 0x02, timeouts por defecto.
        self.h.send_feature_report([0x06, 0x00, 0x01, 0x86, 0xA0, 0x02,
                                    0x00, 0x00, 0xFF, 0x00, 0xFF, 0x01, 0x00, 0x0F])

    def write_word(self, addr, reg, value):
        # Data Write Request: [0x14, addr<<1, len=3, reg, val_hi, val_lo]  (big-endian)
        self.h.write([0x14, addr << 1, 0x03, reg, (value >> 8) & 0xFF, value & 0xFF])

    def read_word(self, addr, reg):
        # Data Write Read Request (escribe el puntero de registro, lee 2 bytes) + Force Send.
        req = [0x11, addr << 1, 0x00, 0x02, 0x01, reg]
        self.h.write(req)
        self.h.write([0x12, 0x00, 0x02])          # Data Read Force Send
        for _ in range(12):
            resp = self.h.read(10, timeout_ms=200)
            if resp and resp[0] == 0x13 and resp[2] == 2:
                # INA226 entrega MSB primero: byte0=resp[3]=MSB, byte1=resp[4]=LSB.
                return (resp[3] << 8) | resp[4]
            self.h.write(req)
            self.h.write([0x12, 0x00, 0x02])
        raise IOError("CP2112: sin respuesta del INA226 (reg 0x%02X). Revisa cableado/direccion." % reg)

    def close(self):
        try:
            self.h.close()
        except Exception:
            pass


def _s16(x):
    return x - 65536 if x >= 32768 else x


def selftest(dev, addr):
    mfr = dev.read_word(addr, REG_MFR_ID)
    die = dev.read_word(addr, REG_DIE_ID)
    ok = (mfr == 0x5449) and ((die >> 4) == 0x226)
    print("Autotest INA226: MFR=0x%04X (esp 0x5449), DIE=0x%04X (esp 0x2260) -> %s"
          % (mfr, die, "OK" if ok else "FALLA"))
    return ok


def _is_ina226(dev, addr):
    """True solo si en addr responde un INA226 valido: verifica AMBOS registros de
    identidad (fabricante 0x5449 Y dispositivo 0x226x). Un ACK suelto que solo eche un
    valor no pasa los dos, asi que esto descarta falsos positivos de direcciones espurias."""
    try:
        return (dev.read_word(addr, REG_MFR_ID) == 0x5449
                and (dev.read_word(addr, REG_DIE_ID) >> 4) == 0x226)
    except Exception:
        return False


def find_ina226(dev, preferred):
    """Direccion de un INA226 VALIDADA por sus dos registros de identidad (no solo por un
    ACK). Escanea 0x40-0x4F. Si varias direcciones pasan la validacion (sintoma de un strap
    A0/A1 marginal), lo avisa y usa la preferida si esta entre ellas, o la primera. Hace el
    medidor plug-and-play sin --addr. Devuelve la direccion o None si ninguna es un INA226."""
    hits = [addr for addr in range(0x40, 0x50) if _is_ina226(dev, addr)]
    if not hits:
        return None
    chosen = preferred if preferred in hits else hits[0]
    if len(hits) > 1:
        print("Aviso: el INA226 responde en varias direcciones (%s): strap A0/A1 marginal, "
              "conviene fijarlo en hardware. Usando 0x%02X."
              % (", ".join("0x%02X" % h for h in hits), chosen))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rshunt", type=float, default=None, help="resistencia del shunt en ohmios (p.ej. 0.002)")
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=0x40, help="direccion I2C preferida; si no responde, se autodetecta en 0x40-0x4F (def 0x40)")
    ap.add_argument("--serial", default=None, help="numero de serie del CP2112 (si hay varios)")
    ap.add_argument("--interval", type=float, default=0.05, help="periodo de muestreo en s (def 0.05 = 20 Hz)")
    ap.add_argument("--out", default="power_log.csv")
    ap.add_argument("--selftest", action="store_true", help="solo verifica el enlace y sale")
    a = ap.parse_args()

    dev = CP2112(a.serial)
    addr = find_ina226(dev, a.addr)
    if addr is None:
        print("Detente: no se detecto ningun INA226 (MFR=0x5449) en 0x40-0x4F. Revisa enlace, alimentacion (VS) y cableado SDA/SCL/GND.")
        dev.close()
        return
    if addr != a.addr:
        print("Aviso: INA226 autodetectado en 0x%02X (no en la 0x%02X por defecto); usando 0x%02X." % (addr, a.addr, addr))
    a.addr = addr
    if not selftest(dev, a.addr):
        print("Detente: el autotest fallo. No registres datos hasta resolverlo.")
        dev.close()
        return
    if a.selftest:
        dev.close()
        return
    if a.rshunt is None:
        print("Falta --rshunt (valor real del shunt de tu modulo).")
        dev.close()
        return

    dev.write_word(a.addr, REG_CONFIG, CONFIG_AVG16)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch_s", "bus_V", "current_A", "power_W"])
        print("Registrando en", a.out, "- Ctrl-C para detener")
        try:
            while True:
                t = time.time()
                vsh = _s16(dev.read_word(a.addr, REG_SHUNT)) * SHUNT_LSB
                vbus = dev.read_word(a.addr, REG_BUS) * BUS_LSB
                i = vsh / a.rshunt
                p = vbus * i
                w.writerow(["%.4f" % t, "%.4f" % vbus, "%.5f" % i, "%.4f" % p])
                f.flush()
                rem = a.interval - (time.time() - t)
                if rem > 0:
                    time.sleep(rem)
        except KeyboardInterrupt:
            print("\nDetenido.")
        finally:
            dev.close()


if __name__ == "__main__":
    main()
