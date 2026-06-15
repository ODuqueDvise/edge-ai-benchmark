#!/usr/bin/env python3
"""Registra potencia con un INA226 por I2C, con marca de tiempo (epoch).

Calcula la corriente a partir de la caida en el shunt (no usa el registro de
calibracion, mas robusto):  I = Vshunt / Rshunt ;  P = Vbus * I.

Corre en un HOST DE REGISTRO (idealmente una Pi auxiliar, no el equipo bajo
prueba, para no sumar carga al equipo medido). Habilita I2C en ese host.

  pip install smbus2
  python scripts/ina226_logger.py --rshunt 0.002 --bus 1 --addr 0x40 --interval 0.05 --out power_log.csv

Detener con Ctrl-C. Ajusta --rshunt al valor real del shunt de TU modulo INA226.
"""
import argparse, time, csv

REG_CONFIG = 0x00
REG_SHUNT = 0x01
REG_BUS = 0x02
CONFIG_AVG16 = 0x4727      # AVG=16, conversion 1.1ms, shunt+bus continuo
SHUNT_LSB = 2.5e-6         # V por bit (registro de shunt voltage)
BUS_LSB = 1.25e-3          # V por bit (registro de bus voltage)


def _s16(x):
    return x - 65536 if x >= 32768 else x


def read_reg(bus, addr, reg):
    d = bus.read_i2c_block_data(addr, reg, 2)   # big-endian
    return (d[0] << 8) | d[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rshunt", type=float, required=True, help="resistencia del shunt en ohmios (p.ej. 0.002)")
    ap.add_argument("--bus", type=int, default=1, help="numero de bus I2C del host")
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=0x40, help="direccion I2C (def 0x40)")
    ap.add_argument("--interval", type=float, default=0.05, help="periodo de muestreo en s (def 0.05 = 20 Hz)")
    ap.add_argument("--out", default="power_log.csv")
    a = ap.parse_args()

    from smbus2 import SMBus
    bus = SMBus(a.bus)
    bus.write_i2c_block_data(a.addr, REG_CONFIG, [CONFIG_AVG16 >> 8, CONFIG_AVG16 & 0xFF])

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch_s", "bus_V", "current_A", "power_W"])
        print("Registrando en", a.out, "- Ctrl-C para detener")
        try:
            while True:
                t = time.time()
                vsh = _s16(read_reg(bus, a.addr, REG_SHUNT)) * SHUNT_LSB
                vbus = read_reg(bus, a.addr, REG_BUS) * BUS_LSB
                i = vsh / a.rshunt
                p = vbus * i
                w.writerow(["%.4f" % t, "%.4f" % vbus, "%.5f" % i, "%.4f" % p])
                f.flush()
                rem = a.interval - (time.time() - t)
                if rem > 0:
                    time.sleep(rem)
        except KeyboardInterrupt:
            print("\nDetenido.")


if __name__ == "__main__":
    main()
