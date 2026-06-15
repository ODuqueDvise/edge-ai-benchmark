# Medicion de energia con INA226 (externo, mismo metodo en Jetson y RPi)

La energia es la metrica donde la comparacion se rompe con mas facilidad. Reglas:
medir en la **entrada DC** de cada equipo, con el **mismo modelo de sensor**, y
registrar con **marca de tiempo** para integrar sobre la ventana del arnes.

## Esquema de conexion (sensado en lado alto)

El INA226 mide la caida en un shunt puesto EN SERIE en el cable POSITIVO de
alimentacion del equipo bajo prueba. La masa (GND) es comun entre la fuente, el
equipo y el host de registro; solo las lineas I2C van al host.

```
  Fuente +  ──►  IN+ [ shunt ] IN-  ──►  + entrada del equipo (Jetson barril ~19V / RPi USB-C 5V)
  Fuente -  ─────────────────────────►  - entrada del equipo   (GND comun)

  INA226:  VCC ── 3.3V del host de registro
           GND ── GND comun
           SDA ── SDA del host
           SCL ── SCL del host
```

- **Host de registro:** una Pi auxiliar (o microcontrolador), NO el equipo medido,
  para no sumarle carga. El mismo host registra la Jetson y la RPi.
- **Lado alto:** cortar el positivo e intercalar el modulo; el equipo se alimenta
  de su fuente normal a traves del shunt.
- **Verifica en TU modulo:** el valor real del shunt (suele ser 0.002 o 0.1 ohm),
  la direccion I2C (0x40 por defecto; cambia con A0/A1) y que no excedes la
  corriente maxima del shunt/pistas.
- El INA226 admite hasta 36V de bus, asi que el mismo sensor sirve para los ~19V
  de la Jetson y los 5V de la RPi.

## Sincronizacion de reloj (obligatoria)

El host de registro y cada equipo bajo prueba deben tener la hora sincronizada por
NTP (`timedatectl` debe mostrar "System clock synchronized: yes"). Si no, las
ventanas epoch del arnes y del log no coincidiran.

## Procedimiento por corrida

```bash
# 1. (host de registro) habilitar I2C e instalar dependencia
pip install smbus2

# 2. (host) iniciar el registro ANTES de la corrida (ajusta --rshunt a tu modulo)
python scripts/ina226_logger.py --rshunt 0.002 --interval 0.05 --out power_log.csv

# 3. (equipo bajo prueba) correr el benchmark normal del arnes
python -m bench.run_benchmark --model models/cnn_baseline.onnx --backend ort \
    --provider tensorrt --device-tag jetson-gpu --input-shape 1,3,224,224 --warmup 50 --iters 1000

# 4. (host) detener el logger (Ctrl-C) y calcular la energia de esa ventana
python scripts/energy_from_window.py --log power_log.csv \
    --result results/jetson-gpu_ort_tensorrt_XXXX.json
```

## Linea base de reposo

Mide la potencia en reposo de cada equipo (sin carga, mismo modo de potencia) y
pasala con `--idle-watts` para reportar la energia NETA atribuible a la inferencia,
ademas del total. Reporta ambas.

## Que reportar

Por cada condicion: energia total (J) y por inferencia (mJ) sobre la ventana,
potencia media (W), potencia de reposo (W), y temperatura inicio/fin. El medidor
externo es la fuente primaria; los sensores internos de la Jetson, solo referencia.
