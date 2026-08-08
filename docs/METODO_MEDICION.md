# Método de medición y análisis — especificación completa

Documento de referencia del arnés `edge-ai-benchmark`. Contiene el detalle que el
artículo de ARTIIS 2026 resume por límite de espacio: la cadena completa que va de una
inferencia individual a cada número publicado, en el nivel de detalle necesario para
reconstruirlo o reproducirlo.

Es también el borrador del capítulo de metodología de la tesis.

> **Cómo se cita desde el artículo.** La sección 3 del artículo describe el diseño y el
> criterio; este documento aporta los parámetros exactos. Ante cualquier discrepancia
> entre ambos, mandan los archivos de `results/`, que registran cada valor por corrida.
>
> **Regla transversal (D22, D23).** Se distingue entre campos MEDIDOS —leídos del sistema
> y guardados en los metadatos de cada corrida— y campos DECLARADOS —etiquetas escritas
> por el operador—. Ninguna afirmación de este documento se apoya en un campo declarativo.

---

## 1. Plataformas y entorno

**Jetson Orin Nano 8 GB** (Orin Nano Engineering Reference Developer Kit Super), dos
condiciones sobre el mismo SoC:

| Condición | Proveedor de ejecución | Cómputo |
|---|---|---|
| `jetson-gpu` | TensorRT | núcleos CUDA y Tensor |
| `jetson-cpu` | CPU EP de ONNX Runtime | 6× Cortex-A78AE |

Entorno congelado durante toda la campaña: JetPack 6.2, kernel `5.15.148-tegra`,
`aarch64`, ONNX Runtime 1.24.0, Python 3.10.12.

**Modo de potencia.** La placa define cuatro: `0 = 15W`, `1 = 25W`, `2 = MAXN_SUPER`,
`3 = 7W`. Todas las mediciones se hicieron en el **modo 0 (15 W)**, que es el de defecto;
en él el reloj de GPU está limitado a 612 MHz. El modo se lee del sistema con
`nvpmodel -q` en cada corrida y se guarda en el campo `nvpmodel` de los metadatos. El
campo `power_mode_declared`, que aparece en los archivos de junio con el valor `MAXN`, es
una etiqueta del operador y **no debe usarse**: ver D22.

**Frecuencias.** No se afirma que el escalado dinámico estuviera desactivado. `jetson_clocks`
figuraba en el procedimiento y el historial de la placa muestra que se ejecutó, pero no
persiste entre reinicios, los archivos de junio no registran gobernador ni frecuencia
—esos campos se añadieron al arnés en julio— y las entradas del historial no llevan marca
de tiempo. La evidencia de estabilidad que sí es auditable es la dispersión entre corridas
(sección 6.4). Ver D23.

**Plataforma de referencia.** Raspberry Pi 5 de 8 GB, solo CPU, solo latencia, sin
instrumentación energética. Campaña oficial restringida al kernel `7.0.0-1014-raspi`
(política de entorno único; el salto desde `1009` movió ResNet-50 sin podar un 8-10 %).

---

## 2. Definición de corrida y selección de campañas

**Una corrida es un proceso nuevo.** El arnés se invoca de cero: carga el modelo,
construye el motor de ejecución, descarta 100 iteraciones de calentamiento y cronometra
2000 con lote unitario y entrada `1,3,224,224`. La réplica es el proceso y no la
iteración porque la construcción del motor y la colocación de memoria se fijan al
arrancar y son idénticas para las 2000 iteraciones de una misma corrida.

**Entrada sintética.** Un tensor fijo, reutilizado en todas las iteraciones: se mide costo
de cómputo, no de entrada/salida.

**Constantes congeladas (D6):** warmup 100, iters 2000, R = 5, entrada `1,3,224,224`.
Fijadas a partir de una corrida piloto con CV del p50 de 0,56 % y no modificadas después.

**Orden de ejecución.** Las condiciones se ejecutaron **por bloques, no intercaladas**: las
cinco corridas de una condición son consecutivas —separadas por segundos o pocos minutos—
y al terminar se pasa a la siguiente. Es una limitación reconocida: en una placa de
refrigeración pasiva el estado térmico podría confundirse con la condición. Acotación
empírica: en toda la campaña de la Jetson las seis zonas térmicas del SoC se mantuvieron
entre 48 y 52 °C, con deriva menor a 1,5 °C por corrida.

> **Matiz de instrumentación.** En la campaña de junio el arnés capturaba
> `thermal_c_start` al cerrar la corrida, de modo que ambas lecturas son posteriores a la
> ejecución. El arreglo —captura antes de cargar el modelo, más `cpu_state_start/end` con
> gobernador, frecuencia por política de `cpufreq` y `vcgencmd get_throttled`— es de julio
> y no cubre estos archivos.

**Selección de campañas (D21).** Cada condición se estima con **una sola sesión de
medición**. `pick_campaign()` en `scripts/analyze_oe3.py` agrupa las corridas por cercanía
temporal (hueco superior a 30 minutos ⇒ sesión distinta), toma el grupo más numeroso
—desempate por el más reciente— y lo recorta a cinco. El criterio opera sobre la
estructura temporal, **no sobre los valores**.

Motivo: la regla anterior, "las cinco más recientes", mezclaba sesiones. En MobileNetV2
sin optimizar tomaba cuatro corridas del 15 de junio más una recomprobación aislada del
20, descartando una del 15. Esa corrida (media geométrica 2,908 ms frente a 2,460-2,482
de la campaña; p99 4,718; máximo 7,506) elevaba el CV entre corridas de 0,35 % a 7,66 % y
era la causa completa de un intervalo anómalo. La recomprobación permanece en `results/` y
queda fuera por el criterio, no por su valor.

La RPi es la excepción: usa **todas** las corridas oficiales del kernel único, porque su
multimodalidad entre procesos se absorbe con R alto en lugar de recortarse.

---

## 3. Modelos y artefactos

Dos redes elegidas por contraste estructural: **MobileNetV2** (compacta, limitada por
memoria) y **ResNet-50** (densa, limitada por cómputo, con redundancia). Pesos
preentrenados de ImageNet-1k, exportados a **ONNX opset 18** en FP32.

Ocho artefactos, cada uno versionado con su SHA-256, que queda registrado en cada archivo
de resultados:

| Modelo | Base | INT8 | Podado | Podado + destilación |
|---|---|---|---|---|
| MobileNetV2 | `609015cb` | `c1eac3d6` | `7be5303c` | `8a6cdf8c` |
| ResNet-50 | `05e5bc14` | `2161a04a` | `940aefb8` | `efffe63b` |

Se distribuyen por Git LFS: toda medida es trazable al archivo exacto que se ejecutó, sin
reexportaciones locales.

---

## 4. Técnicas de optimización

### 4.1 Cuantización INT8 posentrenamiento

Sin reentrenamiento. Activaciones y pesos a entero de 8 bits, esquema **S8S8**, **pesos
por canal**, calibración por **entropía** sobre **300 imágenes** de la partición de
validación de ImageNet-1k, con el mismo preprocesamiento de la sección 5 y disjuntas del
conjunto de evaluación. Script: `scripts/quantize_int8.py`, que imprime el SHA-256 del
artefacto y deja la lista de archivos de calibración como evidencia.

**Dos rutas, una por proveedor.** En CPU se ejecuta el grafo QDQ. En GPU ese grafo es
inejecutable: TensorRT 10.3 rechaza la construcción del motor sobre un nodo de
*dequantize* de sesgo en int32 que genera su propio analizador al procesar el sesgo FP32
de una convolución INT8, y declina el grafo completo cayendo a FP32 con latencia peor que
la nativa. La condición INT8 en GPU se construye desde el modelo FP32 con la calibración
nativa de TensorRT (`scripts/make_trt_calib_table.py`), generada sobre **el mismo conjunto
de 300 imágenes**.

Consecuencia para la interpretación: bajo INT8 las dos condiciones no ejecutan el mismo
artefacto, y el efecto de la técnica está confundido con el de la ruta. Los resultados se
acotan a las rutas de despliegue documentadas de cada backend. Ver D13, D14.

### 4.2 Poda estructurada de canales

DepGraph con importancia L1, global e iterativa hasta una fracción objetivo de MACs, sin
tocar el clasificador (se conservan las 1000 salidas). Se eligió estructurada y no por
magnitud porque la dispersión no estructurada no reduce latencia en hardware denso.

Puntos comprometidos, asimétricos: **ResNet-50 −53 % de MACs**, **MobileNetV2 −33 %**. La
red compacta se degrada antes. Los modelos podados se mantienen en **FP32** para no
introducir la precisión numérica como segunda variable. Script:
`scripts/prune_finetune.py`.

### 4.3 Recuperación tras la poda

Conjunto: subconjunto balanceado de la partición de entrenamiento de ImageNet-1k,
~100 imágenes por clase, **100 203 imágenes**. Se ejecuta en una máquina aparte con GPU de
escritorio, **nunca en el dispositivo de borde**.

| Parámetro | Valor |
|---|---|
| Optimizador | SGD |
| Tasa de aprendizaje inicial | 0,01, planificación coseno |
| Lote | 64 |
| Épocas | 15 |
| Suavizado de etiquetas | 0,1 |
| Precisión | mixta (AMP) |
| Partición | 95/5 para validación interna |
| Destilación | temperatura 4, peso 0,8 sobre el término KD |

**Asimetría deliberada en el criterio de parada.** El ajuste fino directo exporta el mejor
punto de control según la validación interna. La recuperación combinada de poda y
destilación (`scripts/prune_distill.py`) exporta la **última época**, no la mejor: elegir
por la validación interna premiaba el sobreajuste al subconjunto de recuperación, que es
justamente el efecto bajo estudio. Se declara porque afecta la comparación entre variantes.

Ambas variantes producen la misma arquitectura: latencia y energía son idénticas entre
ellas y solo la exactitud las distingue. Ver D15, D16, D18.

---

## 5. Canal de evaluación de exactitud

Conjunto oficial: **ImageNet-V2 matched-frequency**, 10 000 imágenes, 1000 clases,
independiente del usado para entrenar los modelos originales.

**Preprocesamiento** (`bench/datasets.py`), el estándar de torchvision para estas redes.
Se especifica porque altera el top-1 en varios puntos si se cambia:

1. Redimensionado del **lado corto a 256 px**, interpolación **bilineal**, conservando la
   relación de aspecto.
2. **Recorte central de 224×224**.
3. Conversión a punto flotante en `[0,1]`.
4. Normalización con media `(0.485, 0.456, 0.406)` y desviación `(0.229, 0.224, 0.225)`.

El cargador recorre las clases en **round-robin**, de modo que una evaluación truncada con
`--limit` siga siendo representativa. Las cifras oficiales usan el conjunto completo, sin
`--limit`.

**Independencia del dispositivo.** La exactitud es propiedad del artefacto, no del equipo:
las líneas base arrojan el mismo top-1 en las dos condiciones de la Jetson (0,596/0,596 y
0,694/0,695). Las cifras oficiales se midieron en la Jetson; la verificación contra la
validación completa de ImageNet (50 000 imágenes) se ejecutó fuera del dispositivo de
borde por costo de cómputo y cubre los artefactos no cuantizados.

---

## 6. Energía

**Instrumento.** INA226 externo, sensado en **lado alto** sobre la entrada de continua de
la placa, shunt de **0,1 Ω** (R100). Registro desde un **host independiente** —no el equipo
medido— a través de un puente USB-I2C CP2112, a **20 muestras por segundo** (`--interval
0.05`). Dirección I2C fija (`--addr`, 0x40 en la Jetson): la autodetección desincroniza el
bus. Los sensores internos del SoC se conservan solo como referencia cruzada.

**Ventana de integración.** Cada corrida escribe en sus metadatos la ventana `[t0, t1]` de
su ejecución, y la energía es la integral de la potencia sobre ella
(`scripts/energy_from_window.py`).

> **La ventana energética es más ancha que el tiempo de inferencia.** Cubre las 100
> iteraciones de calentamiento, las 2000 cronometradas y la sobrecarga del arnés entre
> iteraciones; la energía por inferencia divide entre las 2000 cronometradas. En
> MobileNetV2 sobre GPU eso son 3,133 ms de ventana por inferencia frente a 2,471 ms de
> latencia, y la sobrecarga relativa crece cuanto más rápida es la inferencia.
>
> **Dividir la energía por inferencia entre la latencia NO devuelve la potencia media.** La
> identidad que sí se cumple, en las cuatro condiciones, es:
>
> `energía por inferencia ÷ ventana por inferencia = potencia media`
>
> Comprobación: 36,587 mJ ÷ 3,133 ms = 11,68 W, la potencia reportada.

**Reposo.** Medido una vez con el equipo sin carga y en el mismo modo de potencia:
**7,80 W**. Se resta como constante en todas las condiciones, de modo que la energía neta
es comparable entre ellas por construcción. Se reportan la total y la neta. El valor
implícito `(total − neta) / ventana` es 7,80 W exactos en las doce condiciones.

---

## 7. Análisis estadístico

### 7.1 Cadena de cálculo

Es una sola y produce todos los números publicados.

1. Cada corrida entrega **2000 latencias crudas**.
2. De ellas se calcula su **media geométrica**: `gm_corrida = exp(media(log x))`.
3. La **tendencia central de una condición** es la media geométrica de esas R medias por
   corrida: `gm_cond = exp(media(log gm_corrida))`.
4. Toda **razón** entre dos condiciones es el cociente de esas tendencias centrales.

El orden importa y por eso se declara: **no** se calcula una razón por corrida para
promediarla después. Se promedia en escala logarítmica dentro de cada condición y se
dividen los resultados.

### 7.2 Intervalos de confianza

En escala logarítmica, sobre la diferencia de log-medias:

```
diff = media(log_num) − media(log_den)
se   = sqrt( s²_num/n_num + s²_den/n_den )      # varianza ENTRE CORRIDAS
gl   = n_num + n_den − 2
IC95 = exp( diff ± t(0.975, gl) · se )
```

**La unidad de réplica es la corrida, nunca la inferencia individual.** Las 2000
iteraciones de una corrida no son independientes entre sí; tratarlas como réplicas
produciría intervalos espurios. Implementado en `ratio_ci()`.

### 7.3 Percentiles

p50, p95 y p99 se calculan sobre las **muestras crudas combinadas** de las R corridas y
describen la **distribución**, incluida la cola. **No son base de ninguna razón.** Como las
distribuciones son asimétricas a la derecha, la media geométrica y la mediana no coinciden
y un cociente de medianas no reproduce la brecha. Ambas familias se publican para que la
diferencia sea visible.

### 7.4 Dispersión entre corridas

CV de las medias geométricas por corrida, publicado por condición en la sección 1b de
`results/OE3_ANALISIS.md`. En las doce condiciones de la Jetson está entre **0,09 % y
0,71 %**. Es la evidencia auditable de homogeneidad de cada campaña.

### 7.5 Contraste no paramétrico (ART)

Transformación de rangos alineados sobre el diseño factorial dispositivo × técnica.
**Unidades: las medias geométricas por corrida en escala logarítmica** —30 observaciones en
el diseño de dos dispositivos (2 × 3 × 5)—, las mismas que alimentan los intervalos. Los
grados de libertad reportados lo confirman: (2, 24) = 30 − 6 parámetros. Implementado en
`art_anova()`; el CSV `results/oe3_tidy_runs.csv` permite verificarlo en R/ARTool.

> **Limitación: el estadístico está saturado.** El alineamiento de la interacción separa
> las seis celdas en bloques de rango **contiguos y disjuntos** —cada celda ocupa cinco
> rangos consecutivos—, de modo que F y η²p dependen solo de esa estructura y resultan
> **idénticos entre modelos** (F = 433,3; η²p = 0,97). Confirma que la separación es
> completa, no cuánto lo es. Citar η²p como tamaño de efecto sería engañoso.

### 7.6 Por qué no se usan valores p

Con campañas de miles de inferencias casi cualquier diferencia resulta "significativa", de
modo que el valor p deja de discriminar. Las conclusiones se apoyan en tamaños de efecto e
intervalos al 95 %. Directriz del director, D11.

---

## 8. Reproducción

```bash
git clone <repositorio> && cd edge-ai-benchmark
git lfs install && git lfs pull          # trae los 8 ONNX reales
python scripts/analyze_oe3.py            # regenera OE3_ANALISIS.md, el CSV y las figuras
```

`scripts/analyze_oe3.py` parte de `results/*.json` —los datos crudos de cada corrida, con
sus 2000 latencias, metadatos y checksums— y reproduce todas las tablas y figuras
publicadas. No hay pasos manuales entre los archivos y los números.

**Referencias cruzadas:** `docs/DECISIONS.md` (por qué de cada decisión),
`docs/BITACORA.md` (cronología), `docs/POWER_MEASUREMENT.md` (montaje del medidor),
`docs/DISENO_INT8_OE1.md` (diseño de la cuantización), `docs/RUNBOOK.md` (operación).
