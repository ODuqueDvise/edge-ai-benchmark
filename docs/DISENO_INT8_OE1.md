# Nota de diseño — Cuantización INT8 (primera técnica del OE1)

Borrador para revisar. Resume el enfoque de la cuantización INT8 sobre los dos modelos
comprometidos (MobileNetV2 y ResNet-50) y se condensa en la decisión **D13**. Es la
primera técnica del orden acordado con el director (D10: INT8 → poda estructurada →
destilación). No cambia el protocolo congelado (D6) ni el dataset de evaluación (D12).

## Objetivo

Producir una variante INT8 de cada modelo y medirla en las tres condiciones
(`jetson-gpu`, `jetson-cpu`, `rpi-cpu`) con el mismo protocolo, para cuantificar el
efecto de la cuantización en tamaño, precisión, latencia y energía, y responder la
pregunta del OE1: **¿la optimización ensancha o cierra la brecha GPU vs CPU?**

## La complicación central (por la que esta nota existe)

INT8 no es un solo artefacto idéntico entre backends. Según la documentación de
ONNX Runtime [verificado]:

- En **CPU** (proveedor CPU EP, para `jetson-cpu` y `rpi-cpu`), la cuantización
  estática produce un modelo ONNX **cuantizado en formato QDQ** (nodos
  Quantize/DeQuantize con las escalas calculadas en calibración).
- En **GPU** (proveedor TensorRT EP, para `jetson-gpu`), el flujo documentado es
  distinto: TensorRT recibe el **modelo en precisión completa + un resultado de
  calibración** y decide internamente cómo cuantizar; no consume un QDQ por defecto.

Es decir, "la variante INT8" puede materializarse de dos formas. Coherente con el
arnés, que ya mide la precisión **por (variante, condición)**: la cuantización en CPU
y en GPU se calibra sobre los **mismos datos**, pero el cuantizador difiere, y eso se
reporta de forma explícita, no se esconde.

## Decisión (D13): PTQ estática, QDQ, un artefacto por modelo

1. **Cuantización estática post-entrenamiento (PTQ)**, no dinámica: es la recomendada
   para CNN y deja escalas fijas (comparables entre corridas). Formato **QDQ**, esquema
   **S8S8** (activaciones y pesos int8), **pesos per-canal**. Calibración de activaciones
   por **Entropy o Percentile** (más robustas a atípicos que MinMax). Pre-procesado con
   `quant_pre_process` (inferencia de formas + fusión) antes de cuantizar. Modelos en
   opset 18 (cumplen el mínimo de opset 10/12 que exige la cuantización).

2. **Un artefacto QDQ INT8 por modelo**: `models/cnn_baseline_int8.onnx` y
   `models/resnet50_baseline_int8.onnx`. Al ser **archivos distintos con checksum
   distinto**, el arnés —ya consciente del modelo— los distingue del V0 **sin ningún
   cambio**, y `RESULTS_LOG.md` los separa solo.

3. **Ejecución por condición**:
   - `jetson-cpu` y `rpi-cpu`: cargar el QDQ directamente en el proveedor CPU EP.
   - `jetson-gpu`: intentar correr el **mismo QDQ** en el proveedor TensorRT EP por
     *cuantización explícita* (TensorRT honra los nodos QDQ). **Esto es un supuesto a
     validar** [Probable, no verificado contra nuestro hardware].

4. **Gate de validación #1 (antes de medir nada en serio):** confirmar en la Orin que
   TensorRT con el QDQ realmente corre en INT8 (motor INT8, no *fallback* a FP16/FP32)
   y da *speedup*. Si TensorRT rechaza el QDQ o no acelera, se cae al **plan B**: dejar
   el modelo FP32 y usar la **calibración nativa de TensorRT** (CalibrationDataReader →
   tabla/flatbuffer), que es el camino que documenta ORT y el de sus ejemplos de
   ResNet-50. En ese caso, el modelo de GPU sería el mismo `.onnx` FP32 (mismo checksum
   que el V0), así que **habría que añadir una etiqueta de variante** (`--variant int8`)
   al arnés para no colisionar con el V0 al agrupar por checksum. Por eso se prefiere el
   QDQ unificado: si valida, no toca el arnés.

## Conjunto de calibración (sin fuga de datos)

La calibración fija los rangos de las activaciones a partir de datos de entrada (no de
etiquetas), pero calibrar sobre el mismo conjunto de evaluación es una fuga leve. Para
evitarlo:

- **Decidido (jun 2026):** calibrar con **~256–500 imágenes de ImageNet-1k val**,
  separadas del ImageNet-V2 de prueba → **sin fuga de datos** (lo más defendible).
- La **evaluación oficial sigue siendo el ImageNet-V2 completo (10.000)**, igual que el
  V0, para que las cifras sean comparables.
- El cuantizador (`quantize_int8.py`) toma un `--calib-dir` agnóstico (cualquier carpeta
  de imágenes con el preprocesamiento estándar), así que la fuente de datos queda
  desacoplada del código. **El mismo conjunto** alimenta la cuantización de CPU (ORT) y,
  si se cae al plan B, la calibración de TensorRT, vía un `CalibrationDataReader`
  determinista. Se versiona la **lista de archivos** usados (no las imágenes), como evidencia.

## Precisión y el riesgo de MobileNetV2 (D9)

MobileNetV2 es sensible a la cuantización (convoluciones separables; rango de pesos
amplio). Mitigaciones, en orden: **per-canal** por defecto; si la caída de top-1 es
grande, probar Percentile/Entropy y **excluir capas sensibles** (la API de *debugging*
de ORT permite localizar dónde diverge más); como último recurso, **QAT**
(entrenamiento consciente de cuantización) reentrenando en PyTorch y reexportando —es
costoso, por lo que se evita salvo necesidad. Se reporta **Δtop-1 vs V0 por condición**;
una caída por elección de modelo (no por la técnica) es en sí un hallazgo.

## Métricas, tamaño y artefactos

- **Tamaño**: INT8 reduce ~4× el peso [Probable]; MobileNetV2 ~14→~4 MB, ResNet-50
  ~102→~26 MB. Ambos quedan **bajo 100 MB**, así que los `*_int8.onnx` **sí se versionan
  en git** (excepción explícita en `.gitignore`, como `cnn_baseline.onnx`); se publica su
  checksum.
- **Latencia / energía / precisión**: mismo protocolo congelado (warmup 100, iters 2000,
  R=5, entrada 1,3,224,224) y mismo medidor; energía con el orquestador
  (`measure_remote.py`). Precisión sobre el V2 completo.

## Expectativa honesta (no asumir el resultado)

INT8 solo acelera donde el hardware lo soporta: *tensor cores* en GPU y CPU ARM con
instrucciones de producto punto. La RPi 5 (Cortex-A76) y la Orin (Cortex-A78) tienen
DotProd [Probable], así que el *speedup* INT8 en CPU es plausible —pero **hay que
medirlo, no darlo por hecho**: con kernels no óptimos puede ser modesto o incluso
negativo (el doc de ORT lo advierte). En GPU, TensorRT INT8 suele dar el mayor salto.
La hipótesis razonable es que INT8 **acelera más a la CPU que a la GPU** (la GPU ya
estaba cerca de su techo), lo que **estrecharía** la brecha del V0 —pero es justo lo
que el experimento debe decidir.

## Qué construir (implementación)

1. `scripts/quantize_int8.py`: `quant_pre_process` → `quantize_static` (QDQ, S8S8,
   per-canal, calibración Entropy/Percentile) con un `CalibrationDataReader` sobre el
   conjunto de calibración; imprime el SHA-256 del `*_int8.onnx`.
2. Validar el QDQ en TensorRT EP en la Orin (gate #1). Documentar el resultado.
3. Si se cae al plan B: extender el backend de ONNX Runtime para las opciones INT8 de
   TensorRT y añadir `--variant` al arnés.
4. Versionar los `*_int8.onnx`, actualizar `.gitignore`, README/RUNBOOK y la matriz
   experimental (filas V1 ya existen). Registrar checksums.

## Fuera de alcance de esta nota

No decide poda estructurada ni destilación (vienen después en el orden D10) ni toca el
protocolo (D6) o el dataset de evaluación (D12).
