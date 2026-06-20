# Bitácora del proyecto

Entradas cronológicas de estado. Ver el procedimiento en `docs/REGISTRO.md`.

## 2026-06-15 — Orlando (Jetson)
- Jetson Orin Nano operativa: arranque desde NVMe (JetPack 6.2), stack CUDA 12.6 / TensorRT 10.3 validado.
- Banco de medición construido y en repositorio público; protocolo congelado (D6).
- Línea base V0 (MobileNetV2 sin optimizar) en jetson-gpu y jetson-cpu: latencia (R=5) y precisión (10k, ImageNet-V2). Hallazgo: GPU ~5x más rápida que CPU sin pérdida de precisión.
- Automatización: `sync_results.sh` (sube resultados) y `build_results_log.py` (genera el log).
- Correo de avance al director con 3 puntos para visto bueno y adjunto de resultados.
- Pendiente: rpi-cpu (Luis), energía (a la espera del INA226 + CP2112), decisión de alcance de modelos (D9), entrada al OE1 (optimización).
- Energía: validado el medidor externo (CP2112 + INA226, shunt R100 0.1 Ω); reposo ~7.8 W, coherente con la referencia interna. jetson-gpu (V0): potencia media 11.68 W, 36.6 mJ/inf total y 12.1 mJ/inf neta. Pendiente jetson-cpu y rpi-cpu.
- Hardware: el shunt R100 (0.1 Ω) topa en ~0.82 A; la RPi requiere un R010 (0.01 Ω).
- Herramientas: `energy_from_window.py` ahora escribe `results/energy_*.json` y `build_results_log.py` lo integra al log automáticamente.

## 2026-06-20 — Orlando (Jetson)
- Cerrada la comparación GPU vs CPU del baseline V0 en la Jetson (tres métricas):
  - Latencia: GPU 2.468 ms vs CPU 12.280 ms (~5×).
  - Precisión: equivalente (top-1 0.5959 vs 0.5961).
  - Energía/inf (total / neta): GPU 36.6 / 12.1 mJ vs CPU 156.8 / 52.3 mJ (~4.3×).
- Hallazgo: GPU y CPU consumen casi la misma potencia (~11.7 W); la ventaja energética de la GPU viene de su velocidad, no de menor consumo. A igual precisión, la GPU domina en latencia y energía.
- Matiz metodológico: la razón de energía (4.3×) queda algo por debajo de la de latencia (4.98×) porque la ventana de GPU incluye el costo único de construir el motor TensorRT; en estado estable se acerca al ~5×.
- Pendiente: rpi-cpu (Luis) para la comparación de dispositivo; entrada al OE1 (optimización: INT8, poda, destilación).

## 2026-06-20 — Respuesta del director (CP2)
- El director aprobó el avance (NVMe, banco montado, repositorio público desde el día 1, línea base coherente) y respondió las tres consultas. Decisiones registradas en DECISIONS D9–D12:
  - Alcance (D9): comprometer MobileNetV2 + las técnicas y AÑADIR ResNet-50 como segundo modelo comprometido (con lugar explícito en el cronograma, no condicional). Prefirió ResNet-50 sobre EfficientNet por su mayor margen para cuantización/poda.
  - Técnicas (D10): INT8 → poda estructurada → destilación al final. La poda en la Orin solo baja latencia si es estructurada; la destilación va de última por su costo.
  - Estadística (D11): no forzar ANOVA clásico; log + Aligned Rank Transform; conclusiones por tamaño de efecto e IC, no por p-valores; cola con mediana/p95/p99.
  - Dataset (D12): ImageNet-V2 confirmado; reportar como "V2"; mismo conjunto en ambos modelos.
- Confirmó GPU vs CPU intra-Jetson como comparación principal y la RPi como referencia de despliegue (ratifica D4).
- Implicación: la matriz experimental se duplica; el cronograma debe ubicar ResNet-50 de forma explícita. Próximo: actualizar Matriz_Experimental y el cronograma (Excel), y arrancar el OE1 con INT8 sobre MobileNetV2.

## 2026-06-20 — Arnés multi-modelo (preparación del 2º modelo, CP4)
- Con ResNet-50 confirmado, se hizo el arnés consciente del modelo para que los resultados de los dos modelos no se mezclen:
  - `metadata.py` añade `model.name`; `run_benchmark.py` y `run_accuracy.py` incluyen el modelo en el nombre del JSON (`<cond>_<modelo>_<backend>_<prov>_<fecha>.json`); `energy_from_window.py` igual.
  - `build_results_log.py` etiqueta por nombre de modelo con respaldo al sha; sigue agrupando por sha, así que los JSON viejos de MobileNetV2 no se rompen. Verificado con dos JSON simulados → dos filas separadas y etiquetadas.
  - `.gitignore`: `cnn_baseline.onnx` (~14MB) versionado; ResNet-50 (~100MB) NO se versiona —supera el límite de 100MB de GitHub—, se comparte por archivo + checksum.
  - Docs actualizadas (README, RUNBOOK, QUICKSTART_JETSON/RPI/ACCURACY) al flujo de dos modelos y a las constantes congeladas (100/2000 donde quedaban 50/1000).
- ResNet-50 exportado (opset 18, 102.4 MB, pesos IMAGENET1K_V2, SHA-256 `05e5bc14444e89b9b47b36c663bc40e061db8d20389d833dcde3c7da667290dc`) y copiado a la Jetson por scp. Pendiente: correr su línea base V0 (jetson-gpu/cpu por Orlando; rpi-cpu por Luis).

## 2026-06-20 — Línea base V0 de ResNet-50 en la Jetson (CP1)
- Medido jetson-gpu (TensorRT) y jetson-cpu, R=5, precisión sobre el set completo (10k V2):
  - Latencia p50: GPU 6.59 ms vs CPU 89.9 ms → **13.6×** (muy estable, CV < 1%).
  - Precisión: GPU 0.6944 / CPU 0.6946 top-1 (equivalente); top-5 0.8859 en ambas.
  - Energía/inf con medidor externo (total / neta): GPU 97.8 / 43.1 mJ vs CPU 1108.7 / 370.0 mJ → **11.3× total, 8.6× neta**. Potencia media GPU 13.93 W vs CPU 11.71 W.
  - Matiz: la razón de energía (11.3× total) queda por debajo de la de latencia (13.6×) porque la GPU saturada tira más potencia; en energía neta baja a 8.6× porque su consumo activo sobre el reposo (6.1 W) supera al de la CPU (3.9 W).
- Hallazgo clave: el speedup de la GPU depende del modelo — **13.6× en ResNet-50 vs 4.98× en MobileNetV2**. ResNet-50 (denso, ~4.1 GFLOPs, convoluciones 3×3) satura la GPU y los tensor cores; MobileNetV2 (~0.3 GFLOPs, convoluciones separables en profundidad, memory-bound) la infrautiliza. Al pasar de MobileNetV2 a ResNet-50 la CPU paga ~7.3× más latencia (12.3→89.9 ms) y la GPU solo ~2.7× (2.47→6.59 ms): por eso la brecha se ensancha.
- Implicación para la tesis: el aporte de la GPU no es constante; el diseño eficiente de MobileNetV2 (lo que lo hace bueno para CPU/borde) es justo lo que limita su ganancia en GPU. Refuerza el contraste de dos modelos (D9).
- El aviso `device_discovery … /sys/class/drm/card1` es cosmético (sondeo de ONNX Runtime en la GPU integrada); el 13.6× confirma que TensorRT estuvo activo.
- Con esto el V0 de AMBOS modelos está completo en la Jetson (3 métricas × 2 condiciones). Pendiente: rpi-cpu de ambos modelos (Luis, requiere shunt R010 0.01 Ω); arrancar OE1 (INT8 sobre MobileNetV2).

## 2026-06-20 — Orquestador de medición remota (CP4, herramienta)
- Nuevo `scripts/measure_remote.py` (Mac): una condición de punta a punta por SSH —chequeos (reloj NTP, checksum del modelo, autotest INA226) → logger local → latencia R → la Jetson commitea (sync_results) → Mac pull → guardia de proveedor (aborta si GPU cae a CPU) → energía → commit—. Wrapper `measure_jetson_model.sh` corre gpu+cpu de un modelo. Trae `--dry-run`.
- Diseño: los JSON de dispositivo viajan por git (no rsync), lo que elimina la colisión de archivos sin trackear. Aborta en rojo ante cualquier anomalía: automatiza la plomería, no el criterio.
- Prerrequisitos (una vez): SSH por llave (`ssh-copy-id`) y NTP en ambas máquinas. Documentado en README y RUNBOOK.

## 2026-06-20 — Guía paso a paso para Luis (RPi)
- Nueva `docs/GUIA_LUIS_RPI.md`: camino lineal de cero a resultados para la condición `rpi-cpu` (clone → git/SSH a GitHub → entorno → modelos+checksum → dataset → pinning → latencia R=5 → precisión → sync), con verificaciones por paso y troubleshooting.
- Cierra huecos que el `QUICKSTART_RPI` no cubría para Luis: setup de git/SSH (identidad noreply propia para evitar GH007, llave SSH, colaborador), recepción de ResNet-50 por archivo, y precisión (antes solo en QUICKSTART_ACCURACY). Energía marcada como diferida (falta shunt R010).
- Enlazada desde README y RUNBOOK.

## 2026-06-20 — Nota de diseño INT8 / OE1 (D13, registrar antes de ejecutar)
- Antes de implementar la primera técnica del OE1 se registró el enfoque en `docs/DISENO_INT8_OE1.md` y como D13 en DECISIONS: PTQ estática QDQ (S8S8, per-canal, Entropy/Percentile), un `*_int8.onnx` por modelo (checksum distinto → el arnés los distingue sin cambios), CPU EP directo y TensorRT por cuantización explícita sobre el mismo QDQ.
- Punto a validar primero: que TensorRT acelere el QDQ en INT8 real; si no, plan B = calibración nativa de TensorRT + etiqueta `--variant`. Verificado en la doc de ORT que la cuantización estática es la recomendada para CNN y que el camino GPU documentado usa calibración de TensorRT (de ahí el plan B).
- Calibración sin fuga (set aparte del de evaluación); evaluación oficial sigue siendo el V2 completo. Pendiente: construir `scripts/quantize_int8.py` y pasar el gate de validación en la Orin.

## 2026-06-20 — quantize_int8.py construido y validado (OE1)
- `scripts/quantize_int8.py`: PTQ estática QDQ (S8S8, per-canal, calibración entropy/percentile/minmax) reusando `bench.datasets.preprocess_image` (preprocesamiento idéntico al de inferencia); CalibrationDataReader sobre `--calib-dir`; escribe la lista de calibración como evidencia e imprime el SHA-256.
- Validado en seco (sandbox): MobileNetV2 14.2 MB → 4.0 MB (3.5×), QDQ válido, carga y corre en ORT CPU (salida 1×1000). Falta el gate en GPU (TensorRT con QDQ) en la Orin.
- `.gitignore`: `*_int8.onnx` y `*.calib.txt` se versionan (<100MB). Calibración decidida: ImageNet-1k val (~256–500), aparte del V2.
- Pendiente: conseguir el set de calibración (ImageNet-1k val, gated en HF), cuantizar ambos modelos, pasar el gate TensorRT, y medir las 3 condiciones con el orquestador.

## 2026-06-20 — Variantes INT8 cuantizadas (OE1, pendiente gate GPU)
- Calibrado con 300 imágenes de ImageNet-1k val (Camino 1, gate de HF aceptado), método entropy, per-canal. Resultados: MobileNetV2 14.2→4.0 MB (3.5×, SHA-256 `124fd2a4f9e60301274145644390145536e9bff07794c866108237f1eb510753`); ResNet-50 102.4→26.2 MB (3.9×, SHA-256 `ed792dcaf3ea0f2461492824d3674b6efaf904e1f6d6e9cbe1c5b0237b20a493`). Ambos verifican en ORT CPU (salida 1×1000).
- Etiquetas en build_results_log (`124fd2a4`→MobileNetV2 INT8, `ed792dca`→ResNet-50 INT8). Los `*_int8.onnx` se versionan (git, <100MB).
- Pendiente: gate de GPU (que TensorRT honre el QDQ en INT8 real) antes de la batería completa; luego R=5 + precisión + energía en las 3 condiciones.

## 2026-06-20 — Gate INT8 en GPU fallido → plan B (D14)
- El QDQ de ResNet-50 (sha `2161a04a`, QuantizeBias=False) está limpio: 0 DequantizeLinear de bias en int32 (verificado cargando el modelo). Pese a ello, TensorRT 10.3 rechaza el motor con "Error Code 4 ... node_Conv_753_bias_dq: input has type Int32" (nodo generado por el propio parser de TensorRT) y declina el grafo: cae a CUDA, p50 32.8 ms (más lento que el FP32 de 6.59 ms; sin INT8).
- Conclusión: el camino QDQ-unificado-en-TensorRT queda descartado empíricamente tras dos intentos (incluido el ajuste QuantizeBias=False + ActivationSymmetric). Se activa el plan B (D14): CPU INT8 con el QDQ (funciona); GPU INT8 con calibración nativa de TensorRT (camino documentado por ORT).
- Implica cambios en el arnés: opciones INT8 del proveedor TensorRT en el backend, etiqueta `--variant` (para no colisionar el checksum del FP32 con el V0) y un script de tabla de calibración. CPU INT8 puede medirse ya con el QDQ; GPU INT8 espera estos cambios.
