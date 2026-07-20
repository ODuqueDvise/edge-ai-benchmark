# Bitácora del proyecto

Entradas cronológicas de estado. Ver el procedimiento en `docs/REGISTRO.md`.

## 2026-06-15 — Jetson
- Jetson Orin Nano operativa: arranque desde NVMe (JetPack 6.2), stack CUDA 12.6 / TensorRT 10.3 validado.
- Banco de medición construido y en repositorio público; protocolo congelado (D6).
- Línea base V0 (MobileNetV2 sin optimizar) en jetson-gpu y jetson-cpu: latencia (R=5) y precisión (10k, ImageNet-V2). Hallazgo: GPU ~5x más rápida que CPU sin pérdida de precisión.
- Automatización: `sync_results.sh` (sube resultados) y `build_results_log.py` (genera el log).
- Correo de avance al director con 3 puntos para visto bueno y adjunto de resultados.
- Pendiente: rpi-cpu (Luis), energía (a la espera del INA226 + CP2112), decisión de alcance de modelos (D9), entrada al OE1 (optimización).
- Energía: validado el medidor externo (CP2112 + INA226, shunt R100 0.1 Ω); reposo ~7.8 W, coherente con la referencia interna. jetson-gpu (V0): potencia media 11.68 W, 36.6 mJ/inf total y 12.1 mJ/inf neta. Pendiente jetson-cpu y rpi-cpu.
- Hardware: el shunt R100 (0.1 Ω) topa en ~0.82 A; la RPi requiere un R010 (0.01 Ω).
- Herramientas: `energy_from_window.py` ahora escribe `results/energy_*.json` y `build_results_log.py` lo integra al log automáticamente.

## 2026-06-20 — Jetson
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
- ResNet-50 exportado (opset 18, 102.4 MB, pesos IMAGENET1K_V2, SHA-256 `05e5bc14444e89b9b47b36c663bc40e061db8d20389d833dcde3c7da667290dc`) y copiado a la Jetson por scp. Pendiente: correr su línea base V0 (jetson-gpu/cpu por mí; rpi-cpu por Luis).

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

## 2026-06-20 — Plan B INT8 en GPU FUNCIONA (gate superado)
- jetson-gpu INT8 con el camino documentado: modelo FP32 + tabla de calibración de TensorRT (`make_trt_calib_table.py`) + `--variant int8 --trt-int8-table`. SIN Error Code 4. p50 = **2.452 ms** (R=1) vs 6.59 ms FP32 → 2.69×; throughput 397.8 ips; potencia interna 8.01 W. Avisos menores: 2 tensores de cola (linear_output, broadcast) sin rango de calibración → caen a FP (impacto despreciable; el grueso corre INT8).
- Hallazgo preliminar (latencia, R=1) que CONTRADICE la hipótesis: el INT8 **NO** estrecha la brecha GPU-CPU. CPU 89.9→35.8 ms (2.51×), GPU 6.59→2.452 ms (2.69×); la brecha pasa de 13.6× a ~14.6× (se ensancha levemente). El INT8 acelera ambos por un factor similar; los tensor cores de la GPU rinden tanto como el INT8 de la CPU ARM (Cortex-A78, DotProd).
- Pendiente: batería R=5 + **precisión** (eje aún sin medir: el INT8 puede degradar top-1, y CPU/GPU usan cuantizadores distintos) + energía, en las 3 condiciones; añadir `--variant`/`--trt-int8-table` al orquestador para automatizar la GPU; opcionalmente, completar la tabla de calibración para que esos 2 tensores también corran INT8.

## 2026-06-20 — Orquestador con soporte INT8 (listo para la batería)
- `measure_remote.py` ahora pasa `--variant` y `--trt-int8-table` a la latencia y a la precisión, e incluye la variante en el patrón de archivos del guardia de proveedor y la energía. Verificado en dry-run.
- La batería oficial INT8 queda en un comando por condición desde el Mac: GPU = FP32 + tabla de calibración + `--variant int8`; CPU = el modelo QDQ. Prerrequisito: SSH por llave (ssh-copy-id) y medidor conectado para `--shunt`.

## 2026-06-20 — Campaña oficial INT8 en la Jetson COMPLETA (CP1)
- Batería completa vía orquestador (R=5 + precisión 10k + energía), 4 condiciones, auto-commiteada. INT8 en GPU por plan B (FP32 + tabla TensorRT, variante int8); en CPU por el QDQ.
- ResNet-50: GPU FP32 6.59→INT8 2.45 ms (2.69×), CPU FP32 89.9→INT8 35.8 ms (2.51×). Top-1 (10k): GPU 0.6944→0.6833, CPU 0.6946→0.6744 (caída 1-2 pts). Energía neta/inf: GPU 43.1→8.4 mJ, CPU 370→161.7 mJ. Brecha 13.6×→14.6×.
- MobileNetV2: GPU FP32 2.468→INT8 1.81 ms (1.36×), CPU FP32 12.28→INT8 12.31 ms (**1.00×, sin ganancia**). Top-1: GPU 0.596→0.591, CPU 0.596→0.589. Energía neta/inf: GPU 12.1→3.0 mJ, CPU 52.3→48.3 mJ. Brecha 5.0×→6.8×.
- HALLAZGOS: (1) el INT8 NO estrecha la brecha GPU-CPU; la ENSANCHA en ambos modelos. (2) En CPU el beneficio del INT8 depende de si el modelo es compute-bound (ResNet-50: 2.5×) o memory-bound (MobileNetV2 depthwise: sin ganancia): el INT8 recorta cómputo, no tráfico de memoria. (3) Precisión casi intacta en ambos (MobileNetV2 NO se degradó → D9 infundado); el cuantizador de CPU (QDQ) degrada algo más que el de GPU (TensorRT) en ResNet-50 (−2.0 vs −1.1 pts). (4) El INT8 ahorra energía en todas las condiciones, más donde también acelera.
- Pendiente: rpi-cpu INT8 (Luis). OE1/INT8 cerrado en la Jetson; siguiente técnica del orden D10: poda estructurada.

## 2026-06-20 — Subconjunto de recuperación para la poda (cerrado)
- Para el reentrenamiento de recuperación de la poda se bajó un subconjunto balanceado de ImageNet-1k de entrenamiento en el Legion (WSL2): ~100 img/clase, 1000 clases, 100 203 imágenes, 3.9 GB en ext4. Guía: `docs/SETUP_IMAGENET_SUBSET.md`.
- Streaming desde HuggingFace con buffer pequeño (`buffer_size=2000`) + reanudación desde disco para no agotar la RAM de WSL (el buffer de 10000 causaba `Killed`). 24 clases quedaron entre 68 y 99 img (mínimo 68); se aceptó así —imbalance trivial para fine-tuning de recuperación—. JPGs truncados: manejo diferido al cargador de entrenamiento (`ImageFile.LOAD_TRUNCATED_IMAGES`), no a un escaneo aparte.

## 2026-06-20 — Verificación del mapeo de clases (paso 0 de la poda)
- `scripts/verify_class_mapping.py`: evalúa el preentrenado de torchvision sobre el subconjunto usando el índice de carpeta como etiqueta. Resultado: ResNet-50 top-1 90.7% (top-5 98.7%), MobileNetV2 77.5% (93.8%) sobre 4000 muestras → el índice de Hugging Face (0000–0999) COINCIDE con el orden de torchvision. No hay que reordenar etiquetas.

## 2026-06-20 — Fase de poda estructurada iniciada (OE1, técnica 2)
- `scripts/prune_finetune.py`: poda estructurada (torch-pruning/DepGraph, importancia L1, global iterativa hasta una fracción de MACs objetivo, sin tocar el clasificador) + reentrenamiento de recuperación (AMP, SGD+coseno, label smoothing, checkpoint por época con resume, dataloader tolerante a corruptos) + export a ONNX FP32 (opset 18, archivo único, SHA-256), idéntico en formato a la línea base.
- Parámetros (D15): ResNet-50 conservar ~50% de MACs, MobileNetV2 ~70% (asimétrico por ser compacto/memory-bound). FP32 para aislar la variable poda. Corre en el Legion (RTX 3060, ~1–1.5 h/modelo estimado); el ONNX se copia a la Jetson para EXP-07/08 (MobileNetV2) y 19/20 (ResNet-50). RPi (09/21) a Luis.
- Máquina de entrenamiento: Lenovo Legion 5 17ACH6H, RTX 3060 Laptop 6 GB, Windows 11 + WSL2 Ubuntu. NO es dispositivo de medición; solo produce los modelos podados.

## 2026-06-20 — ResNet-50 podado: artefacto exportado (para EXP-19/20)
- Poda + reentrenamiento + export completos en el Legion. ResNet-50: MACs 4.12→1.94 G (−53%, conserva 47%), parámetros 25.56→17.42 M (−32%), ONNX 102.4→69.8 MB (−32%: el tamaño sigue a los parámetros, no a los MACs). 15 épocas; val top-1 sobre subconjunto train 95.99% (cifra de salud, NO la precisión real —esa sale del V2 en la Jetson).
- `models/resnet50_pruned.onnx`, FP32, opset 18, SHA-256 `940aefb80c3ea650da12300b6170f74fbd220e8ff76c6306d4db131197473957`. <100 MB → versionable en git si se decide (la base de 102 MB no lo era).
- Pendiente: MobileNetV2 podado (--target-macs 0.7); luego copiar ambos ONNX a la Jetson y medir EXP-19/20 y 07/08 con el orquestador.

## 2026-06-20 — MobileNetV2 podado: artefacto exportado (para EXP-07/08)
- MobileNetV2: MACs 0.32→0.21 G (−33%, conserva 67%), parámetros 3.50→3.16 M (−10%), ONNX 14.2→12.8 MB (−10%). Los parámetros casi no bajan: el clasificador (1280→1000, ~1.28 M) no se poda y el depthwise casi no tiene parámetros; la poda global recorta canales caros en cómputo, no en peso. 15 épocas; val top-1 (subconjunto train) 87.05% (salud, no precisión real —esa sale del V2 en la Jetson).
- `models/cnn_pruned.onnx`, FP32, opset 18, SHA-256 `7be5303c91aef53d0b110ecac8509bcf0c16aa22d925a72f9330070b6d3c9226`.
- Ambos modelos podados listos (ResNet-50 `940aefb8…`, MobileNetV2 `7be5303c…`). Siguiente: copiar a la Jetson y medir EXP-19/20 (ResNet) y 07/08 (MobileNet) con el orquestador.

## 2026-06-20 — Modelos podados en la Jetson + medición en curso (EXP-19/20, 07/08)
- Los dos ONNX podados se copiaron del Legion (WSL) a la Jetson vía el Mac (WSL → carpeta de Windows → scp Mac → scp Jetson), con huellas verificadas: ResNet-50 `940aefb8…`, MobileNetV2 `7be5303c…`.
- Legion habilitado para push a GitHub: llave SSH propia `legion-orlando`, remoto por SSH (igual que la Jetson). Logs de poda/reentrenamiento versionados en `logs/` del repo (push desde el Legion).
- Batería de medición EN CURSO en la Jetson vía orquestador (`measure_jetson_model.sh … --accuracy`): latencia R=5 + precisión V2 (10k) + energía (INA226 conectado), condiciones gpu+cpu, ambos modelos. El podado se mide como el V0 (FP32, archivo y sha propios; sin `--variant` ni tabla de calibración). Resultados → EXP-19/20 (ResNet) y EXP-07/08 (MobileNet); rpi-cpu pendiente (Luis).
- Pendiente al cerrar la batería: consolidar p50 gpu/cpu, top-1 V2 y energía en la matriz, el documento y la bitácora; leer si la poda estrecha la brecha GPU-CPU (contraste con INT8) y la precisión real frente al val de entrenamiento (91.6%/87.1%, distribución de train).

## 2026-06-21 — ResNet-50 podado medido (EXP-19/20) + problema de recuperación detectado
RESULTADOS (ResNet-50 podado −53% MACs, FP32, sha 940aefb8, frente al V0):
- Latencia p50: GPU 6.59→5.11 ms (1.29×), CPU 89.9→47.5 ms (1.89×). Brecha GPU-CPU 13.6×→9.3×.
- Energía neta/inf: GPU 43.1→27.6 mJ, CPU 370→202 mJ. Throughput GPU 196 ips, CPU 20.9 ips.
- Precisión top-1 V2 (10k): GPU 0.5104 / CPU 0.5106 (top-5 0.744) frente a V0 0.694 → −18 puntos.
HALLAZGO (latencia, sólido): la poda ESTRECHA la brecha GPU-CPU (13.6→9.3×), al revés del INT8 (que la ensanchó a 14.6×). El recorte de MACs favorece a la CPU (cómputo-acotada, 1.89×) más que a la GPU (a lote 1 no estaba limitada por cómputo, 1.29×).
PROBLEMA (precisión, a corregir): −18 pts NO es el costo real de la poda, es sobre-ajuste del reentrenamiento a las 100k. Pista: val de entrenamiento 95.99% vs test V2 51% (45 pts de brecha = memorización). Causas: (a) 100 img/clase es poco para recuperar una poda agresiva; (b) el checkpoint se eligió por el val de entrenamiento, que premia al más sobre-ajustado. NO reportar este número de precisión tal cual.
PLAN (para retomar):
  1. Medir MobileNetV2 podado (−33%, suave, mismas 100k) -> su precisión V2 dice si la poda moderada recupera con estos datos [EN CURSO al anotar; cierra EXP-07/08].
  2. Re-entrenar ResNet a poda suave -30% MACs (~1 h en el Legion), artefactos aparte para conservar el de -53%:
     python scripts/prune_finetune.py --model resnet50 --target-macs 0.7 --data ~/imagenet_train_subset --epochs 15 --out models/resnet50_pruned_p30.onnx --ckpt ~/ck_resnet50_p30.pth
     -> diagnóstico + 2º punto de la curva latencia-precisión.
  3. Si aun así cae: escalar recuperación -> más imágenes/clase, o destilación desde el modelo sin podar (técnica 3). Y NO elegir checkpoint por el val de entrenamiento.
  La latencia/energía/brecha son válidas y se conservan; solo el eje de precisión queda pendiente de una recuperación adecuada. Anotar al director: la poda necesita datos de recuperación suficientes, a diferencia del INT8/PTQ que no reentrena.

## 2026-06-21 — MobileNetV2 podado medido (EXP-07/08) + diagnóstico de recuperación
RESULTADOS (MobileNetV2 podado −33% MACs, FP32, sha 7be5303c, frente al V0):
- Latencia p50: GPU 2.468→2.34 ms (1.05×), CPU 12.28→8.59 ms (1.43×). Brecha GPU-CPU 5.0×→3.67×.
- Energía neta/inf: GPU 12.1→7.1 mJ, CPU 52.3→37.4 mJ. Throughput GPU 402 ips, CPU 113 ips.
- Precisión top-1 V2 (10k): GPU 0.4532 / CPU 0.4536 (top-5 0.700) frente a V0 0.596 → −14 puntos.
HALLAZGO CLAVE (cross-técnica): en la CPU memory-bound de MobileNetV2 la PODA acelera 1.43× donde el INT8 no movió nada (1.00×). La poda quita canales completos (menos trabajo y menos tráfico de activaciones); el INT8 en depthwise de la CPU ARM no rindió. La brecha GPU-CPU: el INT8 la ENSANCHA (5.0→6.8×), la poda la ESTRECHA (5.0→3.67×) — consistente con ResNet (13.6→9.3×).
DIAGNÓSTICO (recuperación): MobileNet se podó SUAVE (−33%) y aun así la precisión cayó −14 pts con la misma firma de sobre-ajuste (val train 87% vs test 45%). CONCLUSIÓN: el problema NO es la agresividad de la poda sino la recuperación (100/clase + selección por val de train). Se DESCARTA el re-entrenamiento de ResNet a −30% (no arreglaría la precisión). El arreglo es la recuperación: más imágenes/clase o destilación; decisión de alcance con el director.

## 2026-06-21 — Cierre de sesión: estado de la poda y próximos pasos
ESTADO. Fase de poda (OE1, técnica 2) medida en la Jetson, ambos modelos, GPU+CPU:
- Latencia/energía/brecha: SÓLIDAS. La poda ESTRECHA la brecha GPU-CPU (ResNet 13.6→9.3×, MobileNet 5.0→3.67×), al revés del INT8 que la ensancha. Hallazgo clave: en la CPU memory-bound de MobileNet la poda acelera 1.43× donde el INT8 no movió nada (1.00×).
- Precisión: PROVISIONAL, no reportable. Ambos cayeron por sobre-ajuste a las 100/clase: ResNet 0.694→0.510 (−18), MobileNet 0.596→0.453 (−14). La poda suave (MobileNet −33%) también cayó → la causa es la recuperación, no la agresividad.
- Datos crudos: JSON en results/ (auto-commiteados por la Jetson). Artefactos: resnet50_pruned.onnx (940aefb8, 69.8 MB), cnn_pruned.onnx (7be5303c, 12.8 MB), ambos FP32 opset 18.
PRÓXIMOS PASOS (en orden, para la siguiente sesión):
  1. Subir desde el Mac los docs sin commitear: git add docs/BITACORA.md docs/DECISIONS.md -> commit -> pull --rebase -> push.
  2. Llevar al director la decisión de alcance sobre la precisión de la poda: invertir en recuperación adecuada (más imágenes/clase o destilación) vs reportar el costo bajo recuperación limitada como hallazgo. NO re-entrenar a −30% (descartado: la poda suave también sobre-ajustó).
  3. Según esa decisión: ampliar el subconjunto y re-entrenar, o montar recuperación por destilación (toca técnica 3); luego re-medir precisión.
  4. Consolidar EXP-07/08/19/20 en la matriz (latencia/energía firmes; precisión cuando se resuelva) y en la tabla de resultados del Word (Retos_Tecnicos_Reproducibilidad.docx, ya tiene el reto 4.7).
  5. Pendiente de Luis: rpi-cpu para V0, INT8 y poda (necesita shunt R010).
REFERENCIAS: repo DECISIONS D15 (parámetros poda) y D16 (problema de recuperación); scripts/prune_finetune.py; docs/SETUP_LEGION_CUDA.md y SETUP_IMAGENET_SUBSET.md.

## 2026-06-21 — Recuperación por destilación (KD) implementada; ResNet KD entrenado
- Nuevo `scripts/prune_distill.py`: recuperación de la poda por destilación (maestro = modelo sin podar, congelado; pérdida mixta KD [KL sobre logits suavizados, T=4] + CE dura, alpha=0.8). Poda y export idénticos a prune_finetune (L1 global, FP32, opset 18). Corrige el sesgo de selección: exporta la ÚLTIMA época, no la "mejor" por val de train.
- ResNet-50 KD (−53% MACs) entrenado y exportado: `models/resnet50_pruned_kd.onnx`, sha `efffe63b3e650cd67fbfa237bd5c5311b1dff6a25b94e9d2de85c3bf18292efb`, 69.8 MB. Val de entrenamiento 92.68% (vs 95.99% del fine-tuning normal → menos memorización; falta confirmar en V2).
- EN CURSO: precisión V2 del KD en la Jetson (accuracy-only, jetson-gpu). Latencia/energía NO se re-miden (arquitectura idéntica a la poda con fine-tuning normal → se heredan). Referencias: poda normal 0.510, sin podar 0.694. Gate: si el KD sube claro (>~0.60) → correr MobileNet KD; si no, el cuello es de datos (llevar al director).

## 2026-06-21 — ResNet KD: precisión V2 (la destilación recupera parcialmente)
- ResNet-50 podado −53% MACs, recuperado por DESTILACIÓN: top-1 V2 = 0.5787 (top-5 0.8034), vs 0.510 del fine-tuning normal (+6.8 pts) y 0.694 sin podar. Latencia p50 5.10 ms = idéntica a la poda normal (confirma misma arquitectura; latencia/energía se heredan).
- Lectura: la destilación SÍ funciona como recuperación (+6.8 pts top-1, +6 pts top-5 sobre el fine-tuning normal), pero queda ~12 pts bajo el modelo sin podar → ayuda pero no sustituye datos; el residual es límite del subconjunto de 100/clase. Hallazgo cuantificado: poda −53% MACs cuesta −18 pts con fine-tuning normal, de los cuales la destilación recupera ~7 (a −12); cerrar el resto exige más datos.
- Pendiente: MobileNet KD (par del experimento) + consolidar el set completo V0 / poda-FT / poda-KD para ambos modelos.

## 2026-06-21 — MobileNet KD: precisión V2 + experimento de poda CERRADO
- MobileNetV2 podado −33% MACs, recuperado por DESTILACIÓN: top-1 V2 = 0.5083 (top-5 0.7544), vs 0.453 del fine-tuning normal (+5.5 pts) y 0.596 sin podar. Latencia 2.34 ms = idéntica a la poda normal.
- SET COMPLETO de la poda (top-1 V2): ResNet-50 0.694 (V0) → 0.510 (FT) → 0.579 (KD); MobileNetV2 0.596 → 0.453 → 0.508. La destilación recupera +5.5/+6.8 pts; residual −9/−12 pts = límite de datos (100/clase).
- CIERRE del experimento de poda. Historia: la poda ESTRECHA la brecha GPU-CPU (al revés del INT8) y rescata la CPU memory-bound de MobileNet; su costo de precisión es real pero parcialmente recuperable por destilación, con el residual atribuible al presupuesto de datos de recuperación.
- Pendiente: consolidar en Word (en curso); rpi-cpu (Luis); destilación como técnica 3 independiente (estudiante compacto).

## 2026-06-21 — Git LFS para todos los ONNX
- Se versionan TODOS los ONNX en el repo vía Git LFS (incluido `resnet50_baseline.onnx` ~102 MB, que antes se compartía por archivo + checksum). `.gitattributes` enruta `*.onnx`; el `.gitignore` solo excluye los ONNX temporales de cuantización. Decisión D17.
- Requiere git-lfs en cada máquina para clonar/actualizar y obtener los archivos reales (si no, se reciben punteros de texto): `git lfs install` (Linux: `sudo apt install -y git-lfs`; Mac: `brew install git-lfs`). Setup inicial en el Mac: `git lfs track "*.onnx"` → add `.gitattributes` → commit → push.
- Instrucciones propagadas a RUNBOOK, GUIA_LUIS_RPI y QUICKSTARTs. Los ONNX ya commiteados como blobs normales (cnn_baseline, *_int8) se quedan así; LFS aplica a lo nuevo (no se reescribe historia).

## 2026-06-21 — Orquestador documentado como multiplataforma (host cualquiera)
- Corrección de premisa: `measure_remote.py` corre desde cualquier máquina anfitriona (macOS/Linux/Windows), no solo el Mac; lee el medidor por hidapi (multiplataforma). Generalizado en README y RUNBOOK ("el Mac" → "el host"), conservando lo que sí es de mi Mac (export con pyenv, clon canónico).
- Nueva guía `docs/SETUP_HOST_WINDOWS.md`: cómo correr el orquestador en Windows. Recomendado WSL2 (entorno idéntico a la doc); para el medidor USB-HID, `usbipd-win` para adjuntarlo a WSL, o Windows nativo (hidapi sin driver). Latencia/precisión no requieren host. Puntero agregado en GUIA_LUIS_RPI (Luis puede estar en Windows).

## 2026-06-22 — rpi-cpu V0 validado + análisis de despliegue; guía ampliada a INT8/poda
- Luis subió la línea base V0 en rpi-cpu (latencia R=5 + precisión, ambos modelos; sin energía). Validado: la precisión coincide con la Jetson (ResNet-50 0.695, MobileNetV2 0.596) → modelo y dataset correctos. Notas para Luis: ResNet subió con 10 corridas (protocolo R=5); MobileNetV2 con CV 9% y cola gorda (revisar gobernador `performance` y refrigeración).
- Análisis de despliegue (V0): latencia geom rpi-cpu ResNet 150.8 ms, MobileNet 25.2 ms. La CPU del RPi5 es 1.67×/2.0× más lenta que la CPU de la Jetson. Brecha de despliegue Jetson-GPU vs RPi-CPU: 22.9× (ResNet) / 9.9× (MobileNet) —un orden de magnitud—. Descompone en aceleración GPU (13.7×/4.9×) × mejor CPU del SoC (1.67×/2.0×): el grueso es el acelerador.
- Energía rpi sigue diferida (necesita dongle INA226+CP2112 + shunt R010). INT8 y poda NO necesitan el dongle (solo latencia+precisión). Guía de Luis ampliada (`GUIA_LUIS_RPI.md` §9) con comandos INT8 (`*_int8.onnx`) y poda (`*_pruned.onnx`), el `git lfs pull` y el aviso de que la poda baja la precisión a propósito.

## 2026-06-22 — Conciliación de guías RPi + voz unificada
- Se eliminó la duplicación entre `GUIA_LUIS_RPI.md` (guía completa, única fuente del setup) y `QUICKSTART_RPI.md`, reducida a tarjeta de comandos (V0/INT8/poda + constantes), que remite a la guía para clonar/entorno/gobernador. Evita que diverjan.
- Voz unificada: la documentación (README, RUNBOOK, QUICKSTART, POWER, bitácora, guía de Luis) habla en primera persona; "Orlando" en tercera persona se quitó salvo identificadores literales (git user.name, llave SSH, host de ejemplo).

## 2026-06-22 — Cierre de sesión: estado y próximos pasos
HECHO esta sesión:
- Poda (OE1 técnica 2) cerrada en la Jetson: latencia/energía/brecha definitivas; precisión recuperada parcialmente por destilación (KD). Consolidada en la matriz y en el Word (retos 4.x, secciones 7 y 8).
- Análisis OE3: `scripts/analyze_oe3.py` (tendencia central log, tamaños de efecto + IC, cola), reporte `results/OE3_ANALISIS.md`, figura `results/oe3_brecha_gpu_cpu.{png,pdf}`, sección 8 del Word. Hallazgo: el INT8 ensancha la brecha GPU-CPU, la poda la estrecha (IC estrechos).
- Git LFS: todos los ONNX versionados (D17).
- rpi-cpu V0 (Luis) validado (precisión = Jetson) + análisis de despliegue: brecha Jetson-GPU vs RPi-CPU 22.9× (ResNet) / 9.9× (MobileNet); descompone en GPU (13.7×/4.9×) × mejor CPU del SoC (1.67×/2.0×).
- Docs: orquestador generalizado a cualquier host + `SETUP_HOST_WINDOWS.md`; voz en primera persona; guías RPi conciliadas (GUIA_LUIS completa, QUICKSTART_RPI = tarjeta de comandos).
PRÓXIMOS PASOS (orden):
  1. Subir desde el Mac la tanda de docs sin commitear: `git add README.md docs/` → commit → `pull --rebase` → push.
  2. Pedirle a Luis INT8 + poda en rpi-cpu (latencia + precisión, SIN energía; guía §9). Energía sigue diferida (medidor + shunt R010).
  3. Cuando Luis suba: consolidar rpi-cpu como 3ª columna en `analyze_oe3.py` + cerrar la referencia de despliegue; correr el ART factorial (diseño completo).
  4. Técnica 3: destilación (estudiante compacto) — última del OE1, sin empezar.
  5. Decisión del director sobre la precisión de la poda (nota `Nota_Director_Avance_Poda.md`).
  6. Redacción de capítulos de tesis.

## 2026-06-22 — Autodetección de dirección del INA226 + validación del medidor de Luis
- `ina226_cp2112_logger.py` ahora AUTODETECTA la dirección I2C del INA226: prueba la preferida (0x40) y, si no responde, escanea 0x40-0x4F validando los DOS registros de identidad del INA226 (fabricante y dispositivo), no solo un ACK; si varias direcciones responden (strap A0/A1 marginal) lo avisa. `--addr` pasa a ser solo preferencia. El orquestador lo hereda (llama al logger para autotest y registro), así que el dongle de Luis (que quedó estable en 0x44) funciona sin `--addr`, igual que el de la Jetson (0x40), sin tocar guías. Verificado con py_compile + test lógico con dispositivo simulado.
- Medidor de Luis validado funcionalmente: 17 min @ ~19 Hz, bus ~5.43 V, P=V·I consistente, sin valores no finitos. PENDIENTE físico antes de enviarlo: invertir IN+/IN− (la corriente salía negativa) y confirmar el valor real del shunt (`--rshunt`; cruce con multímetro).

## 2026-07-05 — Respuesta del director (1 jul): camino 1 con condiciones; el eje de precisión de la poda se REABRE
- Llegó la respuesta del director al avance del OE1 (correo 1 jul, carpeta `email/` del proyecto). Aprueba el camino 1 con dos condiciones y dos matices de redacción → registrado como D18. Lo esencial: (a) la atribución de la caída al presupuesto de recuperación queda CONDICIONAL hasta que la corrida KD lo demuestre —la corrida ya está hecha y la recuperación fue PARCIAL (+6.8/+5.5, residual −12/−9), así que por sí sola no zanja el criterio—; (b) chequeo obligatorio antes de cerrar: precisión de V0 y pruned_kd (ambos modelos) sobre la validación completa de ImageNet (50k), para separar la brecha natural val→V2 del sobre-ajuste. El "CIERRE del experimento de poda" anotado el 21 jun fue prematuro en el eje de precisión: vuelve a estado condicional en matriz y Word hasta cumplir ambas condiciones. Latencia/energía/brecha siguen firmes y se reportan aparte (el director lo pidió explícito). No invertir aún en recuperación completa.
- Verificación de integridad del repo (desde el entorno de trabajo): los 8 ONNX intactos (sha256 en disco = oid del puntero LFS en HEAD, los 8; en HEAD todos son punteros LFS, incluidos cnn_baseline y *_int8). Los CSV marcados como modificados solo difieren en fin de línea (`git diff --ignore-cr-at-eol` vacío). El estado "modificado" de los ONNX era ruido de un entorno sin git-lfs, no cambios reales.
- HILO ABIERTO: `scripts/ina226_cp2112_logger.py` tiene ediciones SIN commitear que REVIERTEN la autodetección del 25 jun — dirección I2C vuelve a ser fija (`--addr`, def 0x40) porque el barrido del CP2112 sobre direcciones muertas lo desincroniza y elige direcciones falsas; `--scan` queda como diagnóstico aparte. Hasta commitear y conciliar, la bitácora del 22 jun y el commit del 25 jun describen un comportamiento que ya no es el del árbol de trabajo. Confirmar prueba con el módulo de Luis (0x44) antes de commitear.
PRÓXIMOS PASOS (orden):
  1. Chequeo ImageNet-val (condición 2 de D18): descargar el split de validación (50k) y correr precisión de V0 y pruned_kd de ambos modelos (Legion o Jetson, arnés existente). Descompone los 14-18 pts en brecha natural de V2 vs sobre-ajuste real.
  2. Responder al director en un solo correo: cifras KD (ya medidas) + resultado del chequeo val, con la redacción de D18; preguntar explícito si la recuperación parcial con residual explicado satisface la condición 1 o abre el camino 2. NO responder antes del chequeo.
  3. Aplicar D18 en la matriz (eje de precisión de la poda → condicional) y en el Word (retos 4.x, secciones 7-8, matices de redacción).
  4. Commitear el logger INA226 de dirección fija (tras confirmar prueba en 0x44) y conciliar RUNBOOK/POWER_MEASUREMENT/bitácora con el cambio.
  5. En paralelo: INT8 + poda en rpi-cpu con Luis (guía §9); al subir, 3ª columna en `analyze_oe3.py` + ART factorial.
  6. Después: técnica 3 (destilación, estudiante compacto) y redacción de capítulos.

## 2026-07-05 — Chequeo ImageNet-val CUMPLIDO (condición 2 de D18); correo enviado al director
- Montaje en el Legion (WSL): descarga de la validación completa de ImageNet-1k (50k, streaming HF a carpetas por índice de clase, formato compatible con el cargador del arnés) + `onnxruntime-gpu` (requirió exportar LD_LIBRARY_PATH a las libs CUDA de torch: el wheel 1.27 pide libcudart.so.13, que solo vive en los paquetes pip de torch cu130). Sanidad con `--limit 2000`: 0.806 ≈ referencia. Seis corridas completas (V0/FT/KD × 2 modelos), JSON `acc_legion-gpu_*` en `results/` (push desde el Legion).
- RESULTADO (top-1 val 50k): ResNet-50 0.802 (V0) / 0.633 (FT) / 0.704 (KD); MobileNetV2 0.720 / 0.578 / 0.640. Baselines ≈ torchvision (0.809/0.722; la décima es el re-encode JPG q95).
- LECTURA: la brecha natural val→V2 (~11-12 pts en V0) se mantiene casi constante en los podados (amplificación 0.8-1.7 pts) → la hipótesis de que parte de los 14-18 pts fuera brecha del conjunto de prueba queda DESCARTADA con datos: la caída es costo real del presupuesto de recuperación. KD recupera +7.1/+6.2 pts reales sobre val; residual 8-10 pts = presupuesto de datos (100/clase). Esto FORTALECE el camino 1: hallazgo real, no artefacto de evaluación, parcialmente recuperable y cuantificado.
- Estatus de las corridas: diagnóstico de D18 (device-tag legion-gpu), NO condición de la matriz; las cifras oficiales siguen siendo Jetson/V2. La precisión es propiedad del artefacto+dataset (equivalencia entre dispositivos ya demostrada en V2 sobre 3 plataformas); opcional para reproducibilidad: repetir solo los 2 baselines val en la Jetson (adorno, no necesidad).
- Correo enviado al director (5 jul): cifras, redacción acordada, pregunta explícita de si las condiciones quedan cumplidas o abre camino 2; se mantiene sin invertir en recuperación completa.
PRÓXIMOS PASOS (orden):
  1. Mac: `git pull --rebase` (el Legion pushó los 7 JSON) y commit de esta bitácora + DECISIONS.
  2. Sin esperar al director (él ya dictó la redacción): aplicar los matices en el Word — "poda a esta tasa exigió presupuesto de recuperación no previsto" y "recuperación poda+destilación combinada" (retos 4.x, §7-8) — y marcar el eje de precisión de la poda como CONDICIONAL en la matriz.
  3. Con su respuesta: cerrar (o abrir camino 2) y consolidar las cifras val como anexo del capítulo de poda.
  4. Luis: INT8 + poda en rpi-cpu (guía §9); al subir, 3ª columna en analyze_oe3.py + ART factorial.
  5. Conciliar RUNBOOK/POWER_MEASUREMENT con el logger de dirección fija (docs aún describen autodetección).
  6. Técnica 3 (destilación, estudiante compacto) y redacción de capítulos.

## 2026-07-05 — Cierre de sesión: D18 aplicado en Word y matriz; guías conciliadas; Luis notificado
- D18 aplicado FUERA del repo (carpeta del proyecto): Word `Retos_Tecnicos_Reproducibilidad.docx` — estado al 5 jul con eje de precisión condicional; 4.7 retitulado «un presupuesto de datos mayor del previsto», recuperación etiquetada «combinada poda+destilación», chequeo ImageNet-val incorporado, frase explícita de que no es un veredicto general sobre la técnica; §7 reescrita con las cifras val (0,802→0,704 / 0,720→0,640). Matriz `Matriz_Experimental.xlsx` — EXP-07/08/19/20 pasan a «precisión CONDICIONAL (D18)»; hoja «Resultados poda» con nota de estatus, columna renombrada y sección C nueva con el chequeo val (marcado diagnóstico, no condición).
- Guías conciliadas con el logger de dirección fija. Corrección de premisa: las guías NUNCA describieron la autodetección (el cambio del 22 jun fue «sin tocar guías»); lo agregado es el régimen nuevo. POWER_MEASUREMENT: regla «--addr SIEMPRE fija» (Jetson 0x40 / Luis 0x44), --scan solo diagnóstico con el porqué, selftest por módulo, paso 0 de escaneo. RUNBOOK: nota en el paso 5. GUIA_LUIS §11: dirección 0x44 con ejemplos, invertir IN+/IN− antes del primer registro, confirmar shunt con multímetro, orquestador con --addr 0x44 (verificado en código que measure_remote lo propaga).
- Mensaje enviado a Luis: INT8 + poda en rpi-cpu (guía §9, sin energía), con el aviso de que la precisión de los podados sale baja a propósito.
- Operativo (Legion): acceso SSH restablecido por llave — la cuenta es Microsoft (contraseña no utilizable por OpenSSH en la práctica; «no autoritativo» para net user); usuario oduke SÍ es administrador, así que las llaves van en administrators_authorized_keys, creado con curl.exe (la creación por PowerShell dejaba el archivo ilegible para sshd, presumiblemente codificación). Para onnxruntime-gpu 1.27 en WSL: exportar LD_LIBRARY_PATH a las libs CUDA de los paquetes pip de torch cu130 (pide libcudart.so.13).
- EN ESPERA de externos: respuesta del director (cierre del eje de precisión o camino 2) y corridas de Luis.
- PENDIENTE DE COMMIT: docs/POWER_MEASUREMENT.md, docs/RUNBOOK.md, docs/GUIA_LUIS_RPI.md y esta bitácora.
PRÓXIMO ARRANQUE sugerido: redacción del capítulo de resultados con lo ya firme (V0, INT8, latencia/energía/brecha de la poda); insertar el cierre de precisión cuando el director responda.

## 2026-07-10 — Respuesta del director: D18 CERRADO (implícito) y paper ARTIIS aprobado (D19)
- Correo 1 (~9 jul): "el tema de la latencia lo podemos dejar allí", seguir con los demás objetivos, y propuesta de paper corto para ARTIIS 2026 (deadline 15 ago). Cierre implícito del eje de precisión de la poda: no se publica un resultado en disputa. D18 pasa a CERRADO con la redacción acordada.
- Correo 2 (~10 jul): aprueba el alcance congelado que propusimos (intra-Jetson, 2 modelos × 2 técnicas × 3 métricas, nada depende de rpi-cpu ni destilación), acepta coautoría como asesor académico (Orlando y Luis autores principales), y confirma financiación: la universidad cubre COP 600.000 por ponencia con publicación ISSN/ISBN (CCIS cumple). Participación virtual abarata (sin viaje).
- Verificado en el sitio del congreso: deadline 15 ago (extendido, no habrá otro), notificación 10 sep, versión final e inscripción 30 sep, evento 11-13 nov (Lisboa + Zoom), Springer CCIS (Scopus), revisión DOUBLE-BLIND (→ anonimizar el envío, incluido el enlace al repo), CFP en inglés.
- Registrado como D19. Técnica 3 pospuesta formalmente hasta después del 15 ago.
PRÓXIMOS PASOS (orden):
  1. Esquema del paper (secciones + qué va en cada una, en inglés) → enviarlo al director esta semana para visto bueno.
  2. Verificar tarifa real de inscripción de ARTIIS vs el tope de COP 600.000 (página de registro no publica monto; escribir a info@artiis.org si hace falta) y el límite de páginas del short paper (página de submission).
  3. Con el visto bueno: redactar sobre la plantilla LNCS/CCIS de Springer; versión anonimizada para el envío.
  4. Luis: siguen pendientes INT8+poda en rpi-cpu (entrarían al paper solo como referencia de despliegue si llegan antes del corte interno).
  5. Después del 15 ago: técnica 3 (destilación) y capítulos de tesis (el paper ES el borrador del capítulo de resultados).

## 2026-07-07 — rpi-cpu INT8 + poda: Luis subió; validación (precisión OK, latencia provisional por térmica)
- Luis subió el commit `d253972` (rpi-cpu: latencia R=5 + precisión de INT8 y poda, ambos modelos; sin energía). Bajado con `git pull --rebase` y validado contra la Jetson.
- INTEGRIDAD y PRECISIÓN: firmes. Los 4 sha256 de los JSON coinciden con el HEAD del repo (modelos correctos, no copias). Top-1 V2 = Jetson al tercer decimal: MobileNet INT8 0.5893 (Jetson 0.589), ResNet-50 INT8 0.6744 (=), MobileNet poda 0.4536 (0.453), ResNet-50 poda 0.5106 (0.510). Confirma la equivalencia de dispositivo en la 3ª plataforma → el eje de precisión de la RPi queda VALIDADO; no se re-mide (propiedad de modelo+dataset). La poda sale baja a propósito (esperado).
- LATENCIA: PROVISIONAL, no consolidable aún. p50 mediana / CV entre corridas / térmica: MobileNet INT8 31.1 ms / 12.9% / 70-73 °C; ResNet-50 INT8 67.8 ms / 6.9% / 74-76 °C; MobileNet poda 11.4 ms / 4.7% / 74-77 °C; ResNet-50 poda 62.0 ms / 1.7% / 78-82 °C. Dos causas: (a) TÉRMICA — ResNet-50 poda corrió a 81-82 °C y la RPi5 hace throttling desde 80 °C; su CV 1.7% no es garantía, es throttling estable (sesgo al alza); la batería corrió ~1 h (03:11→04:20) calentando en cascada (70→82 °C) sin cooldown entre condiciones. (b) RUIDO — MobileNet INT8 CV 12.9% con cola a 111 ms; es el problema de gobernador/refrigeración anotado el 22 jun (MobileNet CV 9%), sin resolver.
- HALLAZGO PRELIMINAR (no afirmable con estos datos): en MobileNet el INT8 EMPEORA la latencia (31.1 ms vs 25.2 ms del V0), coherente con la Jetson (depthwise memory-bound: el INT8 no recorta el tráfico de memoria). Requiere re-corrida limpia antes de escribirse.
- PRÓXIMOS PASOS: (1) Luis re-mide SOLO latencia de MobileNet INT8 y ResNet-50 poda (idealmente las 4) con gobernador `performance` + enfriamiento activo + cooldown <70 °C al arrancar cada condición; confirmar `vcgencmd get_throttled` = 0x0. Precisión NO se re-mide. (2) Con la latencia limpia: 3ª columna rpi-cpu en `analyze_oe3.py` + cerrar referencia de despliegue + ART factorial. (3) Nit: el sha de MobileNet INT8 en la entrada del 20 jun (`124fd2a4`) quedó viejo; el modelo real y medido es `c1eac3d6` → revisar el mapa de etiquetas de `build_results_log.py` para que no deje esa fila sin etiquetar.
- PENDIENTE DE COMMIT: esta entrada de bitácora.

## 2026-07-09 — Re-corrida fría de Luis validada; multimodalidad de MobileNet identificada; ancla FP32 en duda; fix de auditoría en el arnés
- Luis subió el commit `8f06655` (10 JSON: R=5 MobileNet INT8 + R=5 ResNet-50 poda, solo latencia). Integridad OK: sha256 de modelos `c1eac3d6` (MobileNet INT8) y `940aefb8` (ResNet-50 poda) = HEAD, proveedor CPUExecutionProvider puro, warmup 100 / iters 2000. Condición térmica del 7 jul CUMPLIDA en lo verificable: temperaturas post-corrida 54-58 °C (INT8) y 62-66 °C (poda), muy por debajo de la campaña caliente (70-73 / 78-82) y del umbral de throttling (80 °C).
- ResNet-50 poda: latencia CONSOLIDABLE. p50 por corrida 61.1-64.0 ms (mediana 61.9, CV ~2%) — idéntico a la campaña caliente del 7 jul (61.1-63.7). Dos campañas con ~17 °C de diferencia y el mismo p50 → el resultado es reproducible e insensible a la térmica en ese rango; el "throttling estable" que se temía el 7 jul no estaba sesgando esta condición. Oficial: campaña del 9 jul; la concordancia entre campañas se reporta como evidencia de robustez.
- MobileNet INT8: NO es ruido térmico, es MULTIMODALIDAD entre procesos. Cada corrida es plana por dentro (mediana por cuartos varía <0.1 ms) pero cada invocación cae en un modo distinto: p50 29.2 / 30.1 / 27.9 / 25.2 / 21.6 ms (rango 39%), con la latencia BAJANDO mientras la temperatura sube — descarta térmica. Mismo patrón en caliente el 7 jul (25.0-35.4). Hipótesis: lotería de colocación de memoria/hilos por invocación en modelo memory-bound. Mediana 27.9 ms PROVISIONAL; para el ART se necesita más R (10) y/o afinidad fija (`taskset`) documentada.
- VERDAD INCÓMODA — ancla FP32 contaminada: los baselines FP32 de rpi-cpu (22 jun) corrieron a 76-84 °C; ResNet-50 FP32 estuvo a 82-84 °C (sobre el umbral de 80), sus 138-152 ms están [probable] inflados por throttling, y son el DENOMINADOR de todo speedup en la RPi. MobileNet FP32 (73-77 °C) al borde y con el mismo patrón multimodal. NO calcular razones de mejora rpi-cpu hasta re-medir los 2 baselines FP32 con protocolo frío.
- Limitación del arnés detectada y CORREGIDA: `thermal_c_start` se capturaba POST-corrida (metadata.collect() corre después de run_latency), así que ambas temperaturas de los JSON históricos son post-corrida (separadas por segundos) — el "arranque en frío <44 °C" de Luis y su governor/get_throttled eran inverificables desde los datos. Fix en bench/: run_benchmark captura la temperatura ANTES de cargar el modelo y agrega `cpu_state_start/end` (governor y cur_khz por política cpufreq + `vcgencmd get_throttled` crudo) + aviso en consola si el gobernador no es performance. Verificado: py_compile OK, degradación a None en máquinas sin cpufreq/vcgencmd, override del fallback OK. CAVEAT de redacción: en los JSON previos a este fix, `thermal_c_start` ≈ temperatura final.
- PRÓXIMOS PASOS: (1) Luis: `git pull` (trae el arnés con auditoría) y re-medir SOLO latencia de los 2 baselines FP32 con el protocolo frío; MobileNet (FP32 e INT8) con R=10 por la multimodalidad. (2) Con eso: 3ª columna rpi-cpu en `analyze_oe3.py` + ART factorial + referencia de despliegue. (3) Decidir si se fija afinidad con `taskset` (y se documenta como constante congelada) o se absorbe la multimodalidad con más R.
- PENDIENTE DE COMMIT: bench/metadata.py, bench/run_benchmark.py y esta entrada de bitácora.

## 2026-07-14 — Baselines FP32 fríos validados; hipótesis de throttling REFUTADA; ancla canónica = kernel 1014; quedan 2 huecos
- Luis subió `123e881` (31 JSON, no 15: corrió DOS tandas de FP32 porque en la primera `vcgencmd` no estaba disponible en el sistema —Ubuntu raspi no lo trae— y la repitió completa con el binario instalado; total MobileNet ×21, ResNet ×10). La auditoría nueva del arnés FUNCIONA: governor `performance` en las 31 corridas, 2.4 GHz fijos, `get_throttled=0x0` donde hubo vcgencmd, arranques fríos reales de 39.7-43.5 °C y todo el resto <58 °C.
- CORRECCIÓN (refuta la entrada del 9 jul): la hipótesis del ancla FP32 "inflada por throttling" era FALSA. ResNet-50 FP32 en frío da MÁS LENTO (mediana 151.5 ms, rango 144.1-153.7, n=10) que la campaña caliente del 22 jun (140.3 ms a 82-84 °C). La térmica queda descartada por dirección del efecto. El corrimiento (~8%) coincide con el salto de kernel 7.0.0-1009→1014-raspi entre el 7 y el 9 jul [suposición: causa no atribuible retroactivamente; ResNet poda cruzó el mismo límite sin moverse 62.0→61.9]. Decisión operativa: el ancla canónica de rpi-cpu es la campaña FRÍA AUDITADA sobre kernel 1014 (11 jul); las cifras de 1009 no se mezclan en razones de mejora.
- MobileNet FP32: mediana 22.49 ms (n=21) ≈ campaña caliente (22.40) — el ancla chica nunca estuvo sesgada. La MULTIMODALIDAD persiste en frío con gobernador fijo (p50 por corrida 19.2-26.4, CV ~9%): es inherente al modelo memory-bound en la RPi, no térmica ni de gobernador. Con n=21 el ART la absorbe → NO se adopta taskset (habría cambiado una constante congelada); se documenta la dispersión.
- CONSOLIDADO en kernel 1014 (medianas de p50): MobileNet FP32 22.49 (n=21) · MobileNet INT8 27.88 (n=5, 9 jul) · ResNet FP32 151.48 (n=10) · ResNet poda 61.94 (n=5, 9 jul). Speedup limpio: ResNet poda 2.45×; INT8 en MobileNet EMPEORA 0.81× (ahora con ancla limpia: el hallazgo preliminar del 7 jul se CONFIRMA y es más fuerte que el 1.0× de la Jetson).
- HUECOS para columna de entorno único: ResNet INT8 (67.8) y MobileNet poda (11.4) solo existen en kernel 1009 → re-medir en 1014 (R=5 y R=10 respectivamente) + completar MobileNet INT8 de 5 a 10 corridas. ~35-40 min de medición de Luis. Provisionales con mezcla de kernel: ResNet INT8 2.24×*, MobileNet poda 1.98×*.
- BRECHA de despliegue Jetson-GPU vs RPi-CPU (medianas; * = provisional por mezcla de kernel): V0 9.1× / 23.0× (MobileNet / ResNet); INT8 15.4× / 27.7×*; poda 4.9×* / 12.1×. La 3ª plataforma REFUERZA el hallazgo central de OE3: el INT8 ENSANCHA la brecha (en MobileNet dramáticamente: 9.1→15.4×, porque en la RPi es contraproducente mientras la GPU sí gana), la poda la ESTRECHA (23.0→12.1× y 9.1→4.9×). Contexto D19: nada del paper depende de esto; si los 2 huecos se cierran antes de agosto, entra como referencia de despliegue.
- PRÓXIMOS PASOS: (1) Luis: ResNet INT8 R=5 + MobileNet poda R=10 + 5 corridas más de MobileNet INT8, mismo protocolo frío, en 1014. (2) Con eso: 3ª columna en `analyze_oe3.py` + ART factorial + cifra de despliegue para el paper. (3) Registrar en el capítulo de método la política "entorno único por columna" (kernel fijado por campaña) y el caveat de `thermal_c_start` pre-fix.
- PENDIENTE DE COMMIT: esta entrada de bitácora.

## 2026-07-19 — Columna rpi-cpu COMPLETA en entorno único (kernel 1014); tanda de Luis CERRADA
- Luis subió `794d682` (20 JSON del 18 jul: ResNet INT8 ×5, MobileNet poda ×10, MobileNet INT8 ×5 adicionales). Las 20 corridas auditadas y limpias: sha correctos (`2161a04a` ResNet INT8, `7be5303c` MobileNet poda, `c1eac3d6` MobileNet INT8), proveedor CPU único, governor `performance`, `get_throttled=0x0` inicio y fin, kernel 1014, temperaturas 43.6-66.7 °C, corridas planas por dentro. Nota operativa de Luis: un reinicio de la Pi revirtió gobernador (`ondemand`) y permisos de `/dev/vcio`; los rehizo antes de medir — el ajuste NO es persistente entre reinicios (documentar en GUIA_LUIS).
- COLUMNA rpi-cpu FINAL (kernel 1014, frío, medianas de p50): MobileNetV2 — FP32 22.49 (n=21), INT8 26.68 (n=10), poda 11.01 (n=10). ResNet-50 — FP32 151.48 (n=10), INT8 74.37 (n=5), poda 61.94 (n=5). Speedups de entorno único: MobileNet INT8 0.84× (CONTRAPRODUCENTE, confirmado con n=10), MobileNet poda 2.04×, ResNet INT8 2.04×, ResNet poda 2.45×.
- BRECHA de despliegue Jetson-GPU vs RPi-CPU (medianas): V0 9.1× / 23.0× (MobileNet / ResNet); INT8 14.8× / 30.4×; poda 4.7× / 12.1×. Tercera plataforma con el MISMO patrón de OE3: el INT8 ensancha la brecha, la poda la estrecha. Cifras listas para la referencia de despliegue del paper (D19) y la 3ª columna de la tesis.
- Multimodalidad caracterizada: MobileNet INT8 con n=10 muestra TRES modos discretos reproducidos en ambas tandas (≈21.7 / ≈25.3 / ≈28-30 ms); ResNet INT8 también presenta modos en 1014 (72.9-79.2 y uno en 89.6) — la multimodalidad no es exclusiva del modelo chico. Se absorbe con R y se documenta; sin taskset (constantes congeladas intactas).
- Confirmación del corrimiento 1009→1014 con datos nuevos: afecta SOLO al ResNet sin podar (FP32 +8%, INT8 +10%); poda y MobileNet ~0%. Causa raíz fuera de alcance; la política de entorno único por columna queda validada como salvaguarda del diseño.
- DECISIÓN: tanda CERRADA; se confirma a Luis. El kernel de la RPi permanece CONGELADO hasta terminar la campaña de energía rpi-cpu (INA226 0x44 pendiente de físico: invertir IN+/IN− y confirmar shunt), para que latencia y energía compartan entorno.
- PRÓXIMOS PASOS: (1) 3ª columna en `analyze_oe3.py` + ART factorial (sesión aparte, repo limpio). (2) GUIA_LUIS: agregar que gobernador y permisos de /dev/vcio se rehacen tras cada reinicio (o dejar servicio systemd). (3) RUNBOOK: regla de congelar kernel/ORT durante una campaña y citar `platform.release` en la matriz. (4) Energía rpi-cpu cuando el dongle esté listo.
- PENDIENTE DE COMMIT: esta entrada de bitácora.

## 2026-07-19 (2) — analyze_oe3 con 3ª columna rpi-cpu + ART factorial en el script; guías actualizadas
- `scripts/analyze_oe3.py`: (a) rpi-cpu como 3er dispositivo con filtro de ENTORNO ÚNICO — solo entra latencia con `platform.release == 7.0.0-1014-raspi` (constante `RPI_RELEASE`); las corridas 1009 (22 jun / 7 jul) quedan fuera automáticamente; (b) selección de corridas: Jetson conserva R=5 estricto (números publicados INTACTOS, verificado), RPi usa TODAS las corridas oficiales (la multimodalidad se absorbe con R alto); (c) sección 2b nueva: brecha de despliegue Jetson-GPU↔RPi-CPU con IC95; (d) sección 5 nueva: ART factorial 2 vías EN el script (alineado por medias de celda + rangos + ANOVA Tipo III con codificación de efectos, soporta celdas desbalanceadas; solo numpy/scipy, sin dependencias nuevas) — el CSV para ARTool/R se mantiene como verificación externa; (e) figura nueva `oe3_brecha_despliegue.png/pdf` (mismo estilo escala de grises).
- RESULTADOS (medias geométricas, IC95): brecha de despliegue V0 23.0× [22.6, 23.4] / 8.9× [8.3, 9.7] (ResNet / MobileNet); INT8 la ensancha a 30.8× [28.0, 33.8] / 14.4× [13.3, 15.7]; poda la estrecha a 12.3× [12.1, 12.6] / 4.7× [4.5, 4.8]. Aceleración en RPi vs V0: ResNet INT8 1.93× [1.77, 2.11], poda 2.40× [2.34, 2.46]; MobileNet INT8 0.85× [0.78, 0.93] (FRENA, IC excluye 1) y poda 1.99× [1.89, 2.09]. ART: interacción dispositivo×técnica dominante en los 4 diseños (η²p 0.78-0.97, p<0.001) — confirmación no paramétrica del hallazgo central de OE3 en 2 y 3 dispositivos.
- Guías: GUIA_LUIS §6 — gobernador y permisos de `/dev/vcio` NO persisten tras reinicio (visto 18 jul), chequeo post-reinicio + servicio systemd opcional + regla de no actualizar el SO con tanda abierta. RUNBOOK — regla transversal de ENTORNO ÚNICO POR COLUMNA con la evidencia del salto 1009→1014.
- Operativo: los 4 artefactos generados viejos (OE3_ANALISIS, CSV, figura) estaban expulsados por iCloud ("Resource deadlock avoided" al sobreescribir); se borraron y regeneraron. Recordatorio: mantener la carpeta "Descargado siempre" en iCloud.
- PENDIENTE DE COMMIT: scripts/analyze_oe3.py, docs/GUIA_LUIS_RPI.md, docs/RUNBOOK.md, results/OE3_ANALISIS.md, results/oe3_tidy_runs.csv, results/oe3_brecha_gpu_cpu.*, results/oe3_brecha_despliegue.*, y esta bitácora.
- SIGUIENTE: redacción del paper ARTIIS con estas cifras como referencia de despliegue opcional (D19); energía rpi-cpu cuando el dongle esté listo.
