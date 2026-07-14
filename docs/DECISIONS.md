# Registro de decisiones

Formato breve tipo ADR. Una entrada por decisión que afecte alcance, metodología o
protocolo. Ver el procedimiento en `docs/REGISTRO.md`.

### D1 — Plataformas: Jetson Orin Nano + Raspberry Pi 5 (may 2026)
- Decisión: reducir de 4 plataformas a 2 (ambas 8 GB de RAM).
- Motivo: son las disponibles; misma RAM aísla la GPU como variable.
- Consecuencia: el eje del estudio pasa a GPU vs CPU en el borde.

### D2 — Tres objetivos específicos; detalle técnico en metodología (jun 2026)
- Decisión: 3 OEs generales; modelos/técnicas/parámetros se definen en metodología.
- Motivo: objetivos defendibles y estables ante ajustes técnicos (directriz del director).

### D3 — Modelo canónico: MobileNetV2 (jun 2026)
- Decisión: línea base MobileNetV2 preentrenado, ONNX opset 18, SHA-256 609015cb…
- Motivo: compatible con ImageNet sin reentrenar; estándar de borde.
- Consecuencia: se comparte por archivo y se verifica por checksum; no se reexporta.

### D4 — Condición jetson-cpu explícita; baseline en 3 condiciones (jun 2026)
- Decisión: medir GPU vs CPU dentro de la misma Jetson como comparación principal.
- Motivo: aísla el aporte de la GPU; Jetson vs RPi confunde acelerador con CPU.

### D5 — Dataset de precisión: ImageNet-V2 matched-frequency (jun 2026)
- Decisión: 10.000 imágenes, 1000 clases; compatible directo con MobileNetV2.
- Descartado: Imagenette (muy grueso), CIFAR (exige reentrenar).

### D6 — Protocolo congelado (jun 2026)
- Decisión: warmup 100, iters 2000, R 5, modo MAXN (Jetson) / governor performance (RPi), entrada 1,3,224,224.
- Motivo: corrida piloto, CV del p50 = 0.56%. No se cambia sin consenso.

### D7 — Energía: medidor externo INA226 + CP2112 (jun 2026)
- Decisión: sensado en lado alto sobre la entrada DC, leído por CP2112 desde el portátil.
- Descartado: Joulescope (costo), medidores USB-C (la Jetson es barril-only).

### D8 — HRM fuera de alcance (jun 2026)
- Decisión: no incluir modelos de razonamiento jerárquico.
- Motivo: otra clase de workload; no encaja en OE1/OE2; fracturaría diseño y cronograma.

### D9 — Alcance de modelos: dos modelos comprometidos (CONFIRMADO por el director, jun 2026)
- Decisión: comprometer MobileNetV2 + las técnicas, y AÑADIR ResNet-50 como segundo modelo comprometido, con lugar explícito en el cronograma (no condicional "si el tiempo da").
- Motivo (director): ResNet-50 es denso y con redundancia → más margen para cuantización y poda; el contraste "modelo ya eficiente (MobileNetV2) vs modelo con margen (ResNet-50)" queda más limpio. Se prefirió ResNet-50 sobre EfficientNet.
- Consecuencia: la matriz experimental se duplica (2 modelos × técnicas × 3 condiciones); el cronograma debe ubicar el segundo modelo de forma explícita. El arnés ya soporta el cambio (otro .onnx + checksum). Reemplaza la propuesta condicional del `Concepto_Alcance_Modelos.md`.

### D10 — Técnicas de optimización: tipo y orden (director, jun 2026)
- Decisión: cuantización INT8 → poda ESTRUCTURADA → destilación al final.
- Motivo (director): en la Orin la poda solo baja latencia si es estructurada (la no estructurada deja dispersión que el hardware denso no aprovecha); la destilación va de última por su costo.
- Consecuencia: implementar poda por canales/filtros (no por magnitud), con reentrenamiento de recuperación; la destilación es el ítem flexible si el cronograma aprieta.

### D11 — Método estadístico: tamaño de efecto sobre p-valores (director, jun 2026)
- Decisión: no forzar el ANOVA clásico. Transformación logarítmica para la tendencia central; Aligned Rank Transform (ART) si se conserva el diseño factorial. Conclusiones apoyadas en tamaños de efecto e intervalos de confianza, no en p-valores (con miles de inferencias casi todo sale "significativo"). La cola se reporta con mediana, p95 y p99.
- Consecuencia: definir herramienta para ART (R/ARTool o equivalente); reportar tamaño de efecto + IC en cada comparación. Afecta el análisis del OE3.

### D12 — Confirmación de dataset (director, jun 2026)
- Decisión: ratifica D5 (ImageNet-V2). Reportar siempre como "V2" y usar el mismo conjunto en ambos modelos.

### D13 — Cuantización INT8: PTQ estática QDQ, un artefacto por modelo (jun 2026)
- Decisión: PTQ estática en formato QDQ (S8S8, pesos per-canal, calibración Entropy/Percentile) sobre MobileNetV2 y ResNet-50; un `*_int8.onnx` por modelo (archivo y checksum distintos → el arnés los distingue sin cambios). CPU EP corre el QDQ directo; en GPU se intenta el mismo QDQ en TensorRT (cuantización explícita).
- Gate de validación: confirmar que TensorRT corre el QDQ en INT8 real con speedup; si no, plan B = calibración nativa de TensorRT sobre el FP32 + etiqueta `--variant` en el arnés (para no colisionar el checksum con el V0).
- Calibración sin fuga: conjunto separado del de evaluación; la evaluación oficial sigue siendo el V2 completo (10k). MobileNetV2: per-canal por defecto, QAT solo como último recurso (D9).
- Consecuencia: los `*_int8.onnx` (<100MB) se versionan en git; primera técnica del OE1 (orden D10). Detalle en `docs/DISENO_INT8_OE1.md`.

### D14 — INT8 en GPU: calibración nativa de TensorRT (plan B activado, jun 2026)
- Contexto: el gate de D13 (correr el QDQ de ONNX Runtime en el proveedor TensorRT) falló dos veces. Con `QuantizeBias=False` el modelo quedó limpio (verificado: 0 DequantizeLinear de bias en int32), pero TensorRT 10.3 igual rechaza la construcción del motor con "Error Code 4 ... node_Conv_753_bias_dq: input has type Int32" —un nodo que genera el propio parser de TensorRT al manejar el bias FP32 de una convolución INT8— y declina todo el grafo (cae a CUDA, sin INT8; p50 32.8 ms > FP32 6.59 ms).
- Decisión: descartar el QDQ unificado en GPU. INT8 definido por backend:
  - CPU (jetson-cpu, rpi-cpu): modelo QDQ (`*_int8.onnx`), corre en el proveedor CPU. Sin cambios; ya validado en ORT.
  - GPU (jetson-gpu): modelo FP32 + calibración nativa de TensorRT (tabla de calibración generada con el calibrador de ONNX Runtime sobre el mismo conjunto), vía el proveedor TensorRT con int8 habilitado. Es el camino documentado por ORT para GPU.
- Consecuencia (cambios en el arnés): (a) el backend ONNX Runtime acepta las opciones INT8 del proveedor TensorRT; (b) se añade `--variant` a run_benchmark/run_accuracy (metadatos + nombre de archivo), porque la GPU reutiliza el `.onnx` FP32 —mismo checksum que el V0— y hay que distinguir la corrida INT8; (c) script para generar la tabla de calibración de TensorRT. La variante INT8 se reporta por (variante, backend), calibrada sobre el mismo conjunto.

### D15 — Poda estructurada: parámetros y conjunto de recuperación (20 jun 2026)
- Decisión: poda estructurada de canales con torch-pruning (DepGraph), importancia L1, poda global iterativa hasta una fracción objetivo de MACs, sin tocar el clasificador (mantiene 1000 salidas). Un punto comprometido por modelo: ResNet-50 conservar ~50% de MACs (reducir 50%), MobileNetV2 ~70% (reducir 30%) —asimétrico porque MobileNetV2 es compacto y memory-bound, se degrada antes—. Un segundo punto más agresivo queda como extensión opcional.
- Reentrenamiento de recuperación: subconjunto balanceado de ImageNet-1k de entrenamiento (~100/clase, 100 203 img, 3.9 GB) en el Legion (AMP, SGD+coseno, label smoothing 0.1, 15 épocas, split 95/5 para validación, se exporta el mejor checkpoint). El índice de clase del subconjunto (carpetas 0000–0999, índice de Hugging Face) se verificó que COINCIDE con el orden de torchvision (ResNet-50 top-1 90.7%, MobileNetV2 77.5% sobre la muestra) → sin remapeo.
- Precisión del modelo podado: FP32, igual que la línea base V0, para aislar la variable poda (no introducir FP16 simultáneamente).
- Consecuencia: scripts `scripts/verify_class_mapping.py` y `scripts/prune_finetune.py` (poda + reentrenamiento + export a ONNX FP32 opset 18). El reentrenamiento corre en el Legion (no en la Jetson); el ONNX resultante se copia a la Jetson para medir EXP-07/08 y EXP-19/20. RPi (EXP-09/21) a Luis.

### D16 — Recuperación de la poda: sobre-ajuste detectado, eje de precisión a corregir (21 jun 2026)
- Contexto: ResNet-50 podado (−53% MACs) + reentrenamiento sobre el subconjunto de 100/clase recupera el val de entrenamiento (95.99%) pero NO generaliza: top-1 V2 0.510 (−18 pts vs V0 0.694). Es sobre-ajuste a las 100k, no el costo intrínseco de la poda.
- Implicación: el eje de precisión de la poda no es comparable con el del INT8 (PTQ sin reentrenar) hasta tener una recuperación adecuada. La latencia/energía/brecha SÍ son válidas: la poda ESTRECHA la brecha GPU-CPU 13.6→9.3× (contraste con el INT8, que la ensanchó a 14.6×).
- Plan (corrección pendiente de datos): (1) medir MobileNetV2 podado (−33%) como diagnóstico de poda suave sobre las mismas 100k; (2) re-entrenar ResNet a −30% MACs (artefacto aparte `resnet50_pruned_p30.onnx`); (3) si la precisión sigue baja, escalar la recuperación (más imágenes/clase o destilación). El checkpoint NO debe elegirse por el val de entrenamiento.
- Consecuencia: EXP-19/20 (y 07/08) quedan con latencia/energía medidas pero precisión PROVISIONAL hasta cerrar la recuperación; no marcar "Hecho" en el eje de precisión de la matriz hasta entonces.

- Actualización D16 (21 jun 2026, tras medir MobileNet): la poda SUAVE de MobileNet (−33%) también cayó −14 pts (V2 0.453), misma firma de sobre-ajuste (train 87% vs test 45%). Confirma que la causa es la RECUPERACIÓN, no la agresividad → se DESCARTA el paso de re-entrenar ResNet a −30%. El arreglo es mejorar la recuperación (más imágenes/clase o destilación) y/o reportar honestamente el costo de precisión bajo recuperación limitada (hallazgo: la poda exige reentrenamiento con datos suficientes, a diferencia del PTQ INT8). Decisión de alcance pendiente con el director. Latencia/energía/brecha firmes: la poda estrecha la brecha y, en la CPU memory-bound de MobileNet, acelera donde el INT8 no (1.43× vs 1.00×).

- Recuperación por destilación implementada (21 jun): `scripts/prune_distill.py` (maestro + KD, exporta la última época). ResNet KD (−53%) entrenado (sha efffe63b, val train 92.68% < 95.99% del fine-tuning normal). Precisión V2 en medición; latencia/energía se heredan de la poda normal (arquitectura idéntica). Gate: si el KD recupera claramente sobre 0.510 se sigue con MobileNet KD; si no, el límite es de datos y se reporta como hallazgo.

- Resultado KD ResNet (21 jun): la destilación recuperó parcialmente — V2 0.510 (fine-tuning normal) → 0.579 (KD), +6.8 pts; sigue −12 pts bajo 0.694. La destilación es la mejor recuperación con los mismos datos, pero el residual es límite de datos. Soporta el camino (1) de la nota al director: reportar como hallazgo cuantificado, con la destilación demostrando recuperabilidad. Falta MobileNet KD para completar el par.

- Resultado KD MobileNet + cierre (21 jun): MobileNet V2 0.453 (FT) → 0.508 (KD), +5.5 pts; residual −8.8 vs 0.596. Set completo de poda consolidado (ver BITACORA). Conclusión del eje de precisión: la destilación es la mejor recuperación con los datos disponibles (recupera ~5-7 pts), y el residual (~9-12 pts) es límite de datos → se reporta como hallazgo cuantificado (camino 1 de la nota al director).

### D17 — Versionar todos los ONNX en git vía Git LFS (21 jun 2026)
- Decisión: todos los modelos ONNX se versionan en el repo mediante Git LFS (`.gitattributes` enruta `*.onnx`), incluido el baseline ResNet-50 FP32 (~102 MB) que antes no cabía por el límite de 100 MB por archivo de GitHub. Se elimina el reparto por archivo/scp/checksum.
- Motivo: tener el conjunto completo y reproducible de artefactos (baselines, INT8, podados, KD) en un solo lugar versionado; los podados/KD son resultado de entrenamiento (no triviales de regenerar) y conviene fijarlos junto a su checksum.
- Consecuencia: requiere git-lfs instalado en cada máquina (Mac, Jetson, Legion, RPi) para clonar/actualizar y obtener los archivos reales: `git lfs install` (Linux `sudo apt install -y git-lfs`). Cuota gratuita de LFS 1 GB almacenamiento / 1 GB-mes transferencia (~300 MB de ONNX, holgado). Los ONNX ya commiteados como blobs normales (cnn_baseline, *_int8) se conservan; LFS aplica solo a lo nuevo (no se reescribe historia). `.gitignore` actualizado; temporales de cuantización siguen excluidos.
- Nota (5 jul 2026): en HEAD los 8 ONNX figuran como punteros LFS (incluidos cnn_baseline y *_int8), es decir, la migración terminó cubriéndolos a todos. Verificación de integridad: el sha256 en disco coincide con el oid del puntero en los 8.

### D18 — Precisión de la poda: camino 1 aprobado por el director, con condiciones (1 jul 2026)
- Contexto: respuesta del director (correo 1 jul 2026, "RE: Avance OE1") a la consulta del 21 jun sobre los dos caminos para la precisión de los modelos podados.
- Decisión: camino 1 aprobado — reportar el costo de recuperación de la poda como hallazgo legítimo frente al PTQ INT8 que no reentrena. El resultado de latencia/brecha (la poda estrecha la brecha GPU-CPU y acelera la CPU memory-bound de MobileNetV2) se reporta aparte y protegido: no depende de la precisión.
- Condición 1: la atribución de la caída al presupuesto de recuperación (y no a la poda en sí) queda CONDICIONAL hasta que la corrida de recuperación por destilación lo demuestre. Estado: la corrida KD ya estaba hecha al llegar la respuesta (D16); recuperó +6.8/+5.5 pts con residual −12/−9 → recuperación PARCIAL, no satisface el criterio de forma inequívoca. Se responde al director junto con el chequeo de la condición 2.
- Condición 2 (nueva tarea, antes de cerrar): evaluar los modelos podados+recuperados contra la validación completa de ImageNet (50k), no solo V2, para separar la brecha natural val→V2 del sobre-ajuste real. Aplica a V0 y pruned_kd de ambos modelos.
- Condición 2 CUMPLIDA (5 jul 2026, corridas `acc_legion-gpu_*` sobre ImageNet-val 50k): baselines val 0.802/0.720 (≈ referencia torchvision). La brecha natural val→V2 (~11-12 pts) se mantiene casi constante en los podados (amplificación solo 0.8-1.7 pts) → la caída es costo REAL de la recuperación, no artefacto de V2. Caída V0→KD sobre val: 9.8 pts (ResNet 0.802→0.704) y 8.0 (MobileNet 0.720→0.640); la KD recupera +7.1/+6.2 pts reales sobre val. Diagnóstico, no condición de matriz (device-tag legion-gpu; oficiales siguen siendo Jetson/V2). Correo enviado al director el 5 jul con cifras y pregunta explícita camino 1 vs 2; a la espera de su respuesta.
- Redacción exigida: presentar como "la poda estructurada a esta tasa exigió un presupuesto de recuperación no previsto" (no como veredicto general sobre la poda); etiquetar la corrida como "recuperación poda+destilación combinada" para no confundirla con la destilación como técnica 3.
- Alcance: NO invertir aún en la recuperación completa con más datos (eje secundario frente a la comparación de hardware). Si la corrida de destilación "sale rara", se reabre la conversación del camino 2.
- Consecuencia: el eje de precisión de la poda se REABRE como condicional (revierte el "cierre" registrado el 21-22 jun en matriz y Word, retos 4.x y secciones 7-8) hasta cumplir ambas condiciones; nueva corrida de precisión sobre ImageNet-val pendiente.
- CERRADO (jul 2026): tras el chequeo val, el director dio el tema por terminado ("el tema de la latencia lo podemos dejar allí", correo ~9 jul) y propuso publicar los resultados (D19) — cierre implícito pero inequívoco: no se propone publicar un resultado en disputa. El eje de precisión de la poda pasa de condicional a DEFINITIVO con la redacción acordada.

### D19 — Paper corto para ARTIIS 2026 (jul 2026)
- Decisión: escribir y enviar un artículo corto a ARTIIS 2026 (Lisboa, 11-13 nov; Springer CCIS, indexado Scopus) con fecha límite 15 de agosto de 2026 (ya es la extendida). Propuesto por el director (~9 jul); alcance propuesto por nosotros y aprobado por él (~10 jul).
- Alcance CONGELADO: solo lo ya medido y definitivo — comparación GPU vs CPU intra-Jetson, dos modelos (MobileNetV2/ResNet-50), dos técnicas (INT8/poda estructurada), tres métricas (latencia/energía/precisión); hallazgo central: INT8 y poda mueven la brecha en sentidos opuestos (tamaños de efecto + IC); costo de recuperación de la poda cuantificado; arnés público como soporte de reproducibilidad. Nada del paper depende de rpi-cpu (si llega a tiempo, entra como referencia de despliegue) ni de la destilación.
- Autoría: Orlando y Luis autores principales (investigación y redacción); director coautor como asesor académico.
- Consecuencias: (a) técnica 3 (destilación) POSPUESTA hasta después del 15 ago; (b) el paper se escribe en inglés (CCIS); (c) revisión double-blind → la versión de envío debe ANONIMIZARSE, incluido el enlace al repo (ODuqueDvise delata autor); el repo se cita en la versión final; (d) participación virtual por Zoom (presentación oral en el evento, 11-13 nov); (e) financiación: la universidad cubre COP 600.000 por ponencia con publicación ISSN/ISBN (CCIS cumple) — VERIFICAR la tarifa real de inscripción de ARTIIS contra ese tope; (f) fechas: notificación 10 sep, versión final e inscripción 30 sep.
- Primer entregable: esquema del artículo (secciones y contenido) para visto bueno del director, semana del 6-12 jul.
