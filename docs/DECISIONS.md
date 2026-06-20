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
