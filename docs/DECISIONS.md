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

### D9 — Alcance de modelos: 1 comprometido + 1 extensión (EN CONSULTA, jun 2026)
- Propuesta: comprometer MobileNetV2; recuperar un segundo modelo más pesado (ResNet-50/EfficientNet) como extensión condicionada al tiempo.
- Motivo: un solo modelo debilita la validez externa; MobileNetV2 ya optimizado puede subestimar el efecto de las técnicas.
- Estado: pendiente de confirmación del director (ver `Concepto_Alcance_Modelos.md`).
