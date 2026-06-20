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
