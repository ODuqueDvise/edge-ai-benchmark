# Procedimiento de registro (estados y decisiones)

Objetivo: que el estado del proyecto y las decisiones queden registrados de forma
sistemática y de bajo costo, no de memoria ni a última hora. Es la traza del
proyecto (insumo del OE3 y del capítulo de metodología).

Tres archivos, cada uno con su propósito:

- `docs/DECISIONS.md` — **registro de decisiones** (qué se decidió, por qué, qué se descartó). Una entrada por decisión que cambie alcance, metodología o protocolo.
- `docs/BITACORA.md` — **bitácora cronológica** de estado: avances, mediciones, bloqueos. Entradas cortas.
- `results/RESULTS_LOG.md` — **datos** (generado automáticamente por `build_results_log.py`; no se edita a mano).

> Regla de oro: si una decisión no está en `DECISIONS.md`, no es oficial. Si un
> avance no está en `BITACORA.md`, no pasó.

## Para retomar el proyecto (recuperar contexto)

Antes de trabajar —tú, Luis o Cowork/Claude— **lee primero**, en este orden:
`DECISIONS.md` (qué se decidió y por qué), `BITACORA.md` (qué pasó y dónde quedó) y
`results/RESULTS_LOG.md` (los datos). Eso reconstruye el estado sin depender de la
memoria de nadie. Solo después, actúa.

## Puntos de chequeo (cuándo registrar)

**CP1 — Tras cada campaña de medición.**
Ya subes datos con `bash scripts/sync_results.sh`. Añade además una línea en
`BITACORA.md`: qué se midió, en qué condición y el hallazgo en una frase. Si la
medición motivó una decisión (cambiar un parámetro, descartar algo), va a `DECISIONS.md`.

**CP2 — Tras cada reunión o correo con el director.**
Registra en `DECISIONS.md` lo que él confirmó o pidió cambiar (alcance, técnicas,
método estadístico), con fecha. Es lo que evita "¿el profe había dicho que sí a esto?".

**CP3 — Al cerrar un hito (H1–H9 del plan).**
Entrada en `BITACORA.md` marcando el hito y el estado; verifica que `RESULTS_LOG.md`
y las guías reflejen el avance.

**CP4 — Ante cualquier cambio de alcance, protocolo o metodología.**
Registro en `DECISIONS.md` **antes** de ejecutar el cambio (no después). Si tocas las
constantes congeladas del protocolo, esto es obligatorio.

**CP5 — Cierre semanal (sugerido: viernes).**
Entrada corta en `BITACORA.md`: estado, qué sigue, bloqueos. Sincroniza con Luis.

## Plantillas

**Decisión (`DECISIONS.md`):**
```
### Dxx — <título corto> (<fecha>)
- Decisión: <qué se decidió>
- Motivo: <por qué>
- Alternativas / descartado: <qué no se eligió>
- Consecuencia: <qué implica / cómo se aplica>
```

**Bitácora (`BITACORA.md`):**
```
## <fecha> — <equipo/responsable>
- <avance / medición / bloqueo, en 1–2 líneas>
```

## Apoyo

Puedes pedirle a Cowork que redacte la entrada (de decisión o bitácora) a partir de
lo ocurrido en la sesión; revísala y la confirmas. Y subes todo con el flujo de git
habitual (o `sync_results.sh` para lo de `results/`).
