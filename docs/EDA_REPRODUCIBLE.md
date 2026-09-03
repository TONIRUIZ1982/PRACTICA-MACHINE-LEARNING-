# EDA reproducible

El script `scripts/run_eda.py` permite regenerar las comprobaciones esenciales del análisis exploratorio sin duplicar reglas de limpieza en un notebook ni publicar resultados derivados del dataset.

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.run_eda --deduplicate
```

Genera localmente en `artifacts/eda/`:

- un resumen agregado de filas, duplicados, ausentes y tasa de cancelación;
- distribución del objetivo;
- gráfico de valores ausentes.

Los notebooks existentes pueden usar este script como referencia para conservar el análisis visual detallado. Antes de integrar la rama EDA de otro integrante, debe revisarse que no exporte el CSV procesado, no impute todo el dataset antes de separar los periodos y no conserve resultados derivados en Git.
