# Cierre tecnico y comprobacion de entrega

Este documento permite comprobar que el proyecto se puede revisar sin publicar el
dataset original, modelos entrenados ni resultados locales.

## Antes de compartir o entregar

Desde la raiz del proyecto, con el entorno virtual activado, ejecutar:

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.run_pipeline check -- --require-data --require-model
```

La comprobacion valida lo siguiente:

1. Existen los documentos de variables, decisiones, resultados, informe,
   defensa y pendientes.
2. Git no tiene versionados los directorios locales `data/raw/`, `artifacts/`
   ni `models/`.
3. El CSV local conserva la estructura esperada.
4. El modelo final XGBoost existe localmente y se puede cargar.

No reentrena modelos ni vuelve a evaluar 2017: la prueba final queda preservada.

## Secuencia reproducible

Los comandos se ejecutan desde la raiz y todos usan el mismo entorno:

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.run_pipeline validate -- --deduplicate
& ".\.venv\Scripts\python.exe" -m scripts.run_pipeline eda -- --deduplicate
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

Las fases de comparacion u optimizacion no se vuelven a lanzar para una revision
ordinaria, ya que pueden tardar y la evaluacion final sobre 2017 ya esta cerrada.

## Decisiones que quedan en manos del equipo

- Validar el modelo elegido frente a las propuestas de los otros integrantes.
- Elegir las figuras de resultados locales que iran al informe.
- Completar autores, repositorio final y aportaciones compartidas.
- Revisar los notebooks que finalmente se integren antes de hacer merge en
  `main`.
