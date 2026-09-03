# Optimización y selección final

## Metodología

Los hiperparámetros de regresión logística, árbol, Random Forest y XGBoost se exploran mediante `RandomizedSearchCV`. Los pliegues de `TimeSeriesSplit` respetan el orden temporal dentro del conjunto de entrenamiento.

La red neuronal compara tres arquitecturas acotadas. Cada una utiliza parada temprana con una partición estratificada creada únicamente a partir del entrenamiento.

Después del ajuste se buscan los umbrales que maximizan F1 en la validación temporal. El ganador se determina exclusivamente con esa métrica. El test de 2017 no participa en ninguna decisión.

## Ejecución de la optimización

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.optimize_models --deduplicate
```

La intensidad se puede controlar con `--search-iterations`, `--cv-splits`, `--search-jobs` y `--neural-epochs`.

## Evaluación final

Solo cuando no se vayan a realizar más cambios se añade:

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.optimize_models --deduplicate --evaluate-test
```

La opción evalúa únicamente el ganador de validación con su umbral ya fijado. Después de ejecutar esta orden, 2017 se considera consumido y sus resultados no deben utilizarse para modificar el modelo.

## Artefactos

Los parámetros, métricas, curvas, matrices y modelos se guardan en `artifacts/model_optimization/`. Esta carpeta está ignorada por Git y no publica información derivada del dataset.
