# Evidencias para la evaluación

Este documento facilita la revisión de los requisitos del proyecto y localiza la evidencia reproducible de cada uno.

## 1. Implementación y comparación de modelos

Se implementan los cinco modelos obligatorios con las mismas variables, partición temporal y métricas:

- Regresión logística.
- Árbol de decisión.
- Random Forest.
- XGBoost como Gradient Boosting.
- Red neuronal multicapa con Keras/TensorFlow.

Evidencia: `src/hotel_cancellation/model_suite.py`, `scripts/train_all_models.py`, `docs/MODELOS_PREDICCION.md` y `tests/test_model_suite.py`.

La selección final usa F1 de validación temporal y el modelo elegido es XGBoost. Los resultados finales se encuentran en `docs/RESULTADO_FINAL.md`.

## 2. Evaluación técnica y visualización

F1 se justifica como métrica principal porque equilibra la detección de cancelaciones y las falsas alarmas. También se calculan accuracy, precision, recall y ROC-AUC.

El flujo genera para cada candidato matriz de confusión, curva ROC y métricas comunes. También genera una curva ROC comparativa y la importancia de variables de Random Forest.

Evidencia: `src/hotel_cancellation/comparison.py`, `src/hotel_cancellation/optimization.py`, `scripts/optimize_models.py` y `docs/RESULTADO_FINAL.md`.

Las figuras se generan localmente en `artifacts/model_optimization/` y se incorporan al informe PDF de entrega sin publicar el dataset ni artefactos entrenados.

## 3. Automatización y estructura

El proyecto separa datos, variables, preprocesamiento, modelos, optimización e inferencia en módulos reutilizables. El punto de entrada `scripts/run_pipeline.py` agrupa validación, EDA, comparación, optimización, comprobación e inferencia.

Evidencia: `src/hotel_cancellation/`, `scripts/run_pipeline.py`, `scripts/check_delivery.py` y `docs/CIERRE_TECNICO.md`.

La evaluación de 2017 queda protegida: el script rechaza reutilizarla tras guardar una evaluación final.

## 4. Calidad y documentación

El repositorio incluye README, dependencias fijadas, `.gitignore`, documentación metodológica, docstrings y pruebas automatizadas.

Evidencia: `README.md`, `requirements.txt`, `.gitignore`, `docs/` y `tests/`.

La comprobación de entrega verifica la presencia de documentación, la disponibilidad local del CSV y modelo, y que Git no rastrea datos, modelos o resultados locales.

## 5. Bonus técnico

El proyecto incorpora dos elementos adicionales:

1. Optimización de hiperparámetros con `RandomizedSearchCV` y validación cruzada temporal.
2. Interpretabilidad mediante importancia de variables de Random Forest, generada como figura y tabla local reproducible.

Evidencia: `src/hotel_cancellation/optimization.py`, `docs/OPTIMIZACION_MODELOS.md` y `artifacts/model_optimization/random_forest_feature_importance.png` tras la ejecución local.
