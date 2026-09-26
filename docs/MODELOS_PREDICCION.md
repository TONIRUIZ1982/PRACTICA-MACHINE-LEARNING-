# Comparación de modelos de predicción

## Alcance

El flujo implementa los cinco algoritmos obligatorios del enunciado:

1. Regresión logística.
2. Árbol de decisión.
3. Random Forest.
4. XGBoost como algoritmo de Gradient Boosting.
5. Red neuronal multicapa con Keras de TensorFlow.

Todos reciben las mismas 23 variables, el mismo entrenamiento temporal y el mismo periodo de validación. La imputación, el escalado y la codificación se ajustan solo con entrenamiento.

## Evaluación común

F1 es la métrica principal porque interesa encontrar cancelaciones sin ignorar el coste de generar falsas alarmas. También se calculan accuracy, precision, recall y ROC-AUC.

El script genera localmente:

- tabla comparativa CSV y JSON;
- matriz de confusión y curva ROC de cada modelo;
- curva ROC conjunta;
- importancia de variables del Random Forest;
- modelos entrenados en formato `joblib` o `.keras`.

Todo se guarda bajo `artifacts/model_comparison/`, carpeta ignorada por Git.

## Ejecución

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.train_all_models --deduplicate
```

Para mostrar el progreso de Keras:

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.train_all_models --deduplicate --neural-verbose 2
```

Durante el desarrollo, `--models` permite ejecutar un subconjunto sin alterar el flujo:

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.train_all_models --deduplicate --models decision_tree random_forest
```

## Protección del test final

El periodo de 2017 no se evalúa por defecto. Cuando el equipo haya cerrado modelos, variables, hiperparámetros y umbral, se podrá añadir `--evaluate-test`. Esa opción evalúa únicamente el ganador determinado previamente por F1 de validación.

## Configuración inicial

Los hiperparámetros actuales son puntos de partida reproducibles, no una optimización final. El ajuste se realizará después de verificar los cinco modelos y solo con entrenamiento/validación.
