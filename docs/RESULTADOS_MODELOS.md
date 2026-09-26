# Resultados iniciales de los modelos

## Condiciones de la comparación

Los cinco modelos se entrenaron con las mismas 23 variables y el conjunto deduplicado. Se utilizó entrenamiento hasta septiembre de 2016 y validación entre octubre y diciembre de 2016. El periodo de 2017 permanece sin evaluar.

Los hiperparámetros son configuraciones iniciales reproducibles y el umbral de clasificación es 0,5. Por tanto, esta tabla sirve como línea base común, no como resultado final optimizado.

## Resultados de validación

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Árbol de decisión | 0,7766 | 0,7177 | 0,3418 | **0,4631** | 0,7896 |
| Red neuronal Keras | **0,7810** | 0,7811 | 0,3097 | 0,4435 | 0,8147 |
| XGBoost | 0,7759 | 0,8035 | 0,2710 | 0,4053 | 0,8272 |
| Regresión logística | 0,7724 | 0,7748 | 0,2713 | 0,4019 | 0,8010 |
| Random Forest | 0,7528 | **0,8560** | 0,1477 | 0,2520 | **0,8275** |

## Lectura técnica

- El árbol de decisión es el ganador provisional según F1, la métrica principal acordada.
- La red neuronal consigue la mayor accuracy y el segundo mejor F1.
- Random Forest y XGBoost presentan la mejor capacidad de ordenación según ROC-AUC.
- Random Forest es muy preciso cuando predice una cancelación, pero con umbral 0,5 deja demasiadas cancelaciones sin detectar. Su F1 bajo no implica que deba descartarse antes de revisar el umbral y sus hiperparámetros.
- Ningún modelo debe declararse ganador final hasta realizar el ajuste exclusivamente con entrenamiento y validación.

## Siguiente decisión

El siguiente paso recomendado es ajustar hiperparámetros y estudiar el umbral de clasificación de los candidatos con mayor potencial. El modelo definitivo se elegirá por F1 de validación y solo entonces se evaluará una vez con 2017.
