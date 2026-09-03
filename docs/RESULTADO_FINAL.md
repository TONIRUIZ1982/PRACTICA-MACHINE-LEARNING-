# Resultado final del sistema de predicción

## Proceso de selección

Se mantuvo el periodo de 2017 fuera de todas las decisiones. Los modelos se ajustaron con datos hasta septiembre de 2016 y se compararon en la validación de octubre a diciembre de 2016.

La optimización utilizó validación cruzada temporal para los modelos clásicos, tres arquitecturas de red neuronal y ajuste del umbral de clasificación para maximizar F1. La selección se realizó exclusivamente por F1 de validación.

## Comparación optimizada en validación

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Umbral |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0,7707 | 0,5808 | 0,6691 | **0,6218** | 0,8277 | 0,3619 |
| Random Forest | 0,7590 | 0,5610 | 0,6658 | 0,6089 | **0,8283** | 0,4854 |
| Red neuronal pequeña | 0,7656 | 0,5760 | 0,6380 | 0,6054 | 0,8199 | 0,2092 |
| Regresión logística | 0,7311 | 0,5166 | **0,7102** | 0,5981 | 0,8004 | 0,3959 |
| Red neuronal ancha | 0,7277 | 0,5121 | 0,7158 | 0,5970 | 0,8145 | 0,1019 |
| Red neuronal profunda | 0,7459 | 0,5399 | 0,6651 | 0,5960 | 0,8132 | 0,1498 |
| Árbol de decisión | 0,7348 | 0,5229 | 0,6741 | 0,5889 | 0,7671 | 0,3850 |

## Modelo seleccionado

Se selecciona **XGBoost** porque logra el mayor F1 en validación (0,6218). Su umbral de clasificación queda fijado en **0,3619**; se eligió en validación y no se modificó posteriormente.

## Evaluación final - 2017

El ganador se evaluó una única vez con reservas de llegada en 2017:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,6866 |
| Precision | 0,5055 |
| Recall | 0,8289 |
| F1 | **0,6280** |
| ROC-AUC | 0,8098 |
| Umbral | 0,3619 |

El modelo mantiene y mejora ligeramente F1 en el periodo futuro. El recall elevado indica que identifica la mayoría de cancelaciones, a cambio de generar algunas falsas alarmas; es un intercambio razonable para una política de prevención de cancelaciones.

## Regla de cierre

El conjunto de prueba de 2017 ya ha sido utilizado y queda cerrado. Cualquier mejora posterior debe documentarse como un nuevo experimento y no debe utilizar estos resultados para volver a seleccionar hiperparámetros, variables o umbrales.
