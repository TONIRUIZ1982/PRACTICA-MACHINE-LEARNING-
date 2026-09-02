# Modelo base de cancelación

## Objetivo

Disponer de una primera referencia reproducible antes de probar modelos más complejos. Se utiliza regresión logística porque es rápida, interpretable y permite comprobar que todo el flujo funciona de principio a fin.

## Decisiones aplicadas

- Predicción en el momento de realizar la reserva.
- Separación temporal: entrenamiento hasta septiembre de 2016, validación entre octubre y diciembre de 2016 y prueba final durante 2017.
- `reservation_status` y `reservation_status_date` se excluyen siempre por fuga de información.
- `assigned_room_type`, `booking_changes` y `days_in_waiting_list` se excluyen porque pueden cambiar después de la reserva.
- `company` se excluye por su elevada proporción de valores ausentes.
- Los valores ausentes numéricos se sustituyen por cero y los categóricos por `Unknown`.
- Las categorías se codifican con one-hot y los valores desconocidos en validación no producen errores.
- Las variables numéricas se estandarizan.
- `agent` se excluye: tiene muchas categorías, bastantes ausentes y no mejoró la validación temporal.
- El día del mes se excluye porque no aportó mejora incremental.
- Se evita duplicar noches, huéspedes e historial mediante totales o indicadores que repiten la información original.
- Se añade `has_children` y se representa la semana de llegada de forma cíclica para conservar la estacionalidad.

La selección definitiva reduce la entrada de 30 a 23 variables. En la comparación local mejoró ligeramente tanto F1 como ROC-AUC respecto al baseline inicial. El detalle está en `docs/SELECCION_VARIABLES.md`.

La imputación, codificación y estandarización se ajustan exclusivamente con el conjunto de entrenamiento para evitar contaminación de validación.

## Métricas

La métrica principal es **F1**, ya que combina precisión y recall y la cancelación no está perfectamente equilibrada. También se muestran accuracy, precisión, recall y ROC-AUC para ofrecer una visión completa.

## Ejecución local

Desde la raíz del repositorio:

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.train_logistic_baseline --deduplicate
```

El modelo, las métricas y los gráficos se guardan en `artifacts/logistic_baseline/`. Esta carpeta está ignorada por Git porque contiene resultados derivados del dataset.

El conjunto de prueba de 2017 no se evalúa por defecto. Solo cuando el equipo haya cerrado las decisiones del modelado se podrá ejecutar expresamente:

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.train_logistic_baseline --deduplicate --evaluate-test
```

## Uso desde un notebook

Los notebooks deben importar `prepare_model_frame`, `build_logistic_pipeline` y `evaluate_classifier` en lugar de repetir el preprocesamiento. De este modo todos los integrantes trabajan con las mismas reglas y los modelos se pueden comparar de forma justa.
