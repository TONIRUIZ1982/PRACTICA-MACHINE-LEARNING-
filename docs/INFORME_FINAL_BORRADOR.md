# Informe final - Predicción de cancelaciones hoteleras

## 1. Problema y objetivo

El objetivo es estimar, justo después de confirmar una reserva, la probabilidad de que sea cancelada. Esta predicción permite anticipar la ocupación y planificar acciones preventivas.

La variable objetivo es `is_canceled`: `1` representa una reserva cancelada y `0` una reserva no cancelada.

## 2. Datos y análisis exploratorio

El dataset contiene 119.390 registros y 32 columnas. Se identificaron 31.994 duplicados exactos, por lo que el escenario principal utiliza una deduplicación reproducible. Los valores ausentes se concentran en `company`, `agent`, `country` y `children`.

El EDA reproducible se ejecuta con `scripts/run_eda.py` y guarda sus salidas solamente en `artifacts/eda/`.

## 3. Diseño del sistema

El proyecto está organizado en módulos reutilizables para carga y validación, selección de variables, preprocesamiento, entrenamiento, optimización, evaluación e inferencia.

La separación temporal simula el uso real del sistema:

| Conjunto | Periodo de llegada |
|---|---|
| Entrenamiento | Hasta septiembre de 2016 |
| Validación | Octubre-diciembre de 2016 |
| Test final | Año 2017 |

La imputación, codificación y escalado se ajustan solo con datos de entrenamiento.

## 4. Variables

Se eliminan las fugas directas `reservation_status` y `reservation_status_date`. También se excluyen variables potencialmente posteriores a la reserva (`assigned_room_type`, `booking_changes`, `days_in_waiting_list`) y `company` por su elevada ausencia.

El proceso de selección redujo los predictores de 30 a 23, eliminando `agent`, el día del mes y variables derivadas redundantes. Se conserva la estacionalidad mediante representaciones cíclicas de la semana de llegada.

## 5. Modelos comparados

Se implementaron regresión logística, árbol de decisión, Random Forest, XGBoost y una red neuronal multicapa con Keras. Todos comparten variables, periodos y métricas.

F1 se selecciona como métrica principal porque equilibra la detección de cancelaciones y el control de falsas alarmas. También se muestran accuracy, precision, recall y ROC-AUC.

## 6. Optimización y elección final

Se aplicó búsqueda aleatoria con validación cruzada temporal a los modelos clásicos, se compararon tres arquitecturas Keras y se seleccionó el umbral que maximiza F1 en validación.

XGBoost fue el ganador en validación con F1 de 0,6218 y ROC-AUC de 0,8277. El umbral seleccionado fue 0,3619.

## 7. Resultado final

El modelo XGBoost se evaluó una única vez en el test de 2017:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,6866 |
| Precision | 0,5055 |
| Recall | 0,8289 |
| F1 | 0,6280 |
| ROC-AUC | 0,8098 |

El modelo mantiene el resultado en un periodo futuro y detecta la mayoría de las cancelaciones. El test queda cerrado y no se usa para nuevos ajustes.

## 8. Inferencia

El script `scripts/predict_cancellations.py` aplica el pipeline XGBoost a un CSV de reservas nuevas y devuelve una probabilidad de cancelación junto con la decisión basada en el umbral seleccionado.

## 9. Limitaciones y mejoras

- La tasa de cancelación cambia entre periodos, por lo que se recomienda vigilar la deriva temporal.
- El umbral debe adaptarse al coste de una falsa alarma en la operativa del hotel.
- Podrían evaluarse explicaciones SHAP, una API FastAPI o un registro de experimentos como extensiones.

## 10. Trabajo compartido

El equipo ha trabajado con responsabilidad compartida, revisando conjuntamente las decisiones sobre variables, datos, modelos y documentación. El historial de commits permite identificar las aportaciones técnicas realizadas durante el proyecto.

> Pendiente antes de entrega: completar en equipo nombres, URL final del repositorio y el apartado de contribuciones acordado para la evaluación.
