# Análisis y comprensión del dataset

** Análisis y comprensión del dataset

**** Evaluación Final – Machine Learning**

*Proyecto: Predicción de cancelación de reservas hoteleras*

## 1. Contexto del problema

El dataset proporcionado para la evaluación contiene información sobre reservas de hoteles realizadas a lo largo del tiempo, incluyendo características del cliente, comportamiento de reserva y variables relacionadas con la posible cancelación. El objetivo de la práctica es utilizar técnicas de Machine Learning para construir un sistema de clasificación binaria que permita predecir si una reserva será cancelada o no.

## 2. Variable objetivo (target)

La variable objetivo será «is_canceled». Según la documentación del dataset, toma el valor 1 cuando la reserva fue cancelada y 0 cuando no fue cancelada. Por tanto, estamos ante un problema de clasificación binaria.

| Variable | Tipo | Interpretación |
| --- | --- | --- |
| is_canceled | Binaria | 1 = cancelada |
| is_canceled | Binaria | 0 = no cancelada |

## 3. Variables disponibles

El dataset incluye variables sobre el hotel, fechas de llegada, duración de la estancia, composición del grupo, alimentación, país, canales de marketing y distribución, historial del cliente, características de la habitación, depósito, agente/empresa, precio y solicitudes especiales, además del estado final de la reserva.

| Variable(s) | Tipo orientativo | Descripción |
| --- | --- | --- |
| hotel | Categórica | Tipo de hotel |
| lead_time | Numérica | Días entre la reserva y la llegada |
| arrival_date_year / month / week / day | Temporal/categórica | Fecha de llegada |
| stays_in_weekend_nights / stays_in_week_nights | Numérica | Duración de la estancia |
| adults / children / babies | Numérica | Composición del grupo |
| meal | Categórica | Tipo de comida |
| country | Categórica | País de origen |
| market_segment | Categórica | Segmento/canal de marketing |
| distribution_channel | Categórica | Canal de distribución |
| is_repeated_guest | Binaria | Cliente repetidor |
| previous_cancellations | Numérica | Cancelaciones anteriores |
| previous_bookings_not_canceled | Numérica | Reservas previas no canceladas |
| reserved_room_type / assigned_room_type | Categórica | Habitación reservada/asignada |
| booking_changes | Numérica | Cambios de reserva |
| deposit_type | Categórica | Tipo de depósito |
| agent / company | Categórica/ID | Agente o empresa; pueden ser nulos |
| days_in_waiting_list | Numérica | Días en lista de espera |
| customer_type | Categórica | Tipo de cliente |
| adr | Numérica | Precio medio diario |
| required_car_parking_spaces | Numérica | Plazas de parking solicitadas |
| total_of_special_requests | Numérica | Peticiones especiales |
| reservation_status | Categórica | Estado final de la reserva |
| reservation_status_date | Temporal | Fecha de actualización del estado |

## 4. Decisiones de Machine Learning que debemos justificar

**Definición del target: **Usaremos is_canceled como variable objetivo porque el enunciado la define como la variable binaria que representa la cancelación.

**Variables predictoras: **Debemos decidir qué variables utilizar como entradas del modelo después de revisar calidad de datos, valores nulos, cardinalidad, relevancia y posibles problemas de fuga de información.

**Tratamiento de variables categóricas: **Será necesario transformar las variables categóricas a una representación que puedan utilizar los modelos, evaluando una estrategia adecuada para cada caso.

**Valores nulos: **Hay que identificar qué columnas contienen valores ausentes y decidir, según su significado y distribución, si se imputan, se agrupan o se eliminan.

**Variables de fecha: **Debemos decidir cómo representar las fechas. El análisis debe comprobar si conviene conservar sus componentes, transformarlos o derivar nuevas variables.

**Variables de identificación: **Agent y company son identificadores y pueden contener muchos valores nulos. Antes de utilizarlos debemos comprobar su utilidad predictiva y evitar tratarlos como variables numéricas ordinarias.

**Posible data leakage: **Hay que revisar especialmente variables que describen el estado final de la reserva. Si una variable solo se conoce después de que la cancelación haya ocurrido, no debe utilizarse para predecir una cancelación futura.

**Separación train/test: **Debemos separar los datos en entrenamiento y prueba antes de ajustar transformaciones, para evitar que información del conjunto de prueba influya en el entrenamiento.

**Escalado: **La necesidad de escalar dependerá del algoritmo. Será especialmente relevante para modelos sensibles a la escala, como la regresión logística y la red neuronal.

**Métrica principal: **El enunciado permite accuracy, precision, recall, F1-score o AUC-ROC. La elección final debe justificarse después de analizar la distribución de clases y el objetivo del negocio.

**Comparación de modelos: **Los cinco modelos obligatorios deben evaluarse con métricas comunes para poder compararlos de forma coherente.

**Selección del modelo final: **El mejor modelo se seleccionará según la métrica principal definida, considerando también las métricas secundarias y el comportamiento observado en las matrices de confusión y curvas ROC.

## 5. Punto crítico: posible fuga de información

La variable «reservation_status» merece una revisión especial. El dataset la describe como el estado final de la reserva (por ejemplo, Check-Out, Canceled o No-Show), y «reservation_status_date» como la fecha en que se actualizó dicho estado. Estas variables están estrechamente relacionadas con el resultado que queremos predecir y podrían contener información disponible únicamente después de que la cancelación haya ocurrido.

Por ello, antes del entrenamiento debemos comprobar el momento en que estaría disponible cada variable. Si el objetivo es predecir una cancelación en el momento de la reserva, no deberíamos utilizar variables que revelen el estado final de esa reserva. Esta decisión debe quedar explicada en el informe.

## 6. Qué debemos comprobar en el EDA

Número de filas y columnas.

Tipos de datos de todas las variables.

Valores nulos por columna.

Distribución de is_canceled y posible desequilibrio de clases.

Distribución de variables numéricas y detección de valores extremos.

Cardinalidad de variables categóricas.

Relación entre variables relevantes y la cancelación.

Posibles duplicados.

Variables altamente relacionadas o redundantes.

Variables que puedan producir data leakage.

##7. Modelos obligatorios

| Modelo | Obligatorio | Motivo de comparación |
| --- | --- | --- |
| Regresión logística | Sí | Modelo de referencia y clasificación lineal. |
| Árbol de decisión | Sí | Modelo basado en reglas y particiones. |
| Random Forest | Sí | Ensemble de árboles, útil para relaciones no lineales. |
| Gradient Boosting | Sí | Modelo de boosting; puede implementarse con XGBoost, LightGBM o CatBoost. |
| Red neuronal multicapa (Keras/TensorFlow) | Sí | Modelo no lineal basado en redes neuronales. |


Métrica principal: se decidirá después de analizar el balance de clases y el objetivo de predicción.

Métricas secundarias: Accuracy, Precision, Recall, F1-score y ROC-AUC, cuando sean apropiadas.

Matriz de confusión para analizar falsos positivos y falsos negativos.

Curvas ROC comparativas.


##9. Conclusión de esta fase

El dataset permite plantear un problema de clasificación binaria cuyo objetivo es predecir la cancelación de una reserva hotelera. La variable objetivo es is_canceled. La principal decisión metodológica que debemos investigar antes de entrenar los modelos será la preparación de las variables y, especialmente, la posible fuga de información de las variables que reflejan el estado final de la reserva. El EDA será fundamental para justificar estas decisiones.

