# Selección de variables del modelo base

## Objetivo

Reducir complejidad y redundancias sin utilizar el conjunto de prueba de 2017. Todas las comparaciones se realizaron con el mismo entrenamiento temporal y el periodo de validación de octubre a diciembre de 2016.

## Variables eliminadas

| Variable | Motivo |
|---|---|
| `agent` | 253 valores distintos y 14,64 % de ausentes en entrenamiento. Eliminarla mantuvo F1 y mejoró ligeramente ROC-AUC. Además evita depender de códigos que pueden cambiar. |
| `arrival_date_day_of_month` | No mostró mejora incremental y su interpretación lineal no representa bien un ciclo mensual. |
| `total_nights` | Es la suma exacta de las noches de semana y fin de semana, que se conservan por aportar más detalle. |
| `total_guests` | Es la suma exacta de adultos, niños y bebés, que se conservan. |
| `has_previous_bookings` | Repite de forma binaria la información de cancelaciones y reservas anteriores; eliminarla mejoró la validación. |
| `arrival_month_sin` y `arrival_month_cos` | Duplicaban la estacionalidad ya representada con la semana cíclica, que ofrece mayor resolución. |

Las exclusiones previas se mantienen: `reservation_status`, `reservation_status_date`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list` y `company`. `arrival_date_year` continúa utilizándose para separar los periodos, no como predictor.

## Variables conservadas deliberadamente

- `country`: aunque tiene numerosas categorías, eliminarla redujo claramente F1 y ROC-AUC.
- Noches de semana y fin de semana: conservan el patrón de estancia, no solo la duración total.
- Adultos, niños y bebés: mantienen la composición concreta del grupo.
- `has_children`: añade una relación no lineal sencilla y fácil de interpretar.
- Semana cíclica: conserva la estacionalidad sin tratar la semana 1 y la 53 como valores lejanos.

## Resultado de la comparación

| Configuración | Variables | F1 validación | ROC-AUC validación |
|---|---:|---:|---:|
| Baseline inicial | 30 | 0,3913 | 0,7968 |
| Selección final | 23 | 0,4019 | 0,8010 |

La mejora es moderada, pero el modelo queda más sencillo, estable e interpretable. El conjunto de prueba de 2017 no se utilizó para tomar estas decisiones.
