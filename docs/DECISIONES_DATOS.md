# Propuesta de decisiones sobre los datos

> Estado: propuesta técnica para revisión y aprobación de los tres integrantes. No representa todavía una decisión definitiva del equipo.

## Resumen ejecutivo

| Tema | Propuesta inicial | Motivo principal |
|---|---|---|
| Momento de predicción | Justo después de confirmar la reserva | Permite actuar antes de la llegada y limita el uso a información disponible. |
| Duplicados | Modelo principal con registros exactos deduplicados y comparación de sensibilidad con el original | Los duplicados cambian materialmente la distribución del objetivo. |
| Ausentes | Imputación explícita según significado; excluir `company` inicialmente | Evita confundir ausencia con una categoría frecuente. |
| Variables | Usar información conocida al confirmar; excluir fugas y variables posteriores | Evita métricas artificialmente elevadas. |
| Partición | Separación temporal con entrenamiento, validación y prueba futuras | Simula mejor el uso real del modelo. |

## 1. Momento de predicción

### Propuesta

Predecir la probabilidad de cancelación **justo después de que la reserva quede confirmada y antes de que se produzcan cambios posteriores**.

### Justificación

- Es un momento operativo claro y reproducible.
- El hotel todavía dispone de tiempo para planificar ocupación y acciones preventivas.
- Permite utilizar datos de la reserva, del cliente y de su historial conocidos en ese instante.
- Impide utilizar información generada después de la confirmación.

### Consecuencia

Se excluyen del primer modelo:

- `reservation_status` y `reservation_status_date`, por revelar el resultado final.
- `assigned_room_type`, `booking_changes` y `days_in_waiting_list`, porque pueden actualizarse después de confirmar la reserva.

## 2. Registros duplicados

### Evidencia observada

- Filas originales: **119.390**.
- Duplicados exactos sobrantes: **31.994**.
- Filas pertenecientes a grupos repetidos: **40.165**.
- Grupos distintos con repeticiones: **8.171**.
- Mayor grupo de filas idénticas: **180 registros**.
- Filas después de `drop_duplicates()`: **87.396**.
- Tasa de cancelación original: **37,04 %**.
- Tasa después de deduplicar: **27,49 %**.

El dataset no contiene un identificador único de reserva. Por ello no puede demostrarse solo con estas columnas si todas las repeticiones son errores técnicos o si algunas representan reservas distintas con características idénticas.

### Propuesta

1. Conservar siempre intacto el CSV original.
2. Utilizar el conjunto deduplicado como escenario principal de modelado para evitar que ciertas combinaciones queden sobrerrepresentadas.
3. Repetir la comparación final de modelos con el conjunto original como análisis de sensibilidad.
4. Documentar la diferencia de resultados entre ambos escenarios.

No se eliminarán filas por coincidencias parciales; únicamente duplicados exactos y dentro de una transformación reproducible.

## 3. Valores ausentes

| Variable | Ausentes | Porcentaje | Propuesta |
|---|---:|---:|---|
| `children` | 4 | <0,01 % | Imputar `0` y comprobar después que el total de huéspedes sea coherente. |
| `country` | 488 | 0,41 % | Crear categoría `Unknown`; agrupar países muy infrecuentes si es necesario. |
| `agent` | 16.340 | 13,69 % | Tratar como categoría y sustituir ausentes por `No agent`; comparar también un modelo sin esta variable. |
| `company` | 112.593 | 94,31 % | Excluir del modelo inicial. |

La imputación se realizará dentro del pipeline y se ajustará únicamente con el conjunto de entrenamiento.

## 4. Controles de calidad adicionales

- 180 registros sin huéspedes contabilizados.
- 715 reservas con cero noches.
- 403 registros con cero adultos, aunque algunos pueden incluir niños o bebés.
- 1 valor de `adr` negativo.
- 1 valor de `adr` superior a 1.000.
- 3.148 reservas con `lead_time` superior a 365 días.

### Propuesta

- Revisar los casos sin huéspedes y con cero noches antes de decidir si se excluyen.
- No eliminar automáticamente valores extremos de `lead_time`.
- Analizar los dos valores extremos de `adr`; aplicar una transformación robusta o límites solo dentro del pipeline si se justifica.

## 5. Variables para el primer modelo

### Numéricas y binarias

- `lead_time`
- `stays_in_weekend_nights`
- `stays_in_week_nights`
- `adults`
- `children`
- `babies`
- `is_repeated_guest`
- `previous_cancellations`
- `previous_bookings_not_canceled`
- `adr`
- `required_car_parking_spaces`
- `total_of_special_requests`

### Categóricas

- `hotel`
- `arrival_date_month`
- `meal`
- `country`
- `market_segment`
- `distribution_channel`
- `reserved_room_type`
- `deposit_type`
- `agent`
- `customer_type`

### Uso especial

- `arrival_date_year`: se utilizará para separar periodos, no como predictor del primer modelo.
- `arrival_date_week_number` y `arrival_date_day_of_month`: se evaluarán como variables cíclicas. Se evitará añadir representaciones temporales redundantes sin validación.

### Variables derivadas propuestas

- `total_nights`.
- `total_guests`.
- `has_children`.
- `has_previous_bookings`.
- codificación cíclica de mes o semana.

### Codificación

- Las categorías se transformarán con un codificador capaz de ignorar valores nuevos.
- En 2017 aparecen 18 países no vistos previamente, pero solo afectan a 28 registros.
- También aparecen 49 agentes nuevos, que afectan a 1.664 registros de 2017. Por ello `agent` debe manejar categorías desconocidas y se comparará su utilidad frente a un modelo sin esa columna.

## 6. Separación de entrenamiento, validación y prueba

### Propuesta principal sobre datos deduplicados

| Conjunto | Periodo de llegada | Registros | Tasa de cancelación |
|---|---|---:|---:|
| Entrenamiento | 01/07/2015-30/09/2016 | 44.991 | 24,21 % |
| Validación | 01/10/2016-31/12/2016 | 10.713 | 28,18 % |
| Prueba final | 01/01/2017-31/08/2017 | 31.692 | 31,91 % |

La fecha se construirá con `arrival_date_year`, `arrival_date_month` y `arrival_date_day_of_month`. No se utilizará `reservation_status_date` porque está vinculada al resultado final.

### Justificación

- El modelo aprende con el pasado y se evalúa con reservas futuras.
- Evita mezclar aleatoriamente periodos con condiciones distintas.
- La tasa de cancelación aumenta entre los tres periodos; esta deriva temporal hace que una prueba futura sea especialmente importante.
- El conjunto de prueba no se utilizará para seleccionar hiperparámetros ni para elegir el modelo.

Como comparación metodológica se podrá informar también una validación estratificada aleatoria, pero la conclusión principal se basará en la prueba temporal.

## 7. Orden de implementación propuesto

1. Validar estas decisiones entre los tres integrantes.
2. Implementar una función reproducible de deduplicación y controles de calidad.
3. Construir la fecha de llegada y efectuar la separación temporal.
4. Crear el preprocesamiento dentro de un pipeline.
5. Entrenar un modelo base sencillo.
6. Comparar después los cinco modelos exigidos con los mismos conjuntos y métricas.
7. Ejecutar el análisis de sensibilidad con los duplicados originales.

## 8. Puntos que requieren aprobación del equipo

- Confirmar el momento de predicción propuesto.
- Aceptar el conjunto deduplicado como escenario principal.
- Aprobar las reglas de imputación.
- Confirmar la lista inicial de variables.
- Aprobar la división temporal y reservar 2017 como prueba final.
