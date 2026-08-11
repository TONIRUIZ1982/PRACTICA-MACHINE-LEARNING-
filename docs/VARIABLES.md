# Definición y selección inicial de variables

## Objetivo

Este documento define las variables de `dataset_practica_final.csv` antes de iniciar el análisis en Jupyter. La finalidad es compartir una interpretación común de los datos, justificar qué columnas pueden utilizarse y prevenir fugas de información.

La variable objetivo es `is_canceled`:

- `0`: reserva no cancelada.
- `1`: reserva cancelada.

## Perfil inicial

- **119.390 registros** y **32 variables**.
- **31.994 duplicados exactos**. No deben eliminarse hasta comprobar si son duplicados técnicos o reservas legítimamente iguales.
- No canceladas: **75.166 (62,96 %)**.
- Canceladas: **44.224 (37,04 %)**.
- Ausentes: `children` 4; `country` 488 (0,41 %); `agent` 16.340 (13,69 %); `company` 112.593 (94,31 %).

## Momento de predicción

El equipo debe fijar cuándo se predice la cancelación. La propuesta inicial es hacerlo **después de crear la reserva y antes de la llegada**. Solo podrán emplearse datos conocidos en ese instante.

## Clasificación inicial

### Objetivo

- `is_canceled`.

### Candidatas iniciales

`hotel`, `lead_time`, `arrival_date_year`, `arrival_date_month`, `arrival_date_week_number`, `arrival_date_day_of_month`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `meal`, `country`, `market_segment`, `distribution_channel`, `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled`, `reserved_room_type`, `deposit_type`, `agent`, `customer_type`, `adr`, `required_car_parking_spaces` y `total_of_special_requests`.

Su inclusión definitiva dependerá del EDA, la calidad de los datos y los resultados de validación.

### Condicionadas por el momento de predicción

- `assigned_room_type`: puede asignarse después de crear la reserva.
- `booking_changes`: acumula cambios realizados durante la vida de la reserva.
- `days_in_waiting_list`: puede conocerse después de pasar por la lista de espera.

Solo se utilizarán si se demuestra que estaban disponibles al predecir.

### Excluidas inicialmente

- `reservation_status`: revela directamente el resultado final; es una fuga de información.
- `reservation_status_date`: contiene información posterior al resultado; es una fuga de información.
- `company`: se excluye del primer modelo por tener un 94,31 % de valores ausentes, aunque se conserva en el archivo original.

## Diccionario completo

| Variable | Tipo conceptual | Definición | Observaciones iniciales |
|---|---|---|---|
| `hotel` | Categórica nominal | Tipo de hotel: urbano o vacacional. | 2 categorías. |
| `is_canceled` | Binaria | Indica si la reserva fue cancelada. | Objetivo; sin ausentes. |
| `lead_time` | Numérica discreta | Días entre la creación de la reserva y la llegada. | Rango 0-737; revisar extremos. |
| `arrival_date_year` | Temporal discreta | Año previsto de llegada. | 2015-2017; valorar validación temporal. |
| `arrival_date_month` | Categórica cíclica | Mes previsto de llegada. | 12 categorías. |
| `arrival_date_week_number` | Temporal cíclica | Semana del año prevista para la llegada. | Rango 1-53. |
| `arrival_date_day_of_month` | Temporal cíclica | Día del mes previsto para la llegada. | Rango 1-31. |
| `stays_in_weekend_nights` | Numérica discreta | Noches de fin de semana reservadas. | Rango 0-19. |
| `stays_in_week_nights` | Numérica discreta | Noches entre semana reservadas. | Rango 0-50. |
| `adults` | Numérica discreta | Número de adultos. | Rango 0-55; validar extremos y ceros. |
| `children` | Numérica discreta | Número de niños. | 4 ausentes; rango 0-10. |
| `babies` | Numérica discreta | Número de bebés. | Rango 0-10. |
| `meal` | Categórica nominal | Régimen de comidas contratado. | 5 categorías; revisar códigos. |
| `country` | Categórica nominal | País de procedencia del cliente. | 177 categorías; 488 ausentes. |
| `market_segment` | Categórica nominal | Segmento de mercado de la reserva. | 8 categorías. |
| `distribution_channel` | Categórica nominal | Canal de distribución de la reserva. | 5 categorías. |
| `is_repeated_guest` | Binaria | Indica si el cliente ya había reservado. | Valores 0 y 1. |
| `previous_cancellations` | Numérica discreta | Cancelaciones anteriores del cliente. | Rango 0-26. |
| `previous_bookings_not_canceled` | Numérica discreta | Reservas anteriores no canceladas. | Rango 0-72. |
| `reserved_room_type` | Categórica nominal | Código anonimizado de la habitación reservada. | 10 categorías. |
| `assigned_room_type` | Categórica nominal | Código anonimizado de la habitación asignada. | 12 categorías; validar disponibilidad temporal. |
| `booking_changes` | Numérica discreta | Número de cambios en la reserva. | Rango 0-21; validar disponibilidad temporal. |
| `deposit_type` | Categórica nominal | Tipo de depósito o garantía. | 3 categorías. |
| `agent` | Identificador categórico | Código del agente de la reserva. | 333 códigos; 13,69 % de ausentes. |
| `company` | Identificador categórico | Código de la empresa vinculada. | 94,31 % de ausentes; excluir inicialmente. |
| `days_in_waiting_list` | Numérica discreta | Días en lista de espera. | Rango 0-391; validar disponibilidad temporal. |
| `customer_type` | Categórica nominal | Tipo de cliente o contratación. | 4 categorías. |
| `adr` | Numérica continua | Tarifa media diaria. | Rango -6,38 a 5.400; revisar extremos. |
| `required_car_parking_spaces` | Numérica discreta | Plazas de aparcamiento solicitadas. | Rango 0-8. |
| `total_of_special_requests` | Numérica discreta | Número de peticiones especiales. | Rango 0-5. |
| `reservation_status` | Categórica nominal | Estado final de la reserva. | Excluir: fuga directa. |
| `reservation_status_date` | Fecha | Fecha del último estado registrado. | Excluir: información posterior. |

## Variables derivadas para evaluar

- `total_nights = stays_in_weekend_nights + stays_in_week_nights`.
- `total_guests = adults + children + babies`.
- `has_children`: indica si viajan niños o bebés.
- `has_previous_bookings`: indica si existe historial previo.
- `room_changed`: compara habitación reservada y asignada, solo si esta última es válida temporalmente.
- Transformaciones cíclicas de mes o semana para representar estacionalidad.

## Decisiones pendientes del equipo

1. Momento exacto de predicción.
2. Tratamiento de los duplicados.
3. Imputación de `children`, `country` y `agent`.
4. Exclusión inicial de `company`.
5. Uso de las variables condicionadas temporalmente.
6. Agrupación de categorías poco frecuentes.
7. Validación aleatoria o temporal para datos de 2015-2017.

Estas decisiones deberán justificarse con el análisis exploratorio y validarse dentro de pipelines, no solo mediante correlaciones.
