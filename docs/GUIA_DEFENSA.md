# Guía para la defensa

## Preguntas clave y respuestas breves

### ¿Cuál es el objetivo de negocio?

Estimar la cancelación al confirmar la reserva para anticipar la ocupación y aplicar acciones preventivas.

### ¿Por qué la separación es temporal?

Porque entrena con reservas pasadas y mide el comportamiento en reservas futuras. Una división aleatoria mezclaría periodos y ocultaría parte de la deriva temporal.

### ¿Por qué F1 es la métrica principal?

Porque combina precision y recall. Queremos detectar cancelaciones, pero evitando generar demasiadas alertas erróneas.

### ¿Qué información se excluyó?

Se excluyeron estados finales y fechas asociadas por fuga de información. También variables que pueden cambiar tras reservar y `company` por tener más del 94 % de valores ausentes.

### ¿Por qué se eliminaron variables adicionales?

Se compararon sobre validación temporal. `agent` no mejoró el resultado, el día del mes no añadió señal y varios agregados repetían información que ya estaba disponible en sus variables de origen.

### ¿Por qué ganó XGBoost?

Fue el mejor modelo por F1 de validación tras optimizar hiperparámetros y umbral. Además mantuvo un F1 de 0,6280 en el periodo futuro de 2017.

### ¿Por qué el umbral no es 0,5?

0,5 es una convención, no una decisión de negocio. El umbral 0,3619 maximiza F1 en validación y permite detectar más cancelaciones manteniendo precisión razonable.

### ¿Qué pasó con 2017?

Se reservó hasta elegir modelo, hiperparámetros y umbral. Después se evaluó una sola vez y quedó cerrado para evitar seleccionar el modelo mirando los resultados finales.

### ¿Cómo se hace una predicción nueva?

Se carga el pipeline XGBoost local y se ejecuta `scripts/predict_cancellations.py` con un CSV que contenga las variables disponibles en el momento de la reserva.
