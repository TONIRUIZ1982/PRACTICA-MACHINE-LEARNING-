# Práctica final - Predicción de cancelaciones hoteleras

Repositorio de la práctica final de Machine Learning. El proyecto estima la probabilidad de cancelación de una reserva hotelera inmediatamente después de su confirmación.

## Guía de revisión

La revisión se ha organizado por ramas para conservar las aportaciones de cada integrante y facilitar la trazabilidad del trabajo.

| Integrante / finalidad | Rama | Contenido principal |
| --- | --- | --- |
| Antonio José Ruiz Expósito - cierre técnico | [`cierrentrega/toni`](../../tree/cierrentrega/toni) | Pipeline final, selección de variables, comparación y optimización de modelos, inferencia, pruebas, documentación, notebook final y presentación. |
| Vanessa Romero | [`feature/vane`](../../tree/feature/vane) | Análisis exploratorio y materiales de preparación asociados a su aportación. |
| Jeronimo Javier | [`feature-/jero`](../../tree/feature-/jero) | Modelos y notebooks comparativos desarrollados en su rama. |

La rama recomendada para revisar la solución técnica consolidada es [`cierrentrega/toni`](../../tree/cierrentrega/toni).

## Verificación reproducible

En la rama de cierre:

1. Crear un entorno con Python 3.11 o superior.
2. Instalar dependencias con `python -m pip install -r requirements.txt`.
3. Copiar el CSV de la asignatura en `data/raw/dataset_practica_final.csv`.
4. Ejecutar `python -m pytest -q`.
5. Ejecutar `python -m scripts.run_pipeline check -- --require-data --require-model`.

El CSV original, los modelos entrenados, los resultados generados y el PDF de entrega permanecen fuera de Git por privacidad, tamaño y reproducibilidad.

## Entregables

- Notebook de análisis exploratorio.
- Notebook del modelo final.
- Presentación de defensa.
- Documentación de variables, decisiones de datos, resultados y cierre técnico.
- Memoria técnica en PDF para la entrega por PontIA.