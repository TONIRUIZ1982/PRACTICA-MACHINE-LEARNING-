# Predicción de cancelaciones hoteleras

Proyecto final de Machine Learning para estimar la probabilidad de cancelación de una reserva justo después de confirmarla.

## Equipo

- Antonio José Ruiz Expósito
- Vanessa Romero
- Jero (GitHub: `veneloforte`)

El trabajo se ha desarrollado con responsabilidad compartida. Las decisiones sobre datos, variables, modelos, validación y documentación se han revisado conjuntamente.

## Datos y objetivo

El dataset contiene 119.390 registros y 32 variables. La variable objetivo es `is_canceled`: `1` representa una cancelación y `0` una reserva no cancelada. El CSV original se mantiene solo en local en `data/raw/dataset_practica_final.csv` y no se publica en GitHub.

El escenario principal elimina 31.994 duplicados exactos. Se excluyen las fugas directas `reservation_status` y `reservation_status_date`, además de variables que pueden cambiar después de confirmar la reserva.

## Metodología

La división temporal utiliza entrenamiento hasta septiembre de 2016, validación entre octubre y diciembre de 2016, y un test final reservado para 2017.

Se comparan regresión logística, árbol de decisión, Random Forest, XGBoost y una red neuronal multicapa con Keras/TensorFlow. F1 es la métrica principal porque equilibra la detección de cancelaciones y las falsas alarmas.

## Resultado final

XGBoost obtuvo el mayor F1 de validación (`0,6218`) con un umbral de `0,3619`. En la evaluación final de 2017 obtuvo accuracy de `0,6866`, precision de `0,5055`, recall de `0,8289`, F1 de `0,6280` y ROC-AUC de `0,8098`.

El test de 2017 queda cerrado y no se reutiliza para seleccionar variables, hiperparámetros ni umbrales.

## Instalación

Se recomienda Python 3.11 o superior. Crear un entorno virtual, activarlo e instalar las dependencias con `python -m pip install -r requirements.txt`.

Copiar el CSV proporcionado por la asignatura en `data/raw/dataset_practica_final.csv` antes de ejecutar el proyecto.

## Ejecución

Los comandos principales se ejecutan desde la raíz del repositorio:

- `python -m scripts.run_pipeline validate -- --deduplicate`
- `python -m scripts.run_pipeline eda -- --deduplicate`
- `python -m scripts.run_pipeline compare`
- `python -m scripts.run_pipeline optimize`
- `python -m scripts.run_pipeline check -- --require-data --require-model`
- `python -m scripts.run_pipeline predict -- --input ruta/reservas_nuevas.csv --output ruta/predicciones.csv`

La optimización bloquea la reutilización accidental del test de 2017.

## Estructura

- `src/hotel_cancellation/`: código reutilizable de datos, variables y modelos.
- `scripts/`: validación, entrenamiento e inferencia.
- `notebooks/`: EDA y notebook final de apoyo.
- `docs/`: decisiones, resultados, informe y guía de defensa.
- `presentations/`: presentación de apoyo para la defensa.
- `tests/`: pruebas automatizadas.

## Documentación

- `docs/VARIABLES.md`: variables y fugas de información.
- `docs/DECISIONES_DATOS.md`: duplicados, ausentes y partición temporal.
- `docs/RESULTADO_FINAL.md`: selección y métricas finales.
- `docs/CIERRE_TECNICO.md`: verificación antes de la entrega.
- `docs/INFORME_FINAL_BORRADOR.md`: base del PDF para PontIA.

Los datos originales, modelos entrenados y resultados generados se mantienen en directorios ignorados por Git.
