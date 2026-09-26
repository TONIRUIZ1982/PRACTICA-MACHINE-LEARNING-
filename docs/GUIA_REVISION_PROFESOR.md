# Guía de revisión para el profesorado

## Punto de entrada

La rama `main` actúa como índice de revisión. La implementación final está disponible en `cierrentrega/toni`.

## Ramas disponibles

| Rama | Aportación |
| --- | --- |
| `cierrentrega/toni` | Cierre técnico: pipeline, modelos, evaluación temporal, inferencia, tests, documentación, notebook final y presentación. |
| `revision/vanessa` | Análisis exploratorio y preparación desarrollados por Vanessa Romero. |
| `revision/jeronimo` | Modelos y notebooks comparativos desarrollados por Jeronimo Javier. |

## Criterios de entrega comprobados

- La validación temporal reserva 2017 como test final.
- Se comparan regresión logística, árbol de decisión, Random Forest, XGBoost y red neuronal.
- XGBoost es el modelo seleccionado por F1 de validación.
- Existen pruebas automatizadas y un control que impide versionar CSV, modelos, artefactos y resultados locales.
- El README de la rama de cierre incluye instalación, ejecución, resultados y estructura del proyecto.

## Material de defensa

La rama de cierre contiene el notebook `02_modelo_final.ipynb`, la presentación en `presentations/` y la documentación de apoyo en `docs/`.