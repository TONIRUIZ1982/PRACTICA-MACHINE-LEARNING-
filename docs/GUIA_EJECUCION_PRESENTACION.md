# Guía de ejecución para la presentación

## Antes de salir de casa

1. Comprueba que el CSV está en `data/raw/dataset_practica_final.csv`.
2. Abre PowerShell y ejecuta:

```powershell
cd "C:\PRACTICA MACHINE LEARNING"
powershell -ExecutionPolicy Bypass -File .\scripts\iniciar_presentacion.ps1
```

Debe aparecer el mensaje **“Todo listo para la defensa”**.

## Durante la presentación

Si necesitas mostrar el notebook o ejecutar una celda, usa:

```powershell
cd "C:\PRACTICA MACHINE LEARNING"
powershell -ExecutionPolicy Bypass -File .\scripts\iniciar_presentacion.ps1 -AbrirJupyter
```

JupyterLab se abrirá en el navegador. Abre `notebooks/02_modelo_final.ipynb`.

## Orden recomendado de exposición

1. Problema y objetivo: predecir la cancelación tras confirmar una reserva.
2. Datos y decisiones: duplicados, valores ausentes, fugas y partición temporal.
3. Modelos comparados: regresión logística, árbol, Random Forest, XGBoost y red neuronal.
4. Resultado: XGBoost, F1 de 0,6280 y ROC-AUC de 0,8098 en el test de 2017.
5. Reproducibilidad: README, tests, scripts y control de entrega.

No entrenes modelos durante la defensa. Muestra el notebook, los resultados ya obtenidos y la presentación. Para cerrar JupyterLab, vuelve a PowerShell y pulsa `Ctrl+C`.