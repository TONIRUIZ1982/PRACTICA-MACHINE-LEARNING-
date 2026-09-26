[CmdletBinding()]
param(
    [switch]$AbrirJupyter
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Dataset = Join-Path $ProjectRoot 'data\raw\dataset_practica_final.csv'
$Notebook = Join-Path $ProjectRoot 'notebooks\02_modelo_final.ipynb'

Write-Host "`n=== Preparación de la presentación ===" -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $Python)) {
    throw "No se encuentra el entorno virtual en $Python. Ejecuta: python -m pip install -r requirements.txt"
}
if (-not (Test-Path -LiteralPath $Dataset)) {
    throw "No se encuentra el CSV en $Dataset. Copia el archivo de la asignatura antes de continuar."
}
if (-not (Test-Path -LiteralPath $Notebook)) {
    throw "No se encuentra el notebook final: $Notebook"
}

Write-Host "Comprobando datos, modelo y documentación..." -ForegroundColor Yellow
& $Python -m scripts.run_pipeline check -- --require-data --require-model
if ($LASTEXITCODE -ne 0) {
    throw "La comprobación de entrega no ha superado todos los controles."
}

Write-Host "`nTodo listo para la defensa." -ForegroundColor Green
Write-Host "Notebook principal: $Notebook"
Write-Host "Presentación: $(Join-Path $ProjectRoot 'presentations\Defensa_Cancelaciones_Hoteleras_v1.pptx')"

if ($AbrirJupyter) {
    $env:JUPYTER_RUNTIME_DIR = Join-Path $ProjectRoot '.jupyter_runtime'
    $env:JUPYTER_CONFIG_DIR = Join-Path $ProjectRoot '.jupyter_config'
    $env:JUPYTER_DATA_DIR = Join-Path $ProjectRoot '.jupyter_data'
    New-Item -ItemType Directory -Force -Path $env:JUPYTER_RUNTIME_DIR,$env:JUPYTER_CONFIG_DIR,$env:JUPYTER_DATA_DIR | Out-Null
    $Jupyter = Join-Path $ProjectRoot '.venv\Scripts\jupyter-lab.exe'
    Write-Host "`nIniciando JupyterLab. Para cerrarlo, pulsa Ctrl+C en esta ventana." -ForegroundColor Cyan
    & $Jupyter --notebook-dir=$ProjectRoot
}
else {
    Write-Host "`nPara abrir JupyterLab cuando vayas a exponer, repite este comando añadiendo -AbrirJupyter." -ForegroundColor Cyan
}