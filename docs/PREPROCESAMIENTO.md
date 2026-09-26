# Módulo inicial de preparación de datos

> Estado: implementación provisional para revisión del equipo. No modifica el CSV original ni entrena modelos.

## Objetivo

Este módulo traslada a código reutilizable las decisiones propuestas en `DECISIONES_DATOS.md`. Se ha creado fuera de los notebooks para que los cinco modelos obligatorios utilicen exactamente la misma carga, validación y separación de datos.

Las fechas de corte, la deduplicación y algunas exclusiones continúan siendo configurables hasta que los tres integrantes las aprueben.

## Archivos añadidos

```text
src/hotel_cancellation/
├── __init__.py
├── config.py
└── data.py

scripts/
├── __init__.py
└── validate_data.py

tests/
├── __init__.py
└── test_data.py
```

### `config.py`

Centraliza las 32 columnas esperadas, la variable objetivo, las fugas de información, las variables cuya disponibilidad temporal debe confirmar el equipo y la conversión de meses.

### `data.py`

Incluye funciones comentadas para cargar el CSV local, validar su esquema, crear `arrival_date`, deduplicar opcionalmente, seleccionar predictores y separar cronológicamente los datos. Las funciones devuelven copias y no sobrescriben el archivo original.

### `validate_data.py`

Desde la raíz del repositorio:

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.validate_data
```

Para aplicar la propuesta provisional de deduplicación:

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.validate_data --deduplicate
```

Con el CSV facilitado, el segundo comando debería informar de 119.390 filas cargadas, 31.994 duplicados eliminados, 44.991 filas de entrenamiento, 10.713 de validación y 31.692 de prueba.

### Pruebas automáticas

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

Si el CSV no está en `data/raw/dataset_practica_final.csv`, las pruebas de integración se omiten, pero las pruebas unitarias siguen funcionando.

## Decisiones protegidas por el código

- `reservation_status` y `reservation_status_date` nunca se ofrecen como predictores.
- `assigned_room_type`, `booking_changes` y `days_in_waiting_list` se excluyen por defecto mientras se confirma el momento de predicción.
- `company` se excluye inicialmente por sus ausentes, pero puede activarse para una comparación.
- Los cortes temporales son parámetros visibles y pueden cambiarse tras la revisión del equipo.

## Próximo paso

Cuando el equipo confirme estas reglas, el módulo podrá alimentar un único pipeline. Después se implementará la regresión logística como línea base y se comparará con los otros cuatro modelos obligatorios.
