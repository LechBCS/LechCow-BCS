# LechCow-BCS

Estimación de la **condición corporal (BCS, *Body Condition Score*)** de vacas lecheras Holstein a partir de fotografías, usando visión por computadora.

Proyecto del curso **TC3009C — Inteligencia Artificial Avanzada para la Ciencia de Datos** (Tecnológico de Monterrey), desarrollado con el **CAETEC** como socio formador y siguiendo la metodología **CRISP-DM**.

## Problema y objetivo

El BCS es una calificación (escala 1 a 5) del nivel de reservas de grasa de una vaca. Se usa para cuidar su salud, su producción de leche y su fertilidad. Hoy se asigna a ojo, lo que lleva tiempo y varía según quién califica.

El objetivo es un modelo que estime el BCS a partir de fotos de la parte trasera y lateral de la vaca, con un **error absoluto medio (MAE) de 0.25 puntos o menos**.

## Datos

| Fuente | Uso | En este repositorio |
|---|---|---|
| **CAETEC**: fotos de ~25 vacas del establo, con BCS del equipo y del sistema DeLaval | Evaluación del modelo | **No.** Son datos confidenciales y se guardan en un Drive privado del equipo |
| **ScienceDB**: *Dairy cow body condition score target detection data set* (DOI [10.57760/sciencedb.16704](https://doi.org/10.57760/sciencedb.16704)), 53,566 imágenes, clases 3.25 a 4.25 | Entrenamiento | Sólo una muestra pequeña en `data/sample/` |

## Enfoque

1. **ETL con PySpark** (`notebooks/ETL_BCS_PySpark.ipynb`): extrae las imágenes, obtiene el BCS y el ID de vaca, limpia (duplicados por hash SHA-256, archivos vacíos, tomas sin ID), corrige la orientación, redimensiona a 640 px y guarda JPEG **sin metadatos EXIF/GPS**. Las tablas se guardan en Parquet. Corre en local, en Colab o en el clúster Spark de 3 máquinas del equipo.
2. **Validación cruzada k-fold agrupada por vaca** (`StratifiedGroupKFold`, k = 5, semilla 42). Ninguna vaca aparece en más de un fold, para que el modelo no aprenda a reconocer a la vaca en lugar de su condición corporal.
3. **Modelado con YOLO** (`notebooks/YOLO_k_*_BCS.ipynb`): un detector localiza a la vaca y un clasificador estima su BCS sobre el recorte. Se entrena un par de modelos por fold.
4. **Evaluación** (`notebooks/YOLO_BCS_5Fold_Reevaluation.ipynb`): recarga los cinco pares de modelos y calcula MAE, RMSE y aciertos dentro de ±0.25 por fold, por vaca y en conjunto, comparando contra un modelo base.

El volumen total (~5.8 GB, sin datos en tiempo real) no requiere un enfoque de Big Data, pero el pipeline está escrito en PySpark para poder escalar (ver Paso 9 del notebook de ETL).

## Estructura del repositorio

```
notebooks/      ETL en PySpark, entrenamiento YOLO por fold y evaluación de 5 folds
data/sample/    Muestra de ScienceDB generada por el ETL (imágenes, muestra.csv, folds_resumen.csv)
docs/           Entregables del curso, una carpeta por fase de CRISP-DM
  1_business_understanding/   Business Understanding, One Page y MoU con el CAETEC
  2_data_understanding/       Data Understanding y reporte de calidad de datos
  3_data_preparation/         Data Preparation e inventario de limpieza
  4_modeling/                 Modeling
  5_datos/                    Reporte del Reto Datos
  privacidad_seguridad/       Reportes individuales de privacidad y seguridad
.github/        Plantilla de PR y CI (protección de datos y lint)
```

## Cómo ejecutar

```bash
pip install -r requirements.txt
```

Abre `notebooks/ETL_BCS_PySpark.ipynb` y, en el Paso 0:

- **Local o Colab:** `MODO = "local"` y `DATA_PATH` apuntando a la muestra.
- **Clúster Spark:** `MODO = "cluster"`, copia la muestra a `/tmp/bcs_sample` en las 3 máquinas y en el driver, y usa `DATA_PATH = "/tmp/bcs_sample"`.

Los notebooks de YOLO corren en Google Colab con GPU.

## Privacidad

Las fotos, etiquetas y registros del CAETEC están protegidos por el MoU y **nunca se suben a este repositorio**. El CI rechaza los PR que incluyan archivos HEIC, carpetas de datos crudos, credenciales, imágenes fuera de `data/sample/` o archivos de más de 5 MB.

## Equipo

- Carlos Martínez
- Charbel Chávez
- Dongju Mun
- Evan Santana
- Juan Pablo Arteaga
- Juan Sebastián Chitiva
