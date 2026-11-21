# Notebooks

Las carpetas siguen el orden del pipeline. Cada etapa usa lo que produce la anterior.

| Etapa | Carpeta | Qué contiene |
|---|---|---|
| 1 | `1_datos/` | `ETL_BCS_PySpark.ipynb`: extracción, limpieza y carga de las imágenes con PySpark |
| 2 | `2_recortes/` | `YOLO_Detector.ipynb` localiza a la vaca y `SAM.ipynb` recorta su silueta (los recortes que usan todos los modelos) |
| 3 | `3_modelado_yolo_cnn/` | Primeras iteraciones de modelado: clasificadores YOLO (`YOLO_Baseline`, `YOLO_dropout`, comparación y reevaluación en 5 folds) y la ResNet-50 afinada (`Modelo_CNN_Vacuno`). En `yolo_por_fold/` están los entrenamientos de YOLO de cada fold |
| 4 | `4_modelado_dinov2/` | `Comparacion_Modelos_BCS.ipynb` compara características congeladas contra la CNN afinada en 5 folds; `DINOv2_Ridge_BCS.ipynb` entrena el modelo aprobado (DINOv2-L + Ridge) en la partición train/val/test |
| 5 | `5_evaluacion/` | `Evaluacion_Modelos_BCS.ipynb`: métricas, criterios de éxito, pruebas estadísticas y diagnósticos de todos los modelos. En `scripts_benchmark/` están los scripts que generaron las predicciones que lee |

Las fotos y etiquetas del CAETEC son confidenciales y no están en el repositorio; los notebooks las leen del Drive privado del equipo o de una carpeta local fuera del repo.
