# Scripts de los experimentos de Evaluation

Estos scripts generaron las predicciones que lee `../Evaluacion_Modelos_BCS.ipynb`. El modelo principal (DINOv2-L + Ridge) también tiene su propia libreta, `notebooks/4_modelado_dinov2/DINOv2_Ridge_BCS.ipynb`, que da los mismos resultados.

Los datos del CAETEC son confidenciales y no están en el repositorio. Los scripts esperan esta carpeta fuera del repo:

```
~/bcs_data/
  imagenes.csv              tabla de imágenes del Drive del equipo
  nube/datos/recortes/      recortes de SAM (salida de SAM.ipynb) y recortes.csv
  drive/                    fotos limpias y etiquetas de caja (solo para yolo_caja.py)
```

Todas las salidas (caché de características, predicciones y modelos) se escriben dentro de `~/bcs_data/`. Las predicciones llevan los identificadores de las vacas, así que nunca se suben al repositorio.

## Orden para correrlos

Desde esta carpeta y con las dependencias de `requirements.txt`:

| Paso | Comando | Qué hace |
|---|---|---|
| 1 | `python split_holdout.py` | Partición única train/val/test por animal (`grouped_holdout`, semilla 42) |
| 2 | `python extraer.py dinov2_l mascara_clahe 448` | Características congeladas (repetir con `dinov2_b`, `resnet50` y tamaño 224 para la ablación) |
| 3 | `python holdout.py` | E0, Ridge y MLP sobre las características (alpha elegido en val) y E1 entrenada solo con train |
| 4 | `python yolo_cls.py` | YOLO26s-cls base y con dropout sobre los recortes de SAM |
| 5 | `python yolo_caja.py yolo dino` | YOLO26s-cls y DINOv2-L + Ridge sobre el recorte por caja, con y sin las 100 fotos externas |
| 6 | `python folds_todo.py rapido yolo_sam yolo_caja` | Los mismos modelos en los 5 folds por animal (solo validación) |

Archivos de apoyo: `datos.py` (carga y filtros de los recortes), `ridge.py` (Ridge con pesos por sesión) y `e1_cnn.py` (ResNet-50 afinada, receta 05 de `Modelo_CNN_Vacuno.ipynb`).

En el reporte solo se usan las corridas de YOLO sin fotos externas.
