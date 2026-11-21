"""YOLO26s-cls (base y dropout 0.3) como en YOLO_Baseline / YOLO_dropout, sobre los mismos recortes SAM + CLAHE y el split único."""
import gc, shutil, sys, time
import numpy as np, pandas as pd, torch
from ultralytics import YOLO
from datos import cargar, DATOS, RAIZ

SAL = RAIZ / 'holdout'
SEED, EPOCHS, PATIENCE, BATCH, CLS_SIZE, DROPOUT = 42, 60, 15, 8, 224, 0.30
DEVICE = 'mps'
d = cargar().merge(pd.read_csv(SAL / 'split_holdout.csv'), on='archivo_limpio', validate='1:1')
CLS = SAL / 'yolo_cls_datos'
if not CLS.exists():
    for s in ['train', 'val', 'test']:
        for b in sorted(d['bcs'].unique()):
            (CLS / s / f'{b:.2f}').mkdir(parents=True, exist_ok=True)
    for i, r in d.iterrows():
        shutil.copy2(DATOS / r['archivo_mascara_clahe'], CLS / r['split'] / f"{r['bcs']:.2f}" / f'image_{i:06d}.jpg')

def train_model(name, dropout=None):
    kw = dict(data=str(CLS), epochs=EPOCHS, patience=PATIENCE, imgsz=CLS_SIZE, batch=BATCH, device=DEVICE, workers=2,
              seed=SEED, deterministic=True, project=str(SAL / 'yolo_runs'), name=name, plots=True, exist_ok=True)
    if dropout is not None: kw['dropout'] = dropout
    m = YOLO('yolo26s-cls.pt'); m.train(**kw)
    return SAL / 'yolo_runs' / name / 'weights' / 'best.pt'

def predecir(path, imgs):
    m = YOLO(str(path)); top1, esperado = [], []
    vals = np.array([float(m.names[k]) for k in sorted(m.names)])
    for i in range(0, len(imgs), BATCH):
        for r in m.predict(imgs[i:i + BATCH], imgsz=CLS_SIZE, device=DEVICE, augment=False, verbose=False):
            p = r.probs.data.cpu().numpy()
            top1.append(float(r.names[r.probs.top1])); esperado.append(float(p @ vals))
    return np.array(top1), np.array(esperado)

imgs = [str(DATOS / r) for r in d['archivo_mascara_clahe']]
out = d[['archivo_limpio']].copy()
for name, drop in [('classifier_baseline', None), ('classifier_dropout', DROPOUT)]:
    t0 = time.time(); best = train_model(name, drop)
    out[f'YOLO_{name}_top1'], out[f'YOLO_{name}_prob'] = predecir(best, imgs)
    te = d['split'] == 'test'
    print(f'{name}: MAE test top1 {np.abs(out[f"YOLO_{name}_top1"][te] - d.bcs[te]).mean():.3f} · '
          f'prob {np.abs(out[f"YOLO_{name}_prob"][te] - d.bcs[te]).mean():.3f} · {time.time() - t0:.0f} s', flush=True)
    out.to_csv(SAL / 'pred_yolo.csv', index=False)
    gc.collect(); torch.mps.empty_cache()
