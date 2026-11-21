"""Réplica de YOLO_Baseline / YOLO_dropout: recorte por caja (sin máscara), relleno gris cuadrado, + 100 fotos externas solo en train.
Mismo split (604 rancho). Corre A) con externas, B) sin externas y C) DINOv2-L 448 + Ridge sobre los mismos recortes."""
import gc, math, shutil, sys, time
import numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
from datos import cargar, RAIZ
MODOS, sys.argv = set(sys.argv[1:]), sys.argv[:1]   # ridge.py lee sys.argv al importarse

DRIVE, SAL = RAIZ / 'drive', RAIZ / 'holdout'
LABELS, CROPS = DRIVE / 'yolo_labels', SAL / 'caja_recortes'
SEED, EPOCHS, PATIENCE, BATCH, CLS_SIZE, DROPOUT, DEVICE = 42, 60, 15, 8, 224, 0.30, 'mps'
EXTERNOS = {'ext_niu': 3.25, 'ext_mao': 3.5}          # salida del notebook (Excel, Rancho = 0)

# --- funciones copiadas del notebook ---
def read_box(path):
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    assert len(rows) == 1 and len(rows[0]) == 5, path
    return tuple(float(v) for v in rows[0][1:])
def pixel_box(box, size):
    xc, yc, bw, bh = box; w, h = size
    return (max(0, math.floor((xc-bw/2)*w)), max(0, math.floor((yc-bh/2)*h)),
            min(w, math.ceil((xc+bw/2)*w)), min(h, math.ceil((yc+bh/2)*h)))
def square_pad(image):
    side = max(image.size); canvas = Image.new('RGB', (side, side), (114, 114, 114))
    canvas.paste(image, ((side-image.width)//2, (side-image.height)//2)); return canvas

# --- manifiesto: 604 de rancho con el split único + externas con caja ---
d = cargar().merge(pd.read_csv(SAL / 'split_holdout.csv'), on='archivo_limpio', validate='1:1')
d['imagen'] = [DRIVE / a for a in d['archivo_limpio']]
ext = [dict(imagen=p, bcs=v, split='train') for g, v in EXTERNOS.items()
       for p in sorted((DRIVE / 'clean/externo' / g).glob('*')) if (LABELS / f'{p.stem}.txt').is_file()]
m = pd.concat([d.assign(fuente='rancho'), pd.DataFrame(ext).assign(fuente='externo')], ignore_index=True)
m['label'] = [LABELS / f'{Path(p).stem}.txt' for p in m['imagen']]
assert all(Path(p).is_file() for p in m['imagen']) and all(Path(p).is_file() for p in m['label'])
print(pd.crosstab([m.fuente, m.split], m.bcs), flush=True)

CROPS.mkdir(parents=True, exist_ok=True)
m['crop'] = [CROPS / f'image_{i:06d}.jpg' for i in range(len(m))]
for r in m.itertuples():
    if not r.crop.exists():
        with Image.open(r.imagen) as im:
            im = im.convert('RGB')          # igual que el notebook: sin transponer EXIF
        square_pad(im.crop(pixel_box(read_box(r.label), im.size))).save(r.crop)
rancho = m[m.fuente == 'rancho'].reset_index(drop=True)
te = (rancho.split == 'test').to_numpy()
out = rancho[['archivo_limpio', 'id_vaca', 'animal', 'bcs', 'split']].copy()

def carpeta(nombre, filas):
    raiz = SAL / nombre
    if not raiz.exists():
        for s in ['train', 'val', 'test']:
            for b in sorted(m['bcs'].unique()): (raiz / s / f'{b:.2f}').mkdir(parents=True, exist_ok=True)
        for r in filas.itertuples(): shutil.copy2(r.crop, raiz / r.split / f'{r.bcs:.2f}' / r.crop.name)
    return raiz

def predecir(path, imgs):
    from ultralytics import YOLO
    mod = YOLO(str(path)); vals = np.array([float(mod.names[k]) for k in sorted(mod.names)]); t1, pr = [], []
    for i in range(0, len(imgs), BATCH):
        for r in mod.predict(imgs[i:i + BATCH], imgsz=CLS_SIZE, device=DEVICE, augment=False, verbose=False):
            t1.append(float(r.names[r.probs.top1])); pr.append(float(r.probs.data.cpu().numpy() @ vals))
    return np.array(t1), np.array(pr)

if 'yolo' in MODOS:
    from ultralytics import YOLO
    for datos_nombre, filas in [('caja_con_ext', m), ('caja_sin_ext', m[m.fuente == 'rancho'])]:
        data = carpeta(f'yolo_{datos_nombre}', filas)
        for name, drop in [('base', None), ('dropout', DROPOUT)]:
            t0, run = time.time(), f'{datos_nombre}_{name}'
            kw = dict(data=str(data), epochs=EPOCHS, patience=PATIENCE, imgsz=CLS_SIZE, batch=BATCH, device=DEVICE, workers=2,
                      seed=SEED, deterministic=True, project=str(SAL / 'yolo_runs'), name=run, plots=True, exist_ok=True)
            if drop is not None: kw['dropout'] = drop
            YOLO('yolo26s-cls.pt').train(**kw)
            out[f'YOLO_{run}_top1'], out[f'YOLO_{run}_prob'] = predecir(SAL / 'yolo_runs' / run / 'weights/best.pt', [str(p) for p in rancho.crop])
            print(f'{run}: MAE test top1 {np.abs(out[f"YOLO_{run}_top1"][te] - out.bcs[te]).mean():.3f} · '
                  f'prob {np.abs(out[f"YOLO_{run}_prob"][te] - out.bcs[te]).mean():.3f} · {time.time() - t0:.0f} s', flush=True)
            out.to_csv(SAL / 'pred_caja_yolo.csv', index=False); gc.collect(); torch.mps.empty_cache()

if 'dino' in MODOS:
    import extraer as X, ridge as R
    mod, f = X.crear('dinov2_l', 448), []
    for i in range(0, len(rancho), 16):
        x = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB').resize((448, 448), Image.BICUBIC)) for p in rancho.crop[i:i + 16]]))
        x = ((x.permute(0, 3, 1, 2).float() / 255 - X.MEDIA) / X.DESV).to(X.DEV)
        a, b = X.feats(mod, 'dinov2_l', x), X.feats(mod, 'dinov2_l', torch.flip(x, dims=[3]))
        f.append(((a['cls_parches'] + b['cls_parches']) / 2).float().cpu().numpy())
    Xf = np.concatenate(f); np.save(SAL / 'caja_dinov2_l_448_cls_parches.npy', Xf)
    assert (R.d['archivo_limpio'].to_numpy() == rancho['archivo_limpio'].to_numpy()).all()
    tr, va = np.where(rancho.split == 'train')[0], np.where(rancho.split == 'val')[0]
    alphas = np.logspace(-1, 5, 25)
    err = [R.mae_w(R.Y[va], R.ajustar(Xf, tr, a).predict(Xf[va]), R.PESO[va]) for a in alphas]
    a = alphas[int(np.argmin(err))]
    out['ridge_caja_dinov2_l_448_cls_parches'] = R.ajustar(Xf, tr, a).predict(Xf)
    print(f'DINOv2-L 448 caja + Ridge: alpha {a:.1f} · MAE test {np.abs(out["ridge_caja_dinov2_l_448_cls_parches"][te] - out.bcs[te]).mean():.3f}', flush=True)
    out[['archivo_limpio', 'ridge_caja_dinov2_l_448_cls_parches']].to_csv(SAL / 'pred_caja_dino.csv', index=False)
