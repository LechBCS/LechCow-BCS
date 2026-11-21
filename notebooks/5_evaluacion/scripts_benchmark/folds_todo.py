"""Los 5 folds de Charbel (por animal, semilla 42) con el mismo protocolo que el split único:
test = fold k; val = 20 % de los animales restantes (grouped_holdout); train = el resto.
Train ajusta, val elige (alpha / early stopping), test solo se mide. Modos: rapido, yolo_sam, yolo_caja."""
import gc, hashlib, os, sys, time
import numpy as np, pandas as pd
MODOS, sys.argv = set(sys.argv[1:]), sys.argv[:1]
from datos import cargar, DATOS, RAIZ, K
from split_holdout import grouped_holdout          # misma función del notebook YOLO

SAL = RAIZ / 'folds'; SAL.mkdir(exist_ok=True)
d = cargar()                                        # mismo orden que caches, E1 y ridge.py
d['animal_id'] = d['animal']

# --- partición por fold y verificaciones de fuga ---
part = {}
for k in range(1, K + 1):
    dev = d[d.fold != k]
    fit, val = grouped_holdout(dev, 0.20, 42)
    s = pd.Series('test', index=d.index); s[d.archivo_limpio.isin(fit.archivo_limpio)] = 'train'; s[d.archivo_limpio.isin(val.archivo_limpio)] = 'val'
    part[k] = s.to_numpy()
    g = pd.DataFrame({'a': d.animal, 's': part[k]}).groupby('a')['s'].nunique()
    assert g.max() == 1, f'animal en dos particiones, fold {k}'
pd.DataFrame({f'fold{k}': v for k, v in part.items()}).assign(archivo_limpio=d.archivo_limpio).to_csv(SAL / 'particiones.csv', index=False)

def guardar(nombre, pred):
    f = SAL / f'pred_{nombre}.csv'
    pd.DataFrame({'archivo_limpio': d.archivo_limpio, nombre: pred}).to_csv(f, index=False)

def hecho(nombre): return (SAL / f'pred_{nombre}.csv').exists()

if 'rapido' in MODOS:
    # pixeles idénticos cruzando train/test (recortes SAM y por caja)
    import yolo_caja as C
    for col, rutas in [('sam', [DATOS / r for r in d.archivo_mascara_clahe]), ('caja', list(C.rancho.crop))]:
        h = np.array([hashlib.sha256(open(p, 'rb').read()).hexdigest() for p in rutas])
        for k in part:
            cruzan = set(h[part[k] == 'test']) & set(h[part[k] != 'test'])
            assert not cruzan, f'{col}: {len(cruzan)} imágenes idénticas cruzan train/test en fold {k}'
    print('verificación: ningún animal ni imagen idéntica cruza particiones en los 5 folds', flush=True)

    import ridge as R, glob
    assert (R.d.archivo_limpio.to_numpy() == d.archivo_limpio.to_numpy()).all()
    feats = {os.path.basename(f)[:-4]: np.load(f, allow_pickle=True)['X'] for f in sorted(glob.glob(str(RAIZ / 'cache' / '*.npz')))}
    feats['caja_dinov2_l_448_cls_parches'] = np.load(RAIZ / 'holdout' / 'caja_dinov2_l_448_cls_parches.npy')
    Y = R.Y.copy()
    pred = np.zeros(len(d))
    for k in part:
        tr = part[k] == 'train'
        pred[part[k] == 'test'] = d[tr].groupby('id_vaca')['bcs'].first().mean()
    guardar('E0_constante', pred)

    def correr(X, tipo, alphas, Yuse=None):
        R.TIPO = tipo; pred, al = np.zeros(len(d)), []
        for k in part:
            tr, va, te = (np.where(part[k] == s)[0] for s in ['train', 'val', 'test'])
            R.Y = Y if Yuse is None else Yuse[k]
            err = [R.mae_w(Y[va], R.pred_de(R.ajustar(X, tr, a), X[va]), R.PESO[va]) for a in alphas]
            a = alphas[int(np.argmin(err))]; al.append(a)
            pred[te] = R.pred_de(R.ajustar(X, tr, a), X[te])
        R.Y = Y
        return pred, al
    for nombre, X in feats.items():
        for tipo, alphas in [('ridge', np.logspace(-1, 5, 25)), ('mlp', np.logspace(-2, 2, 5))]:
            if tipo == 'mlp' and 'dinov2_l' not in nombre: continue
            n = f'{tipo}_{nombre}'
            if hecho(n): continue
            p, al = correr(X, tipo, alphas); guardar(n, p)
            print(f'{n:52s} MAE {np.abs(p - Y).mean():.3f} · alphas {[f"{a:g}" for a in al]}', flush=True)

    # control: etiquetas barajadas entre sesiones de train -> debe quedar como E0
    rng = np.random.default_rng(0); Yp = {}
    for k in part:
        y = Y.copy(); tr = part[k] == 'train'
        ses = d[tr].groupby('id_vaca')['bcs'].first()
        y[tr] = d[tr].id_vaca.map(dict(zip(ses.index, rng.permutation(ses.values)))).to_numpy()
        Yp[k] = y
    p, _ = correr(feats['dinov2_l_mascara_clahe_cls_parches_448'], 'ridge', np.logspace(-1, 5, 25), Yp)
    guardar('control_barajado_dinov2_l_448', p)
    print(f'control etiquetas barajadas (DINOv2-L 448 + Ridge): MAE {np.abs(p - Y).mean():.3f}', flush=True)

    if not hecho('E1_cnn_afinada'):
        import torch, e1_cnn as E
        pred, t0 = np.zeros(len(d)), time.time()
        for k in part:
            m, mu, sd = E.entrenar(d[part[k] == 'train'], E.SEMILLA + k)
            te = np.where(part[k] == 'test')[0]
            pred[te] = E.predecir(m, torch.tensor(te), mu, sd)
            del m; torch.mps.empty_cache()
        guardar('E1_cnn_afinada', pred)
        print(f'E1: MAE {np.abs(pred - Y).mean():.3f} · {time.time() - t0:.0f} s', flush=True)

def yolo(config, rutas, extra=None):
    """config: nombre; rutas: recorte por fila de d; extra: DataFrame de externas (crop, bcs) solo para train."""
    import torch, yolo_caja as C
    from ultralytics import YOLO
    clases = sorted(set(d.bcs) | (set(extra.bcs) if extra is not None else set()))
    for drop in [None, 0.30]:
        n = f'YOLO_{config}_{"dropout" if drop else "base"}'
        if hecho(n + '_prob'): continue
        top1, prob = np.full(len(d), np.nan), np.full(len(d), np.nan)
        for k in part:
            t0, raiz = time.time(), SAL / 'yolo_datos' / f'{config}_f{k}'
            if not raiz.exists():
                for s in ['train', 'val', 'test']:
                    for b in clases: (raiz / s / f'{b:.2f}').mkdir(parents=True, exist_ok=True)
                for i, (p, b, s) in enumerate(zip(rutas, d.bcs, part[k])):
                    os.symlink(p, raiz / s / f'{b:.2f}' / f'image_{i:06d}.jpg')
                if extra is not None:
                    for j, r in enumerate(extra.itertuples()):
                        os.symlink(r.crop, raiz / 'train' / f'{r.bcs:.2f}' / f'ext_{j:06d}.jpg')
            kw = dict(data=str(raiz), epochs=60, patience=15, imgsz=224, batch=8, device='mps', workers=2, seed=42,
                      deterministic=True, project=str(SAL / 'yolo_runs'), name=f'{n}_f{k}', plots=False, exist_ok=True)
            if drop: kw['dropout'] = drop
            best = SAL / 'yolo_runs' / f'{n}_f{k}' / 'weights' / 'best.pt'
            if not (best.parent.parent / 'listo').exists():
                YOLO('yolo26s-cls.pt').train(**kw); (best.parent.parent / 'listo').touch()
            te = np.where(part[k] == 'test')[0]
            top1[te], prob[te] = C.predecir(best, [str(rutas[i]) for i in te])
            print(f'{n} fold {k}: MAE top1 {np.abs(top1[te] - d.bcs.to_numpy()[te]).mean():.3f} · prob {np.abs(prob[te] - d.bcs.to_numpy()[te]).mean():.3f} · {time.time() - t0:.0f} s', flush=True)
            gc.collect(); torch.mps.empty_cache()
        guardar(n + '_top1', top1); guardar(n + '_prob', prob)

if 'yolo_sam' in MODOS:
    yolo('sam', [DATOS / r for r in d.archivo_mascara_clahe])
if 'yolo_caja' in MODOS:
    import yolo_caja as C
    assert (C.rancho.archivo_limpio.to_numpy() == d.archivo_limpio.to_numpy()).all()
    rutas = list(C.rancho.crop)
    yolo('caja_con_ext', rutas, C.m[C.m.fuente == 'externo'][['crop', 'bcs']])
    yolo('caja_sin_ext', rutas)
