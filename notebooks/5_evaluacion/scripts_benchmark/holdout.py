"""Train/test único: E0, Ridge/MLP sobre características congeladas (alpha elegido en val) y E1 (entrena solo con train)."""
import glob, os, sys, time
import numpy as np, pandas as pd
from datos import cargar, RAIZ
import ridge as R          # reutiliza ajustar/pred_de/ALPHAS (Ridge); el MLP se arma con TIPO='mlp'

SAL = RAIZ / 'holdout'
d = cargar().merge(pd.read_csv(SAL / 'split_holdout.csv'), on='archivo_limpio', validate='1:1')
assert (d['archivo_limpio'].to_numpy() == R.d['archivo_limpio'].to_numpy()).all()
tr, va, te = (np.where(d['split'] == s)[0] for s in ['train', 'val', 'test'])
Y, PESO = R.Y, R.PESO
res = {'E0_constante': np.full(len(d), d.iloc[tr].groupby('id_vaca')['bcs'].first().mean())}

for tipo, alphas in [('ridge', np.logspace(-1, 5, 25)), ('mlp', np.logspace(-2, 2, 5))]:
    R.TIPO = tipo
    for f in sorted(glob.glob(str(RAIZ / 'cache' / '*.npz'))):
        nombre = os.path.basename(f)[:-4]
        if tipo == 'mlp' and not nombre.startswith('dinov2_l'): continue
        X = np.load(f, allow_pickle=True)['X']
        err = [R.mae_w(Y[va], R.pred_de(R.ajustar(X, tr, a), X[va]), PESO[va]) for a in alphas]
        a = alphas[int(np.argmin(err))]
        nombre = ('E4_mlp_' if tipo == 'mlp' else 'ridge_') + nombre
        res[nombre] = R.pred_de(R.ajustar(X, tr, a), X)
        print(f'{nombre:48s} alpha {a:8.1f} · MAE val {min(err):.3f} · MAE test {np.abs(res[nombre][te] - Y[te]).mean():.3f}', flush=True)

if '--sin-e1' not in sys.argv:
    import torch, e1_cnn as E
    t0 = time.time()
    m, mu, sd = E.entrenar(d.iloc[tr], E.SEMILLA)
    res['E1_cnn_afinada'] = E.predecir(m, torch.arange(len(d)), mu, sd)
    print(f'E1 MAE test {np.abs(res["E1_cnn_afinada"][te] - Y[te]).mean():.3f} · train {np.abs(res["E1_cnn_afinada"][tr] - Y[tr]).mean():.3f} · {time.time() - t0:.0f} s', flush=True)

pd.DataFrame(res).assign(archivo_limpio=d['archivo_limpio'], id_vaca=d['id_vaca'], animal=d['animal'], bcs=Y,
                         split=d['split']).to_csv(SAL / 'pred_holdout.csv', index=False)
