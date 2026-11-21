import warnings
warnings.filterwarnings("ignore")
"""E0 (constante) y E2/E3 (características congeladas + Ridge) con los folds de Charbel."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import sys
from datos import cargar, K, SEMILLA, RAIZ

CACHE = RAIZ / 'cache'
SEMILLA_FOLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 42
TIPO = sys.argv[2] if len(sys.argv) > 2 else 'ridge'
FILTRO = sys.argv[3] if len(sys.argv) > 3 else ''
ALPHAS = np.logspace(-1, 5, 25) if TIPO == 'ridge' else np.logspace(-2, 2, 5)
d = cargar(SEMILLA_FOLDS)
PESO = (1 / d.groupby('id_vaca')['bcs'].transform('size')).to_numpy()   # cada sesión pesa 1
Y, FOLD, ANIMAL = d['bcs'].to_numpy(), d['fold'].to_numpy(), d['animal'].to_numpy()

def mae_w(y, p, w): return np.sum(w * np.abs(y - p)) / np.sum(w)

def ajustar(X, idx, alpha):
    if TIPO == 'ridge':
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        m.fit(X[idx], Y[idx], ridge__sample_weight=PESO[idx])
    else:   # E4: cabeza densa (PCA 256 -> 256 -> 1), L2 = alpha
        m = make_pipeline(StandardScaler(), PCA(256, random_state=SEMILLA),
                          MLPRegressor(hidden_layer_sizes=(256,), alpha=alpha, learning_rate_init=1e-3, max_iter=300, random_state=SEMILLA))
        mu = Y[idx].mean()
        m.fit(X[idx], Y[idx] - mu, mlpregressor__sample_weight=PESO[idx])
        m.mu = mu
    return m

def pred_de(m, X):
    return m.predict(X) + getattr(m, 'mu', 0.0)

def elegir_alpha(X, idx):
    # folds internos por animal, solo con el entrenamiento de la vuelta
    anim = np.unique(ANIMAL[idx])
    interno = {a: i % 4 for i, a in enumerate(np.random.default_rng(SEMILLA).permutation(anim))}
    g = np.array([interno[a] for a in ANIMAL[idx]])
    errs = []
    for a in ALPHAS:
        e = [mae_w(Y[idx][g == k], pred_de(ajustar(X, idx[g != k], a), X[idx][g == k]), PESO[idx][g == k]) for k in range(4)]
        errs.append(np.mean(e))
    return ALPHAS[int(np.argmin(errs))]

def validar(X):
    pred, alphas = np.zeros(len(d)), []
    for k in range(1, K + 1):
        tr, te = np.where(FOLD != k)[0], np.where(FOLD == k)[0]
        a = elegir_alpha(X, tr); alphas.append(a)
        pred[te] = pred_de(ajustar(X, tr, a), X[te])
    return pred, alphas

def constante():
    pred = np.zeros(len(d))
    for k in range(1, K + 1):
        tr = FOLD != k
        pred[FOLD == k] = d[tr].groupby('id_vaca')['bcs'].first().mean()
    return pred

if __name__ == '__main__':
    import glob, os, sys
    res = {'E0_constante': constante()}
    for f in sorted(glob.glob(str(CACHE / f'*{FILTRO}*.npz'))):
        z = np.load(f, allow_pickle=True)
        assert (z['archivo'] == d['archivo_limpio'].to_numpy()).all()
        nombre = ('E4_mlp_' if TIPO != 'ridge' else '') + os.path.basename(f)[:-4]
        res[nombre], al = validar(z['X'])
        print(f'{nombre:42s} MAE foto {np.abs(res[nombre] - Y).mean():.3f} · alphas {[f"{a:.0f}" for a in al]}', flush=True)
    pd.DataFrame(res).assign(archivo_limpio=d['archivo_limpio'], id_vaca=d['id_vaca'], animal=d['animal'],
                             bcs=Y, fold=FOLD).to_csv(RAIZ / 'exp' / f'pred_{TIPO}_f{SEMILLA_FOLDS}{"_" + FILTRO if FILTRO else ""}.csv', index=False)
