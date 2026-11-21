"""Partición única train/val/test por animal, con grouped_holdout de los notebooks YOLO (semilla 42)."""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from datos import cargar, RAIZ

SEMILLA, TEST_FRACTION, VAL_WITHIN_DEVELOPMENT = 42, 0.20, 0.20

def grouped_holdout(frame, fraction, seed, extra_training_scores=()):   # copia del notebook
    frame = frame.reset_index(drop=True)
    classes = sorted(frame['bcs'].unique())
    overall = frame['bcs'].value_counts(normalize=True).reindex(classes, fill_value=0)
    best = None
    for fit_idx, held_idx in GroupShuffleSplit(n_splits=2000, test_size=fraction, random_state=seed).split(frame, groups=frame['animal_id']):
        fit, held = frame.iloc[fit_idx], frame.iloc[held_idx]
        if not set(classes).issubset(set(fit['bcs']) | set(extra_training_scores)):
            continue
        dist = held['bcs'].value_counts(normalize=True).reindex(classes, fill_value=0)
        cost = float(np.abs(dist - overall).sum() + abs(len(held) / len(frame) - fraction))
        if best is None or cost < best[0]:
            best = (cost, fit.copy(), held.copy())
    return best[1], best[2]

if __name__ == '__main__':
    d = cargar().assign(animal_id=lambda x: x['animal'])
    dev, test = grouped_holdout(d, TEST_FRACTION, SEMILLA)
    fit, val = grouped_holdout(dev, VAL_WITHIN_DEVELOPMENT, SEMILLA)
    asig = {a: s for s, f in [('train', fit), ('val', val), ('test', test)] for a in f['animal'].unique()}
    d['split'] = d['animal'].map(asig)
    assert d.groupby('animal')['split'].nunique().max() == 1 and d['split'].notna().all()
    assert set(d['bcs']).issubset(set(d.loc[d.split == 'train', 'bcs']))
    d[['archivo_limpio', 'split']].to_csv(RAIZ / 'holdout' / 'split_holdout.csv', index=False)
    print(d.groupby('split').agg(fotos=('bcs', 'size'), animales=('animal', 'nunique'), sesiones=('id_vaca', 'nunique'), bcs_medio=('bcs', 'mean')))
    print(pd.crosstab(d['bcs'], d['split']))
