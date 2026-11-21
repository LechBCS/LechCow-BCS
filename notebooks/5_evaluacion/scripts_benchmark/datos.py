"""Carga los 604 recortes y reproduce los folds de Modelo_CNN_Vacuno (por animal, semilla 42)."""
import re
from pathlib import Path
import numpy as np, pandas as pd

RAIZ = Path.home() / 'bcs_data'
DATOS = RAIZ / 'nube' / 'datos'
K, SEMILLA = 5, 42

def a_bool(v): return str(v).strip().lower() in {'true', '1', 'verdadero', 'si', 'sí', 'yes', 'x'}
def a_relativa(r):
    m = re.search(r'(?:^|/)(clean/.+)$', str(r).replace('\\', '/'))
    return m.group(1) if m else None
def limpiar_id(s): return s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

def cargar(semilla_folds=SEMILLA):
    rec = pd.read_csv(DATOS / 'recortes' / 'recortes.csv', dtype={'id_vaca': str, 'animal': str})
    tab = pd.read_csv(RAIZ / 'imagenes.csv')
    tab['archivo_limpio'] = tab['archivo_limpio'].map(a_relativa)
    tab = tab.dropna(subset=['archivo_limpio']).drop_duplicates('archivo_limpio')
    tab['animal'], tab['usar_tabla'] = limpiar_id(tab['animal']), tab['usar'].map(a_bool)
    rec = rec.drop(columns=['animal', 'ronda'], errors='ignore').merge(
        tab[['archivo_limpio', 'animal', 'ronda', 'usar_tabla']], on='archivo_limpio', how='left')
    rec['id_vaca'] = limpiar_id(rec['id_vaca'])
    d = rec[(rec['estado'] == 'ok') & (rec['fuente'] == 'rancho') & (rec['vista'].fillna('') != 'lateral')
            & (rec['usar_tabla'] == True)].reset_index(drop=True).copy()
    d['ronda'] = d['ronda'].astype(int)
    d['bcs'] = d['bcs'].astype(float)
    animales = sorted(d['animal'].unique())
    orden = np.random.default_rng(semilla_folds).permutation(animales)
    fold = {a: i % K + 1 for i, a in enumerate(orden)}
    d['fold'] = d['animal'].map(fold)
    assert d.groupby('animal')['fold'].nunique().max() == 1
    return d
