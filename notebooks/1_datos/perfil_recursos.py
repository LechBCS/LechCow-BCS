"""Mide tiempo, memoria y CPU de cada etapa del pipeline (evidencia para la decisión de no usar Big Data).

Ejecuta las celdas de código de un notebook del repo en un solo proceso, con algunas variables de
configuración reemplazadas, y mide cada celda: segundos, memoria máxima (RAM del proceso de Python más
la JVM de Spark), CPU promedio y, con PyTorch en MPS, memoria de la GPU. También trae una versión del
ETL en pandas en un solo proceso para comparar contra Spark local.

Uso (desde la raíz del repo, con el entorno del proyecto):
    python notebooks/1_datos/perfil_recursos.py etl     --fotos DIR --salida DIR
    python notebooks/1_datos/perfil_recursos.py pandas  --fotos DIR --salida DIR
    python notebooks/1_datos/perfil_recursos.py dinov2  --salida DIR
    python notebooks/1_datos/perfil_recursos.py consultas --processed DIR/processed --salida DIR

--fotos es una carpeta con el formato de ScienceDB (<bcs>/GS_<vaca>_<n>.jpg). Las salidas son solo
agregados (tiempos y memoria); las tablas y las imágenes se escriben en --salida, fuera del repo.
"""
import argparse, json, os, platform, re, sys, threading, time
from pathlib import Path

import psutil

RAIZ = Path(__file__).resolve().parents[2]


class Monitor:
    """Muestrea cada 0.2 s la memoria (RSS) del proceso y sus hijos, el CPU y la memoria de MPS."""

    def __init__(self):
        self.proc = psutil.Process()
        self.activo = False

    def _procesos(self):
        try:
            return [self.proc] + self.proc.children(recursive=True)
        except psutil.Error:
            return [self.proc]

    def _muestrear(self):
        while self.activo:
            rss, cpu = 0, 0.0
            for p in self._procesos():
                try:
                    rss += p.memory_info().rss
                    cpu += p.cpu_percent(None)
                except psutil.Error:
                    pass
            self.rss_max = max(self.rss_max, rss)
            self.cpu.append(cpu)
            torch = sys.modules.get('torch')
            try:  # torch puede estar a medio importar
                if torch is not None and torch.backends.mps.is_available():
                    self.gpu_max = max(self.gpu_max, torch.mps.driver_allocated_memory())
            except AttributeError:
                pass
            time.sleep(0.2)

    def __enter__(self):
        self.rss_max, self.gpu_max, self.cpu = 0, 0, []
        for p in self._procesos():
            p.cpu_percent(None)
        self.activo, self.t0 = True, time.perf_counter()
        self.hilo = threading.Thread(target=self._muestrear, daemon=True)
        self.hilo.start()
        return self

    def __exit__(self, *exc):
        self.segundos = time.perf_counter() - self.t0
        self.activo = False
        self.hilo.join()

    def fila(self, etapa, **extra):
        cpu = sum(self.cpu) / len(self.cpu) if self.cpu else 0.0
        return {'etapa': etapa, 'segundos': round(self.segundos, 2), 'ram_max_gb': round(self.rss_max / 2**30, 2),
                'cpu_prom_%': round(cpu), 'gpu_max_gb': round(self.gpu_max / 2**30, 2), **extra}


def correr_notebook(ruta, etapas, reemplazos, despues_de):
    """Ejecuta las celdas indicadas en orden; aplica `reemplazos` justo después de la celda `despues_de`."""
    celdas = json.loads(Path(ruta).read_text())['cells']
    ns, filas = {'__name__': '__main__'}, []
    for i, nombre in etapas:
        with Monitor() as m:
            exec(compile(''.join(celdas[i]['source']), f'{Path(ruta).name}:celda{i}', 'exec'), ns)
            if i == despues_de:
                exec(reemplazos, ns)
        filas.append(m.fila(nombre))
        print(filas[-1], flush=True)
    return filas, ns


def perfil_etl(fotos, salida):
    clases = sorted({float(d.name) for d in Path(fotos).iterdir() if d.is_dir()})
    reemplazos = f'DATA_PATH = {str(fotos)!r}\nOUT_PATH = {str(salida)!r}\nCLASES_BCS = {clases!r}\n'
    etapas = [(4, 'configuración'), (8, 'arranque de Spark'), (11, 'extracción (lectura binaria)'),
              (14, 'metadatos + SHA-256'), (17, 'limpieza'), (20, 'selección de vacas'),
              (23, 'k-fold por vaca'), (27, 'transformación de imágenes'), (30, 'carga a Parquet + verificación')]
    filas, ns = correr_notebook(RAIZ / 'notebooks/1_datos/ETL_BCS_PySpark.ipynb', etapas, reemplazos, 4)
    ns['spark'].stop()
    return filas, {'imagenes_leidas': int(ns['n_raw']), 'imagenes_validas': int(len(ns['filas']))}


def perfil_pandas(fotos, salida):
    """El mismo ETL sin Spark: un solo proceso con hashlib, pandas y Pillow."""
    import hashlib
    import pandas as pd
    from PIL import Image, ImageOps
    from sklearn.model_selection import StratifiedGroupKFold

    filas = []
    with Monitor() as m:
        reg = []
        for p in sorted(Path(fotos).rglob('*.jpg')):
            datos = p.read_bytes()
            c = re.match(r'^GS_(\d+)_\d+', p.name)
            reg.append({'image_id': hashlib.sha256(datos).hexdigest(), 'path': str(p), 'bcs': float(p.parent.name),
                        'cow_id': c.group(1) if c else None, 'length': len(datos)})
        df = pd.DataFrame(reg)
    filas.append(m.fila('extracción + metadatos + SHA-256'))
    with Monitor() as m:
        df['motivo_exclusion'] = None
        df.loc[df['cow_id'].isna(), 'motivo_exclusion'] = 'sin_id_vaca'
        df.loc[df.duplicated('image_id') & df['motivo_exclusion'].isna(), 'motivo_exclusion'] = 'duplicado'
        limpio = df[df['motivo_exclusion'].isna()].copy()
        vacas = limpio.groupby('cow_id', as_index=False)['bcs'].first()
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        vacas['fold'] = -1
        for f, (_, idx) in enumerate(sgkf.split(vacas, (vacas['bcs'] * 100).astype(int), vacas['cow_id'])):
            vacas.loc[idx, 'fold'] = f
        limpio = limpio.merge(vacas[['cow_id', 'fold']], on='cow_id')
    filas.append(m.fila('limpieza + k-fold'))
    with Monitor() as m:
        destino = Path(salida, 'clean_pandas')
        for r in limpio.itertuples():
            with Image.open(r.path) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((640, 640))
                (destino / f'bcs={r.bcs}').mkdir(parents=True, exist_ok=True)
                im.save(destino / f'bcs={r.bcs}' / f'{r.image_id[:16]}.jpg', 'JPEG', quality=92)
    filas.append(m.fila('transformación de imágenes'))
    with Monitor() as m:
        Path(salida, 'processed_pandas').mkdir(parents=True, exist_ok=True)
        df.to_parquet(Path(salida, 'processed_pandas', 'imagenes.parquet'), index=False)
        assert len(pd.read_parquet(Path(salida, 'processed_pandas', 'imagenes.parquet'))) == len(df)
    filas.append(m.fila('carga a Parquet + verificación'))
    for f in filas:
        print(f, flush=True)
    return filas, {'imagenes_leidas': len(df), 'imagenes_validas': len(limpio)}


def perfil_dinov2(salida):
    """Notebook del modelo aprobado sin caché: extracción de DINOv2-L en la GPU y entrenamiento de Ridge."""
    reemplazos = f'SALIDA = Path({str(salida)!r})\n'
    etapas = [(2, 'configuración'), (3, 'carga de librerías'), (5, 'lectura de recortes y filtros'),
              (7, 'partición train/val/test'), (9, 'extracción DINOv2-L (448 px, foto + espejo)'),
              (11, 'Ridge: búsqueda de alpha (25 valores) + ajuste'), (13, 'métricas'), (15, 'guardado del modelo')]
    filas, ns = correr_notebook(RAIZ / 'notebooks/4_modelado_dinov2/DINOv2_Ridge_BCS.ipynb', etapas, reemplazos, 2)
    return filas, {'fotos': int(len(ns['datos'])), 'dimension': int(ns['X'].shape[1]), 'dispositivo': ns['DISPOSITIVO']}


CONSULTAS = {
    'Q1 imágenes por motivo de exclusión': (
        "SELECT coalesce(motivo_exclusion, 'ok') AS estado, count(*) AS n FROM imagenes GROUP BY 1",
        lambda im, va: im['motivo_exclusion'].fillna('ok').value_counts()),
    'Q2 vacas e imágenes por fold y BCS': (
        "SELECT fold, bcs, count(DISTINCT cow_id) AS vacas, count(*) AS imagenes FROM imagenes "
        "WHERE fold IS NOT NULL GROUP BY fold, bcs",
        lambda im, va: im.dropna(subset=['fold']).groupby(['fold', 'bcs']).agg(vacas=('cow_id', 'nunique'),
                                                                                imagenes=('image_id', 'size'))),
    'Q3 duplicados por hash': (
        "SELECT image_id, count(*) AS copias FROM imagenes GROUP BY image_id HAVING count(*) > 1",
        lambda im, va: im.groupby('image_id').size().loc[lambda s: s > 1]),
    'Q4 join imágenes-vacas: fotos por vaca y BCS': (
        "SELECT v.bcs, avg(c.n) AS fotos_por_vaca FROM vacas v JOIN "
        "(SELECT cow_id, count(*) AS n FROM imagenes GROUP BY cow_id) c ON v.cow_id = c.cow_id GROUP BY v.bcs",
        lambda im, va: va[['cow_id', 'bcs']].merge(im.groupby('cow_id').size().rename('n').reset_index(), on='cow_id')
                         .groupby('bcs')['n'].mean()),
    'Q5 top 10 archivos más pesados por fold': (
        "SELECT * FROM (SELECT fold, image_id, length, row_number() OVER (PARTITION BY fold ORDER BY length DESC) AS r "
        "FROM imagenes WHERE fold IS NOT NULL) WHERE r <= 10",
        lambda im, va: im.dropna(subset=['fold']).sort_values('length', ascending=False).groupby('fold').head(10)),
}


def perfil_consultas(processed, salida, factores=(1, 10, 100, 1000), repeticiones=3):
    """Las 5 consultas típicas sobre las tablas del ETL, en Spark SQL y en pandas, con las tablas multiplicadas
    por cada factor (copias con image_id y cow_id distintos) para ver cómo crece el tiempo."""
    import statistics
    import pandas as pd
    from pyspark.sql import SparkSession

    im0 = pd.read_parquet(Path(processed, 'imagenes.parquet'))
    va0 = pd.read_parquet(Path(processed, 'vacas.parquet'))
    spark = (SparkSession.builder.master('local[*]').appName('BCS_consultas')
             .config('spark.sql.shuffle.partitions', '8').config('spark.driver.memory', '6g').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    filas = []
    for k in factores:
        copia = lambda df, cols: pd.concat([df.assign(**{c: df[c].astype(str) + f'_{i}' for c in cols}) for i in range(k)],
                                           ignore_index=True)
        im = copia(im0, ['image_id', 'cow_id']) if k > 1 else im0
        va = copia(va0, ['cow_id']) if k > 1 else va0
        ruta = Path(salida, f'tablas_x{k}')
        ruta.mkdir(parents=True, exist_ok=True)
        im.to_parquet(ruta / 'imagenes.parquet', index=False)
        va.to_parquet(ruta / 'vacas.parquet', index=False)
        mb = round(sum(p.stat().st_size for p in ruta.iterdir()) / 2**20, 1)
        for nombre, (sql, fn) in CONSULTAS.items():
            for motor in ('Spark SQL', 'pandas'):
                tiempos = []
                with Monitor() as m:
                    for _ in range(repeticiones):
                        t0 = time.perf_counter()
                        if motor == 'Spark SQL':   # cada corrida lee el Parquet desde disco, como lo haría un usuario
                            spark.read.parquet(str(ruta / 'imagenes.parquet')).createOrReplaceTempView('imagenes')
                            spark.read.parquet(str(ruta / 'vacas.parquet')).createOrReplaceTempView('vacas')
                            spark.sql(sql).collect()
                        else:
                            fn(pd.read_parquet(ruta / 'imagenes.parquet'), pd.read_parquet(ruta / 'vacas.parquet'))
                        tiempos.append(time.perf_counter() - t0)
                filas.append(m.fila(nombre, motor=motor, factor=k, filas=len(im), parquet_mb=mb,
                                    mediana_s=round(statistics.median(tiempos), 3)))
                print(filas[-1], flush=True)
    spark.stop()
    return filas, {'filas_reales': len(im0), 'factores': list(factores)}


def grafica_consultas(json_path, png_path):
    """Tiempo (mediana) de cada consulta contra el número de filas, Spark SQL vs pandas."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = json.loads(Path(json_path).read_text())['etapas']
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=170)
    colores = {'Spark SQL': '#c0612b', 'pandas': '#2b6cb0'}
    for (nombre, motor) in sorted({(f['etapa'], f['motor']) for f in r}):
        pts = sorted((f['filas'], f['mediana_s']) for f in r if f['etapa'] == nombre and f['motor'] == motor)
        ax.plot(*zip(*pts), marker='o', ms=3.5, lw=1.2, color=colores[motor], alpha=0.85,
                label=motor if nombre.startswith('Q1') else None)
        ax.annotate(nombre.split()[0], pts[-1], xytext=(4, 0), textcoords='offset points', fontsize=7,
                    color=colores[motor], va='center')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Filas de la tabla imagenes (escala log)')
    ax.set_ylabel('Tiempo por consulta, mediana de 3 (s, escala log)')
    ax.axvline(r[0]['filas'], color='#888888', ls=':', lw=1)
    ax.text(r[0]['filas'] * 1.1, ax.get_ylim()[0] * 1.3, 'tamaño real (732 filas)', fontsize=7, color='#666666')
    ax.grid(True, which='major', alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0.04, 0.98))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(png_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('perfil', choices=['etl', 'pandas', 'dinov2', 'consultas'])
    ap.add_argument('--processed', type=Path, help='carpeta processed/ que escribe el ETL (para consultas)')
    ap.add_argument('--factores', type=int, nargs='+', default=[1, 10, 100, 1000, 5000],
                    help='veces que se copian las tablas en el perfil de consultas')
    ap.add_argument('--fotos', type=Path)
    ap.add_argument('--salida', type=Path, required=True)
    a = ap.parse_args()
    a.salida.mkdir(parents=True, exist_ok=True)
    if a.perfil == 'dinov2':
        filas, extra = perfil_dinov2(a.salida)
    elif a.perfil == 'consultas':
        filas, extra = perfil_consultas(a.processed.resolve(), a.salida.resolve(), a.factores)
    else:
        filas, extra = (perfil_etl if a.perfil == 'etl' else perfil_pandas)(a.fotos.resolve(), a.salida.resolve())
    equipo = {'equipo': platform.machine(), 'cpu': os.cpu_count(), 'ram_gb': round(psutil.virtual_memory().total / 2**30),
              'python': platform.python_version()}
    resultado = {'perfil': a.perfil, 'equipo': equipo, **extra, 'etapas': filas,
                 'total_segundos': round(sum(f['segundos'] for f in filas), 1)}
    (a.salida / f'perfil_{a.perfil}.json').write_text(json.dumps(resultado, ensure_ascii=False, indent=1))
    if a.perfil == 'consultas':
        grafica_consultas(a.salida / 'perfil_consultas.json', a.salida / 'consultas.png')
    print(json.dumps({k: v for k, v in resultado.items() if k != 'etapas'}, ensure_ascii=False))
