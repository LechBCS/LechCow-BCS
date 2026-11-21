"""Extrae características congeladas (promedio foto + espejo) y las guarda en caché .npz."""
import sys, time
import numpy as np, torch, timm
from PIL import Image
from datos import cargar, DATOS, RAIZ

CACHE = RAIZ / 'cache'; CACHE.mkdir(exist_ok=True)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DESV = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
MODELOS = {'resnet50': 'resnet50.a1_in1k',
           'dinov2_b': 'vit_base_patch14_dinov2.lvd142m',
           'dinov2_l': 'vit_large_patch14_dinov2.lvd142m'}

def crear(corto, tam):
    kw = dict(img_size=tam) if 'dinov2' in corto else {}
    return timm.create_model(MODELOS[corto], pretrained=True, num_classes=0, **kw).eval().to(DEV)

@torch.no_grad()
def feats(m, corto, x):
    if 'dinov2' not in corto:
        return {'final': m(x)}
    t = m.forward_features(x)                         # [B, 1 + parches, D]
    cls, parches = t[:, 0], t[:, 1 + getattr(m, 'num_reg_tokens', 0):].mean(1)
    return {'cls': cls, 'cls_parches': torch.cat([cls, parches], 1)}

def main(corto, variante, tam=224, lote=16):
    d = cargar()
    m = crear(corto, tam)
    salida = {}
    t0 = time.time()
    for i in range(0, len(d), lote):
        imgs = [np.asarray(Image.open(DATOS / r).convert('RGB').resize((tam, tam), Image.BICUBIC))
                for r in d[f'archivo_{variante}'].iloc[i:i + lote]]
        x = (torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255 - MEDIA) / DESV
        x = x.to(DEV)
        a, b = feats(m, corto, x), feats(m, corto, torch.flip(x, dims=[3]))
        for k in a:
            salida.setdefault(k, []).append(((a[k] + b[k]) / 2).float().cpu().numpy())
    for k, v in salida.items():
        np.savez(CACHE / f'{corto}_{variante}_{k}_{tam}.npz', X=np.concatenate(v), archivo=d['archivo_limpio'].to_numpy())
        print(f'{corto} {variante} {k}: {np.concatenate(v).shape} · {time.time() - t0:.0f} s')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
