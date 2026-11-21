"""E1: receta 05 de Modelo_CNN_Vacuno (ResNet-50 afinada, layer4 + cabeza, augmentation), en MPS."""
import math, sys, time
import numpy as np, pandas as pd, timm, torch
import torch.nn.functional as F
from PIL import Image
from datos import cargar, DATOS, RAIZ, K

SEMILLA_FOLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 42
VARIANTE, EXTRACTOR, TAM = 'archivo_mascara_clahe', 'resnet50.a1_in1k', 224
SEMILLA = 42
ENTRENABLES = ('layer4', 'fc')
EPOCAS, LOTE = 15, 32
LR_CAPAS, LR_CABEZA, WEIGHT_DECAY, DROPOUT = 1e-4, 1e-3, 1e-2, 0.2
AUG = dict(volteo=0.5, giro_grados=8, recorte_min=0.90, brillo=0.15, contraste=0.15)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'

datos = cargar(SEMILLA_FOLDS)
IMGS = torch.from_numpy(np.stack([np.asarray(Image.open(DATOS / r).convert('RGB').resize((TAM, TAM), Image.BICUBIC))
                                  for r in datos[VARIANTE]])).permute(0, 3, 1, 2).contiguous()
MEDIA = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
DESV = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

def aumentar(x, gen):
    B, A = x.shape[0], AUG
    u = lambda: torch.rand(B, generator=gen)
    voltear = u() < A['volteo']
    x = torch.where(voltear.to(x.device).view(B, 1, 1, 1), torch.flip(x, dims=[3]), x)
    ang = (u() * 2 - 1) * A['giro_grados'] * math.pi / 180
    esc = A['recorte_min'] + u() * (1 - A['recorte_min'])
    tx, ty = (u() * 2 - 1) * (1 - esc), (u() * 2 - 1) * (1 - esc)
    theta = torch.stack([torch.stack([torch.cos(ang) * esc, -torch.sin(ang) * esc, tx], 1),
                         torch.stack([torch.sin(ang) * esc, torch.cos(ang) * esc, ty], 1)], 1).to(x.device, x.dtype)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), mode='bilinear', padding_mode='border', align_corners=False)
    brillo = (1 + (u() * 2 - 1) * A['brillo']).to(x.device).view(B, 1, 1, 1)
    contraste = (1 + (u() * 2 - 1) * A['contraste']).to(x.device).view(B, 1, 1, 1)
    media = x.mean(dim=(1, 2, 3), keepdim=True)
    return (((x - media) * contraste + media) * brillo).clamp(0, 1)

def lote_normalizado(idx, gen=None):
    x = IMGS[idx].to(DEV).float() / 255
    if gen is not None: x = aumentar(x, gen)
    return (x - MEDIA) / DESV

def crear_modelo():
    m = timm.create_model(EXTRACTOR, pretrained=True, num_classes=1, drop_rate=DROPOUT)
    torch.nn.init.zeros_(m.fc.weight); torch.nn.init.zeros_(m.fc.bias)
    for n, p in m.named_parameters(): p.requires_grad = n.startswith(ENTRENABLES)
    return m.to(DEV)

def modo_entrenamiento(m):
    m.train()
    for n, c in m.named_children():
        if not n.startswith(ENTRENABLES): c.eval()

def entrenar(tr, semilla):
    torch.manual_seed(semilla); gen = torch.Generator().manual_seed(semilla)
    idx = torch.tensor(tr.index.to_numpy())
    ses = tr.groupby('id_vaca')['bcs'].first(); mu, sd = float(ses.mean()), float(ses.std())
    z = torch.tensor(((tr['bcs'] - mu) / sd).to_numpy(), dtype=torch.float32, device=DEV)
    w = torch.tensor((1 / tr.groupby('id_vaca')['id_vaca'].transform('size')).to_numpy(), dtype=torch.float32, device=DEV)
    m = crear_modelo()
    capas = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith('fc')]
    opt = torch.optim.AdamW([{'params': capas, 'lr': LR_CAPAS}, {'params': m.fc.parameters(), 'lr': LR_CABEZA}], weight_decay=WEIGHT_DECAY)
    pasos = len(idx) // LOTE
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCAS * pasos)
    for _ in range(EPOCAS):
        modo_entrenamiento(m)
        orden = torch.randperm(len(idx), generator=gen)
        for i in range(pasos):
            pos = orden[i * LOTE:(i + 1) * LOTE]
            sal = m(lote_normalizado(idx[pos], gen)).squeeze(1)
            wb, zb = w[pos.to(DEV)], z[pos.to(DEV)]
            loss = (wb * (sal - zb) ** 2).sum() / wb.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return m, mu, sd

@torch.no_grad()
def predecir(m, idx, mu, sd, lote=64):
    m.eval(); out = []
    for i in range(0, len(idx), lote):
        x = lote_normalizado(idx[i:i + lote])
        out.append(((m(x).squeeze(1) + m(torch.flip(x, dims=[3])).squeeze(1)) / 2).cpu())
    return mu + sd * torch.cat(out).numpy()

if __name__ == '__main__':
    pred, t0 = np.full(len(datos), np.nan), time.time()
    for k in range(1, K + 1):
        tr, te = datos[datos['fold'] != k], datos[datos['fold'] == k]
        m, mu, sd = entrenar(tr, SEMILLA + k)
        pred[te.index] = predecir(m, torch.tensor(te.index.to_numpy()), mu, sd)
        ptr = predecir(m, torch.tensor(tr.index.to_numpy()), mu, sd)
        print(f'folds {SEMILLA_FOLDS} vuelta {k}: MAE prueba {np.abs(pred[te.index] - te.bcs).mean():.3f} · '
              f'entrenamiento {np.abs(ptr - tr.bcs).mean():.3f} · {time.time() - t0:.0f} s', flush=True)
        del m; torch.mps.empty_cache()
    pd.DataFrame({'archivo_limpio': datos['archivo_limpio'], 'E1_cnn_afinada': pred}).to_csv(RAIZ / 'exp' / f'pred_e1_f{SEMILLA_FOLDS}.csv', index=False)
