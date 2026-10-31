# Recovered historical notebook cell; not a portable CLI.
# Source notebook: SERRevision_IoT_Eval_v1.ipynb
# Source cell number at extraction: 98
# Preserved for inspection. Executing this file can write caches.
# Historical output equivalence has not yet been verified.

import numpy as np, pandas as pd, librosa
from pathlib import Path
from tqdm.auto import tqdm
from joblib import Parallel, delayed

BASE = Path('/content/drive/MyDrive/SERRevision'); TR = BASE / 'SERRevision_Transfer'
OUT = BASE / 'long_window' / 'feature_cache'; OUT.mkdir(parents=True, exist_ok=True)
T = 400
full = pd.read_csv(TR / 'omss_grouped_manifest.csv')
idx = {c: {p.name: p for p in (TR / c).rglob('*.wav')} for c in ('omss', 'noisy')}

def mfcc_long(path, T=T):
    y, _ = librosa.load(str(path), sr=16000, mono=True)
    m = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40, n_fft=512, win_length=512,
                             hop_length=160, n_mels=64, center=True).T
    m = (m - m.mean(0, keepdims=True)) / (m.std(0, keepdims=True) + 1e-8)
    m = np.pad(m, ((0, T - len(m)), (0, 0))) if len(m) < T else m[:T]
    return m.astype(np.float32)[..., None], len(y) / 16000

for c in ('omss', 'noisy'):
    out = OUT / f'mfcc_{c}_T{T}.npz'
    if out.exists():
        print('exists:', out.name); continue
    paths = [idx[c][f] for f in full.FileName]
    res = Parallel(n_jobs=2)(delayed(mfcc_long)(p) for p in tqdm(paths, desc=c))
    X = np.stack([r[0] for r in res]); dur = np.array([r[1] for r in res])
    np.savez(out, X=X, dur=dur); print(c, X.shape)

old = np.load(BASE / 'full_ablation_results/feature_cache/mfcc_omss_d5cadcfaa10e9199.npz')['X']
new = np.load(OUT / f'mfcc_omss_T{T}.npz')
print('old', old.shape, 'new', new['X'].shape)
print('first 128 frames identical to old cache:', np.allclose(old, new['X'][:, :128], atol=1e-4))
dur = new['dur']
print('clips with more than %d frames: %.0f%%' % (T, 100 * (dur * 100 > T).mean()))