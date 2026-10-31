#!/usr/bin/env python3
"""Preserved historical fold-training helpers.

The selected definitions are copied without structural changes from
the inspected long_window_cv.py.

Historical source SHA-256:
aa56d8770c01936a6bcb61a3ee2977fbfd17c446a5aa2162f6f5c5c8756f79e7

Importing this module does not load data or launch training.

Important historical behavior:
- Fusion channel order is noisy, then OMSS.
- Inner validation split uses random_state=42 regardless of training seed.
- Macro recall includes all four labels, with zero_division=0.
- Existing prediction filenames cause run_fold to skip that run.
- Prediction filenames do not encode frame count or training settings.
  Use a separate output directory for each experiment configuration.
- The caller must validate data alignment and group separation.
- The caller must provide out_dir; the preserved function retains its
  original default path.
"""

import gc
import time
from pathlib import Path

import numpy as np
from tensorflow import keras
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedGroupKFold

if __package__:
    from .models import make
else:
    from models import make

class ValUAR(keras.callbacks.Callback):
    def __init__(self, X, y): super().__init__(); self.X, self.y = X, y
    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.X, batch_size=64, verbose=0).argmax(1)
        logs['val_uar'] = recall_score(self.y, p, average='macro', labels=range(4), zero_division=0)

def feats(name, rows, XC):
    if name.startswith('fusion'):
        return np.concatenate([XC['noisy'][rows], XC['omss'][rows]], axis=-1)
    return XC['omss' if name.startswith('omss') else 'noisy'][rows]

def uar(y, p): return float(recall_score(y, p, average='macro', labels=range(4), zero_division=0))

def run_fold(name, fold, seed, T, full, XC, epochs=60, patience=15, out_dir='/content/long_cv'):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f = out / (f'cv_{name}_f{fold}.npz' if seed == 42 else f'cv_{name}_f{fold}_s{seed}.npz')
    if f.exists(): return {'name': name, 'fold': fold, 'seed': seed, 'skipped': True}
    keras.utils.set_random_seed(seed)
    tr_all = full[full.fold != fold]; te = full[full.fold == fold]
    g = tr_all.groupby('OriginalID').agg(y=('y', 'first')).reset_index()
    sub = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    _, va_idx = next(sub.split(g, g.y, groups=g.OriginalID))
    va_ids = set(g.OriginalID.iloc[va_idx])
    va = tr_all[tr_all.OriginalID.isin(va_ids)]; tr = tr_all[~tr_all.OriginalID.isin(va_ids)]
    Xtr, ytr = feats(name, tr.row.values, XC), tr.y.values
    Xva, yva = feats(name, va.row.values, XC), va.y.values
    Xte, yte = feats(name, te.row.values, XC), te.y.values
    cw = len(ytr) / (4 * np.maximum(np.bincount(ytr, minlength=4), 1))
    model = make(name, T)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    cbs = [ValUAR(Xva, yva),
           keras.callbacks.ReduceLROnPlateau('val_uar', mode='max', factor=0.5, patience=5, verbose=0),
           keras.callbacks.EarlyStopping('val_uar', mode='max', patience=patience, restore_best_weights=True)]
    t0 = time.time()
    h = model.fit(Xtr, keras.utils.to_categorical(ytr, 4), epochs=epochs, batch_size=32,
                  validation_data=(Xva, keras.utils.to_categorical(yva, 4)),
                  class_weight=dict(enumerate(cw)), callbacks=cbs, verbose=0)
    prob = model.predict(Xte, batch_size=64, verbose=0)
    np.savez(f, prob=prob, y=yte, ids=te.OriginalID.values, epochs=len(h.history['loss']))
    r = {'name': name, 'fold': fold, 'seed': seed, 'params': int(model.count_params()),
         'epochs': len(h.history['loss']), 'sec': round(time.time() - t0, 1), 'test_uar': round(uar(yte, prob.argmax(1)), 3)}
    keras.backend.clear_session(); gc.collect()
    return r
