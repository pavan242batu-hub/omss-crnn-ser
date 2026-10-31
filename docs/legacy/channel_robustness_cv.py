
import numpy as np, pandas as pd, tensorflow as tf, time, gc
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedGroupKFold
tf.get_logger().setLevel('ERROR')

BASE = Path('/content/drive/MyDrive/SERRevision'); TR = BASE / 'SERRevision_Transfer'
FOLDS = BASE / 'OMSS-attention integration' / 'results' / 'cv' / 'cv_fold_assignment.csv'

def load_data(T):
    fc = BASE / 'long_window' / 'feature_cache'
    full = pd.read_csv(TR / 'omss_grouped_manifest.csv')
    fa = pd.read_csv(FOLDS).set_index('OriginalID')
    full['fold'] = full.OriginalID.map(fa['fold']); full['y'] = full.OriginalID.map(fa['y'])
    assert full.fold.notna().all() and full.y.notna().all()
    full['fold'] = full.fold.astype(int); full['y'] = full.y.astype(int)
    full['row'] = np.arange(len(full))
    XC = {c: np.load(fc / f'mfcc_{c}_T{T}.npz')['X'] for c in ('omss', 'noisy')}
    return full, XC

class SpecAug(layers.Layer):
    def __init__(self, tmax=16, fmax=6, T=128, F=40, **kw):
        super().__init__(**kw); self.tmax, self.fmax, self.T, self.F = tmax, fmax, T, F
    def call(self, x, training=None):
        if not training: return x
        B = tf.shape(x)[0]
        def mask(n, mx):
            w = tf.random.uniform([B, 1], 0, mx + 1, dtype=tf.int32)
            s = tf.random.uniform([B, 1], 0, n, dtype=tf.int32)
            i = tf.range(n)[None, :]
            return tf.logical_not((i >= s) & (i < s + w))
        m = mask(self.T, self.tmax)[:, :, None] & mask(self.F, self.fmax)[:, None, :]
        return x * tf.cast(m, x.dtype)[..., None]
    def get_config(self):
        return {**super().get_config(), 'tmax': self.tmax, 'fmax': self.fmax, 'T': self.T, 'F': self.F}

class TemporalAttention(layers.Layer):
    def __init__(self, units=64, **kw):
        super().__init__(**kw); self.units = units
    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name='W', shape=(d, self.units), initializer='glorot_uniform')
        self.b = self.add_weight(name='b', shape=(self.units,), initializer='zeros')
        self.v = self.add_weight(name='v', shape=(self.units, 1), initializer='glorot_uniform')
        super().build(input_shape)
    def call(self, h):
        s = tf.matmul(tf.tanh(tf.matmul(h, self.W) + self.b), self.v)
        a = tf.nn.softmax(s, axis=1)
        return tf.reduce_sum(h * a, axis=1)

class GuidedAttention(layers.Layer):
    def __init__(self, units=64, use_cue=True, **kw):
        super().__init__(**kw); self.units, self.use_cue = units, use_cue
    def build(self, shapes):
        d = shapes[0][-1]
        self.W = self.add_weight(name='W', shape=(d, self.units), initializer='glorot_uniform')
        self.b = self.add_weight(name='b', shape=(self.units,), initializer='zeros')
        self.v = self.add_weight(name='v', shape=(self.units, 1), initializer='glorot_uniform')
        self.lam = self.add_weight(name='lam', shape=(1,), initializer='zeros')
        super().build(shapes)
    def call(self, inputs):
        h, cue = inputs
        s = tf.matmul(tf.tanh(tf.matmul(h, self.W) + self.b), self.v)
        if self.use_cue: s = s + self.lam * cue
        a = tf.nn.softmax(s, axis=1)
        return tf.reduce_sum(h * a, axis=1)

class ValUAR(keras.callbacks.Callback):
    def __init__(self, X, y): super().__init__(); self.X, self.y = X, y
    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.X, batch_size=64, verbose=0).argmax(1)
        logs['val_uar'] = recall_score(self.y, p, average='macro', labels=range(4), zero_division=0)

def build_single(attention, T, l2=1e-3, drop=0.4):
    reg = regularizers.l2(l2)
    inp = keras.Input((T, 40, 1))
    x = SpecAug(tmax=round(16 * T / 128), T=T)(inp)
    x = layers.GaussianNoise(0.05)(x)
    for f in (32, 64):
        x = layers.Conv2D(f, 3, padding='same', kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPool2D(2)(x); x = layers.Dropout(0.2)(x)
    x = layers.Reshape((T // 4, 10 * 64))(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(drop)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, kernel_regularizer=reg))(x)
    x = TemporalAttention()(x) if attention else layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(drop)(x)
    return keras.Model(inp, layers.Dense(4, activation='softmax', kernel_regularizer=reg)(x))

def build_fusion(variant, T, l2=1e-3, drop=0.4):
    reg = regularizers.l2(l2)
    inp = keras.Input((T, 40, 2))
    x = SpecAug(tmax=round(16 * T / 128), T=T)(inp)
    x = layers.GaussianNoise(0.05)(x)
    xn = layers.Lambda(lambda t: t[..., :1])(x)
    xo = layers.Lambda(lambda t: t[..., 1:])(x)
    cue = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([xo, xn])
    cue = layers.AveragePooling2D((4, 40))(cue)
    cue = layers.Reshape((T // 4, 1))(cue)
    cue = layers.BatchNormalization()(cue)
    blocks = [layers.Conv2D(32, 3, padding='same', kernel_regularizer=reg), layers.BatchNormalization(),
              layers.ReLU(), layers.MaxPool2D(2), layers.Dropout(0.2),
              layers.Conv2D(64, 3, padding='same', kernel_regularizer=reg), layers.BatchNormalization(),
              layers.ReLU(), layers.MaxPool2D(2), layers.Dropout(0.2),
              layers.Reshape((T // 4, 640)), layers.Dense(128, activation='relu', kernel_regularizer=reg)]
    def enc(t):
        for l in blocks: t = l(t)
        return t
    hn, ho = enc(xn), enc(xo)
    g = layers.Dense(1, activation='sigmoid')(layers.Concatenate()([hn, ho, cue]))
    h = layers.Lambda(lambda t: t[0] * t[1] + (1 - t[0]) * t[2])([g, ho, hn])
    h = layers.Dropout(drop)(h)
    h = layers.Bidirectional(layers.GRU(64, return_sequences=True, kernel_regularizer=reg))(h)
    if variant == 'fusion_gap': z = layers.GlobalAveragePooling1D()(h)
    else: z = GuidedAttention(use_cue=(variant == 'fusion_guided'))([h, cue])
    z = layers.Dropout(drop)(z)
    return keras.Model(inp, layers.Dense(4, activation='softmax', kernel_regularizer=reg)(z))

def make(name, T):
    if name.startswith('fusion'): return build_fusion(name, T)
    return build_single(name.endswith('_att'), T)

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
    weights_path = out / f'{name}_f{fold}_s{seed}_T{T}.weights.h5'
    model.save_weights(str(weights_path))
    print(f'Saved trained weights: {weights_path}', flush=True)
    prob = model.predict(Xte, batch_size=64, verbose=0)
    np.savez(f, prob=prob, y=yte, ids=te.OriginalID.values, epochs=len(h.history['loss']))
    r = {'name': name, 'fold': fold, 'seed': seed, 'params': int(model.count_params()),
         'epochs': len(h.history['loss']), 'sec': round(time.time() - t0, 1), 'test_uar': round(uar(yte, prob.argmax(1)), 3)}
    keras.backend.clear_session(); gc.collect()
    return r
