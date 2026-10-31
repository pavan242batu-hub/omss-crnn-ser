#!/usr/bin/env python3
"""GRU-based single-stream and noisy/OMSS fusion model definitions.

Extracted from the inspected long_window_cv.py without altering
the selected class and function definitions.

Contains no audio loading, feature-cache loading, or training launch.

Historical source SHA-256:
aa56d8770c01936a6bcb61a3ee2977fbfd17c446a5aa2162f6f5c5c8756f79e7

This file preserves the original model behavior, including the
original augmentation, shared-layer handling, and attention weights.
It does not establish linkage to every historical manuscript result.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

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
