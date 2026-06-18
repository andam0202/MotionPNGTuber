"""numpy/scipy のみで実装した MFCC 特徴 + kNN による母音(あ/い/う/え/お)判定。

フォルマント2点だけのピーク拾いは話者/フレームで揺れて a/o/u が紛れやすい。
MFCC（スペクトル包絡を十数次元で表現）に置き換え、個人の声で学習した kNN で
判定することで分離が大きく改善する。librosa/scikit-learn は使わず軽量・低遅延。

- 特徴: MFCC c1..c12（c0=音量は捨てて声の大小に不変）をフレーム平均。
- 学習: train_vowel_mfcc.py が各母音を録音→フレーム特徴を全て学習サンプル化。
- 推論: ライブ窓(約40ms)から1ベクトル→ kNN 多数決でラベル。
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct

# 母音 → 口スプライト形状（formant_vowel と同一規約）
from .formant_vowel import VOWEL_TO_SHAPE, vowel_to_shape  # noqa: F401  re-export

N_MFCC = 13          # DCT 後に保持する係数（c0 を含む）
USE_COEFFS = slice(1, 13)  # c1..c12 を特徴に使う（c0=音量は除外）
N_MELS = 26
FMIN = 50.0
FRAME_SEC = 0.025    # 25ms 窓
HOP_SEC = 0.010      # 10ms ホップ


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS,
                    fmin: float = FMIN, fmax: float | None = None) -> np.ndarray:
    """三角メルフィルタバンク (n_mels, n_fft//2+1) を返す。"""
    if fmax is None:
        fmax = sr / 2.0
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        lo, ce, hi = bins[m - 1], bins[m], bins[m + 1]
        if ce > lo:
            fb[m - 1, lo:ce] = (np.arange(lo, ce) - lo) / (ce - lo)
        if hi > ce:
            fb[m - 1, ce:hi] = (hi - np.arange(ce, hi)) / (hi - ce)
    return fb


_FB_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _get_fb(sr: int, n_fft: int) -> np.ndarray:
    key = (sr, n_fft)
    fb = _FB_CACHE.get(key)
    if fb is None:
        fb = _mel_filterbank(sr, n_fft)
        _FB_CACHE[key] = fb
    return fb


def frame_mfccs(x: np.ndarray, sr: int) -> np.ndarray:
    """波形を 25ms/10ms でフレーム化し、各フレームの MFCC(c1..c12) を返す。

    戻り値 shape=(n_frames, 12)。短すぎる場合は (0,12)。
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    if x.size < 64:
        return np.zeros((0, 12), dtype=np.float64)
    # プリエンファシス
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    flen = max(256, int(sr * FRAME_SEC))
    hop = max(128, int(sr * HOP_SEC))
    n_fft = 1
    while n_fft < flen:
        n_fft <<= 1
    win = np.hamming(flen)
    fb = _get_fb(sr, n_fft)
    feats: list[np.ndarray] = []
    if x.size < flen:
        x = np.pad(x, (0, flen - x.size))
    for i in range(0, x.size - flen + 1, hop):
        frame = x[i:i + flen] * win
        spec = np.fft.rfft(frame, n=n_fft)
        power = (np.abs(spec) ** 2) / n_fft
        mel = fb @ power
        logmel = np.log(mel + 1e-10)
        cc = dct(logmel, type=2, norm="ortho")[:N_MFCC]
        feats.append(cc[USE_COEFFS])
    if not feats:
        return np.zeros((0, 12), dtype=np.float64)
    return np.asarray(feats, dtype=np.float64)


def extract_feature(x: np.ndarray, sr: int) -> np.ndarray | None:
    """ライブ用: 窓全体の MFCC をフレーム平均した 1 ベクトル(12,) を返す。"""
    m = frame_mfccs(x, sr)
    if m.shape[0] == 0:
        return None
    return m.mean(axis=0)


class KNNVowel:
    """numpy 実装の標準化付き kNN 母音分類器。.npz で保存/読込。"""

    def __init__(self, X: np.ndarray, y: np.ndarray, labels: list[str],
                 mean: np.ndarray, std: np.ndarray, k: int = 7) -> None:
        self.X = X            # (N, D) 標準化済み
        self.y = y            # (N,) int ラベルインデックス
        self.labels = labels  # idx→母音文字列
        self.mean = mean
        self.std = std
        self.k = k

    @classmethod
    def train(cls, samples: dict[str, np.ndarray], k: int = 7) -> "KNNVowel":
        """samples[母音] = (n, 12) 特徴行列 から学習。"""
        labels = sorted(samples.keys())
        Xs, ys = [], []
        for idx, lab in enumerate(labels):
            f = np.asarray(samples[lab], dtype=np.float64)
            if f.ndim == 1:
                f = f[None, :]
            Xs.append(f)
            ys.append(np.full(f.shape[0], idx))
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        Xn = (X - mean) / std
        return cls(Xn, y, labels, mean, std, k=k)

    def predict(self, vec: np.ndarray) -> tuple[str | None, float]:
        """特徴ベクトル(12,) → (母音, 信頼度0..1)。"""
        if vec is None or self.X.shape[0] == 0:
            return None, 0.0
        q = (np.asarray(vec, dtype=np.float64) - self.mean) / self.std
        d = np.sqrt(np.sum((self.X - q) ** 2, axis=1))
        k = min(self.k, d.shape[0])
        nn = np.argsort(d)[:k]
        votes = np.bincount(self.y[nn], minlength=len(self.labels))
        idx = int(np.argmax(votes))
        conf = float(votes[idx]) / float(k)
        return self.labels[idx], conf

    def save(self, path: str) -> None:
        np.savez(
            path, X=self.X, y=self.y, labels=np.array(self.labels),
            mean=self.mean, std=self.std, k=np.array([self.k]),
        )


def load_model(path: str) -> KNNVowel | None:
    import os
    if not path or not os.path.isfile(path):
        return None
    try:
        d = np.load(path, allow_pickle=False)
        return KNNVowel(
            X=d["X"], y=d["y"], labels=[str(s) for s in d["labels"]],
            mean=d["mean"], std=d["std"], k=int(d["k"][0]),
        )
    except Exception:
        return None
