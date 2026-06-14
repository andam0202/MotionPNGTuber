"""LPCベースのフォルマント推定による母音(あ/い/う/え/お)判定。

リアルタイム口パクのライブ版用。音声チャンクから F1/F2 を推定し、
母音空間での最近傍で母音を決める。numpy のみ（依存追加なし・CPU軽量）。

音量3段階（rms）が「どれだけ開くか」を、本モジュールが「どの母音の口形か」を
担うことを想定。無音/低エネルギーは判定せず呼び出し側で closed にする。
"""
from __future__ import annotations

import numpy as np

# 女性話者のおおよその母音フォルマント (F1, F2) [Hz]。
# キャラ音声に合わせて調整可能。log空間で最近傍を取る。
VOWEL_FORMANTS = {
    "a": (850.0, 1300.0),
    "i": (350.0, 2700.0),
    "u": (400.0, 1100.0),
    "e": (550.0, 2100.0),
    "o": (500.0, 900.0),
}

# 母音 → 口スプライト形状（新スプライト aa/o があれば母音精度が上がる）
VOWEL_TO_SHAPE = {
    "a": "aa",   # 大きく開く（無ければ open にフォールバック）
    "i": "e",    # 横に開く/歯
    "u": "u",    # すぼめ
    "e": "e",
    "o": "o",    # 丸く開く（無ければ u にフォールバック）
}


def _levinson(r: np.ndarray, order: int) -> np.ndarray:
    """自己相関 r から Levinson-Durbin で LPC 係数 a (a[0]=1) を返す。"""
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e = float(r[0])
    if e <= 0:
        return a
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1]) if i > 1 else r[i]
        k = -acc / e
        a_prev = a.copy()
        for j in range(1, i):
            a[j] = a_prev[j] + k * a_prev[i - j]
        a[i] = k
        e *= (1.0 - k * k)
        if e <= 0:
            break
    return a


# F1/F2 の探索帯域 [Hz]（重ならないよう境界を分ける）
F1_BAND = (180.0, 1000.0)
F2_BAND = (1000.0, 3200.0)


def _all_poles(x: np.ndarray, sr: int, order: int, max_bw_hz: float) -> list[tuple[float, float]]:
    """LPC の極から (周波数, バンド幅) のリストを周波数昇順で返す。"""
    x = np.asarray(x, dtype=np.float64).flatten()
    if x.size < 32:
        return []
    if order <= 0:
        order = int(sr / 1000) + 2  # 16kHz→18次（フォルマント抽出の定番）
    x = x - x.mean()
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])  # プリエンファシス
    x *= np.hamming(x.size)
    corr = np.correlate(x, x, mode="full")[x.size - 1:]
    if corr[0] <= 1e-9:
        return []
    a = _levinson(corr[: order + 1], order)
    if a[0] == 0 or not np.all(np.isfinite(a)):
        return []
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 1e-3]  # 上半平面のみ
    if roots.size == 0:
        return []
    angs = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angs * (sr / (2.0 * np.pi))
    bws = -0.5 * (sr / (2.0 * np.pi)) * np.log(np.maximum(np.abs(roots), 1e-9))
    poles = [(float(f), float(b)) for f, b in zip(freqs, bws) if b < max_bw_hz and f > 0]
    poles.sort(key=lambda p: p[0])
    return poles


def estimate_formants(
    x: np.ndarray,
    sr: int,
    order: int = 0,
    n: int = 2,
    max_bw_hz: float = 500.0,
) -> list[float]:
    """音声チャンクから F1, F2 を帯域分割で推定して返す（取れない要素は省く）。

    低い順に拾うのではなく、F1 は F1_BAND、F2 は F2_BAND の中で最も鋭い
    （バンド幅最小＝共鳴が強い）極を選ぶ。F1/F2 間の偽極を F2 と誤らない。
    """
    poles = _all_poles(x, sr, order, max_bw_hz)
    if not poles:
        return []
    f1c = [(f, b) for f, b in poles if F1_BAND[0] <= f <= F1_BAND[1]]
    f2c = [(f, b) for f, b in poles if F2_BAND[0] < f <= F2_BAND[1]]
    out: list[float] = []
    if f1c:
        out.append(min(f1c, key=lambda p: p[1])[0])
    if f2c:
        out.append(min(f2c, key=lambda p: p[1])[0])
    return out[:n]


def classify_vowel(
    x: np.ndarray,
    sr: int,
    formants: dict[str, tuple[float, float]] | None = None,
) -> tuple[str | None, tuple[float, float] | None]:
    """音声チャンク → (母音ラベル or None, (F1,F2) or None)。

    F1/F2 が取れない場合は (None, None)。判定は log 周波数の最近傍。
    """
    fmts = estimate_formants(x, sr, n=2)
    if len(fmts) < 2:
        return None, None
    f1, f2 = fmts[0], fmts[1]
    table = formants or VOWEL_FORMANTS
    best_v = None
    best_d = 1e18
    lf = np.log(np.array([f1, f2]))
    for v, (rf1, rf2) in table.items():
        ref = np.log(np.array([rf1, rf2]))
        d = float(np.sum((lf - ref) ** 2))
        if d < best_d:
            best_d = d
            best_v = v
    return best_v, (f1, f2)


def vowel_to_shape(vowel: str | None, available: set[str]) -> str:
    """母音ラベル → 利用可能な口スプライト形状にフォールバック付きで対応。"""
    if vowel is None:
        return "closed"
    shape = VOWEL_TO_SHAPE.get(vowel, "open")
    if shape in available:
        return shape
    # フォールバック: aa→open, o→u, e→half
    fallback = {"aa": "open", "o": "u", "e": "half", "i": "e"}
    s = shape
    for _ in range(3):
        s = fallback.get(s, "open")
        if s in available:
            return s
    return "open" if "open" in available else "closed"
