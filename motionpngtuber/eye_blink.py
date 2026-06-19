"""まばたき層合成（のっぺらぼう＋開き目＋閉じ目）。

see-through で目を分離した素材を使い、ループ動画の上で:
  1. eyeless パッチで元の目を消す（のっぺらぼう化）
  2. 開き目スプライトを重ねる（既定の見た目）
  3. 閉じ目スプライトを上から「不透明縦ワイプ」で降ろす（webカメラの eyeBlink 連続値）

半透明クロスフェードではなく、各画素は開/閉のどちらか一方になる縦ワイプなので
二重像(ゴースト)が出ない。のっぺらぼう土台のため元の虹彩がはみ出すこともない。
座標は base/loop(1280x960)基準。頭の動きは mouth_track 中心移動で平行追従。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


def load_layer(path: str):
    """RGBA レイヤーを (RGB, alpha float0..1) で読む。無ければ None。"""
    if not path or not os.path.isfile(path):
        return None
    bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgra is None:
        return None
    if bgra.ndim == 3 and bgra.shape[2] == 4:
        rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = bgra[:, :, 3].astype(np.float32) / 255.0
    else:
        rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = np.ones(bgra.shape[:2], np.float32)
    return rgb, alpha


# 後方互換（旧名）
load_eye_sprite = load_layer


@dataclass
class EyeBox:
    cx: float
    cy: float
    hw: float
    hh: float


# see-through層の実位置で再較正した目領域（縦ワイプのまぶた範囲＆左右分割に使用）
DEFAULT_LEFT = EyeBox(468.0, 435.0, 108.0, 66.0)
DEFAULT_RIGHT = EyeBox(838.0, 438.0, 110.0, 66.0)


class EyeBlinkOverlay:
    """のっぺらぼう＋開き目＋閉じ目の層合成によるまばたき。"""

    def __init__(self, eyeless=None, open_eye=None, closed=None,
                 left: EyeBox = DEFAULT_LEFT, right: EyeBox = DEFAULT_RIGHT,
                 open_level: float = 0.25, close_level: float = 0.6,
                 ref_center: tuple[float, float] | None = None,
                 swap: bool = False, split_x: float = 652.0) -> None:
        self._eyeless0 = eyeless    # (rgb, a) 目消し肌
        self._open0 = open_eye      # (rgb, a) 開き目
        self._closed0 = closed      # (rgb, a) 閉じ目
        self.left = left
        self.right = right
        self.open_level = open_level
        self.close_level = close_level
        self.ref_center = ref_center
        self.swap = swap
        self.split_x = split_x
        self._cache: dict = {}

    def has_layers(self) -> bool:
        return self._open0 is not None and self._closed0 is not None

    def _norm(self, raw: float) -> float:
        if self.close_level <= self.open_level:
            return float(np.clip(raw, 0.0, 1.0))
        return float(np.clip((raw - self.open_level) / (self.close_level - self.open_level), 0.0, 1.0))

    def _fit(self, w: int, h: int):
        """各層をフレーム解像度に合わせ、bbox・左右列マスクをキャッシュ。"""
        c = self._cache.get((w, h))
        if c is not None:
            return c
        s = w / 1280.0

        def rs(layer):
            if layer is None:
                return None
            rgb = cv2.resize(layer[0], (w, h), interpolation=cv2.INTER_AREA)
            a = cv2.resize(layer[1], (w, h), interpolation=cv2.INTER_AREA)
            return rgb, a

        eyeless, open_e, closed = rs(self._eyeless0), rs(self._open0), rs(self._closed0)
        # 目領域 bbox（開き+閉じ+消しのα和）
        acc = np.zeros((h, w), np.float32)
        for L in (eyeless, open_e, closed):
            if L is not None:
                acc = np.maximum(acc, L[1])
        ys, xs = np.where(acc > 0.02)
        if ys.size:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        else:
            bbox = (0, 0, w, h)
        sx = self.split_x * s
        c = dict(s=s, eyeless=eyeless, open=open_e, closed=closed, bbox=bbox, split=sx)
        self._cache[(w, h)] = c
        return c

    def _lid(self, box: EyeBox, b: float, ys: np.ndarray, s: float, dy: float) -> np.ndarray:
        """まぶた縦マスク: lid_y より上=1(閉じ目を出す), 下=0。ys は行座標配列。"""
        top = (box.cy - box.hh) * s + dy
        bot = (box.cy + box.hh) * s + dy
        lid = top + b * (bot - top)
        feather = max(2.0, (bot - top) * 0.04)
        return np.clip((lid - ys) / feather, 0.0, 1.0)

    @staticmethod
    def _comp(frame: np.ndarray, rgb: np.ndarray, a: np.ndarray,
              bbox, dxi: int, dyi: int) -> None:
        """rgb を alpha a で frame に合成（(dxi,dyi)平行移動、bbox領域のみ）。"""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = bbox
        cx0, cy0 = max(0, x0 + dxi), max(0, y0 + dyi)
        cx1, cy1 = min(w, x1 + dxi), min(h, y1 + dyi)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sx0, sy0, sx1, sy1 = cx0 - dxi, cy0 - dyi, cx1 - dxi, cy1 - dyi
        A = a[sy0:sy1, sx0:sx1][:, :, None]
        sub = frame[cy0:cy1, cx0:cx1].astype(np.float32)
        spr = rgb[sy0:sy1, sx0:sx1].astype(np.float32)
        frame[cy0:cy1, cx0:cx1] = (sub * (1.0 - A) + spr * A).astype(np.uint8)

    def draw(self, frame: np.ndarray, blink_l: float, blink_r: float,
             cur_center: tuple[float, float] | None = None) -> None:
        if not self.has_layers():
            return
        h, w = frame.shape[:2]
        c = self._fit(w, h)
        s = c["s"]
        dx = dy = 0.0
        if self.ref_center is not None and cur_center is not None:
            dx = cur_center[0] - self.ref_center[0]
            dy = cur_center[1] - self.ref_center[1]
        dxi, dyi = int(round(dx)), int(round(dy))
        bl, br = self._norm(blink_l), self._norm(blink_r)
        if self.swap:
            bl, br = br, bl

        # 1) のっぺらぼう化（元の目を消す）
        if c["eyeless"] is not None:
            self._comp(frame, c["eyeless"][0], c["eyeless"][1], c["bbox"], dxi, dyi)
        # 2) 開き目（既定）
        self._comp(frame, c["open"][0], c["open"][1], c["bbox"], dxi, dyi)
        # 3) 閉じ目を縦ワイプで降ろす（左右独立）
        crgb, ca = c["closed"]
        ys = np.arange(h, dtype=np.float32)[:, None]
        mL = self._lid(self.left, bl, ys, s, dy)    # (h,1)
        mR = self._lid(self.right, br, ys, s, dy)
        split = int(c["split"] + dxi)
        lid_full = np.zeros((h, w), np.float32)
        lid_full[:, :split] = mL
        lid_full[:, split:] = mR
        closed_a = ca * lid_full
        self._comp(frame, crgb, closed_a, c["bbox"], dxi, dyi)
