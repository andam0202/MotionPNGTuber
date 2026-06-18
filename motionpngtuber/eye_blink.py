"""手続き的アナログまばたき描画。

webカメラの eyeBlink 連続値(0..1)で、アバターの目に「上まぶたが降りる」表現を
重ねる。閉じ目スプライト不要・半目も滑らか。アニメの閉じ目(肌色まぶた＋まつ毛
カーブ)を近似する。肌色は実フレームからサンプリングするので色調整に追従する。

座標は base_face / loop 動画(1280x960)基準。頭の動きは mouth_track の中心移動で
平行追従する（アイドルループの主動作は平行移動なので十分）。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EyeBox:
    cx: float
    cy: float
    hw: float  # half width
    hh: float  # half height


# gura base_face(1280x960) で実測した既定の目領域
DEFAULT_LEFT = EyeBox(487.0, 408.0, 80.0, 54.0)   # 視聴者から見て左
DEFAULT_RIGHT = EyeBox(735.0, 408.0, 80.0, 54.0)  # 視聴者から見て右


def _sample_skin(frame: np.ndarray, pt: tuple[float, float]) -> list:
    """指定点(鼻筋など安定領域)の周辺から肌色の中央値を取る。"""
    h, w = frame.shape[:2]
    cx, cy = pt
    y0 = int(np.clip(cy - 22, 0, h - 1)); y1 = int(np.clip(cy + 22, 1, h))
    x0 = int(np.clip(cx - 32, 0, w - 1)); x1 = int(np.clip(cx + 32, 1, w))
    if y1 <= y0 or x1 <= x0:
        return [232, 230, 198]
    patch = frame[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(patch, axis=0).astype(np.uint8).tolist()


def _draw_lid(frame: np.ndarray, box: EyeBox, amount: float, skin: list) -> None:
    """amount(0..1) に応じて上まぶたを楕円マスク内で降ろし、下端にまつ毛を描く。"""
    if amount <= 0.02:
        return
    h, w = frame.shape[:2]
    x0 = int(max(0, box.cx - box.hw)); x1 = int(min(w, box.cx + box.hw))
    y0 = int(max(0, box.cy - box.hh)); y1 = int(min(h, box.cy + box.hh))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    ecx, ecy = rw / 2.0, rh / 2.0
    eax, eay = rw / 2.0 - 1, rh / 2.0 - 1

    # 目形の楕円マスク
    eye_mask = np.zeros((rh, rw), np.uint8)
    cv2.ellipse(eye_mask, (int(ecx), int(ecy)), (int(eax), int(eay)), 0, 0, 360, 255, -1)
    # 上から amount 分を覆う帯
    lid_row = int(amount * rh)
    band = np.zeros((rh, rw), np.uint8)
    band[:lid_row, :] = 255
    cover = cv2.bitwise_and(eye_mask, band) > 0
    if cover.any():
        roi[cover] = skin

    # まつ毛: 楕円の lid_row 位置での弦に沿って、中央が少し下がるカーブ
    yy = (lid_row - ecy) / max(1e-3, eay)
    half = eax * float(np.sqrt(max(0.0, 1.0 - yy * yy))) if abs(yy) < 1.0 else eax * 0.3
    lx0 = int(ecx - half); lx1 = int(ecx + half)
    dip = int(rh * 0.10)
    thick = max(2, int(box.hh * 0.11))
    pts = np.array([[lx0, lid_row - dip // 2],
                    [int(ecx), min(rh - 1, lid_row + dip)],
                    [lx1, lid_row - dip // 2]], np.int32)
    cv2.polylines(roi, [pts], False, (40, 35, 45), thick, cv2.LINE_AA)


class EyeBlinkOverlay:
    """まばたきオーバーレイ。生 eyeBlink を open/close レンジで 0..1 に正規化して描く。"""

    def __init__(self, left: EyeBox = DEFAULT_LEFT, right: EyeBox = DEFAULT_RIGHT,
                 open_level: float = 0.25, close_level: float = 0.6,
                 ref_center: tuple[float, float] | None = None,
                 swap: bool = False,
                 skin_pt: tuple[float, float] = (625.0, 455.0)) -> None:
        self.left = left
        self.right = right
        self.open_level = open_level
        self.close_level = close_level
        self.ref_center = ref_center  # 追従基準(mouth_track中心)
        self.swap = swap
        self.skin_pt = skin_pt  # 肌色採取点(鼻筋)

    def _norm(self, raw: float) -> float:
        if self.close_level <= self.open_level:
            return float(np.clip(raw, 0.0, 1.0))
        return float(np.clip((raw - self.open_level) / (self.close_level - self.open_level), 0.0, 1.0))

    def _shift(self, box: EyeBox, dx: float, dy: float) -> EyeBox:
        return EyeBox(box.cx + dx, box.cy + dy, box.hw, box.hh)

    def draw(self, frame: np.ndarray, blink_l: float, blink_r: float,
             cur_center: tuple[float, float] | None = None) -> None:
        h, w = frame.shape[:2]
        s = w / 1280.0  # 目座標は base_face(1280幅)基準。フレーム解像度に合わせる
        dx = dy = 0.0
        if self.ref_center is not None and cur_center is not None:
            dx = cur_center[0] - self.ref_center[0]
            dy = cur_center[1] - self.ref_center[1]
        bl, br = self._norm(blink_l), self._norm(blink_r)
        if self.swap:
            bl, br = br, bl
        skin = _sample_skin(frame, (self.skin_pt[0] * s + dx, self.skin_pt[1] * s + dy))

        def sb(box: EyeBox) -> EyeBox:
            return EyeBox(box.cx * s + dx, box.cy * s + dy, box.hw * s, box.hh * s)

        _draw_lid(frame, sb(self.left), bl, skin)
        _draw_lid(frame, sb(self.right), br, skin)
