#!/usr/bin/env python3
"""Live2D的フェイストラッキング・パペット（see-throughレイヤー＋webカメラ頭ポーズ）。

see-through で分解した意味レイヤー(髪/顔/目/口…)を z順に重ね、webカメラの頭の向きで
2.5D的に動かす:
  - roll  → 全体を回転
  - yaw/pitch → レイヤー別パララックス平行移動(手前ほど大きく動く)
  - まばたき → 目レイヤー(虹彩/白目)を縦に潰し、背後の顔(のっぺらぼう肌)を見せて閉じる
  - 視線 → 虹彩レイヤーを平行移動

頭ポーズ等は face_pose_server.py が UDP 送信(既定 port 5007)。

  env PULSE_SERVER=unix:/mnt/wslg/PulseServer uv run python live2d_puppet.py
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from motionpngtuber.eye_track_udp import EyeTrackReceiver

ST = "/mnt/c/Users/mao0202/Documents/GitHub/see-through/data/output_marigold/gura_base"
Y_OFF = 160  # 1280x1280層 -> 1280x960

# z順(奥→手前)と パララックス深度(0=奥/動かない, 1=手前/よく動く)
Z_ORDER = [
    ("wings", 0.0), ("tail", 0.0), ("back hair", 0.05),
    ("bottomwear", 0.1), ("legwear", 0.1), ("footwear", 0.1),
    ("topwear", 0.15), ("handwear", 0.15), ("neck", 0.2), ("neckwear", 0.2),
    ("ears", 0.35), ("earwear", 0.35), ("face", 0.4),
    ("eyewhite", 0.45), ("irides", 0.47), ("eyelash", 0.48),
    ("eyebrow", 0.5), ("nose", 0.5), ("mouth", 0.5), ("eyewear", 0.52),
    ("headwear", 0.6), ("front hair", 0.8), ("objects", 0.5),
]
EYE_LAYERS = {"eyewhite", "irides"}
# 目領域(see-through実測, 1280x960): 縦潰しの中心
EYE_CY = 436


def load_layers(st_dir: str):
    import os
    layers = []
    for name, depth in Z_ORDER:
        p = os.path.join(st_dir, f"{name}.png")
        if not os.path.isfile(p):
            continue
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None or im.shape[2] < 4:
            continue
        im = im[Y_OFF:Y_OFF + 960]  # 1280x960へ
        rgb = im[:, :, :3].astype(np.float32)
        a = (im[:, :, 3].astype(np.float32) / 255.0)
        if a.max() < 0.02:
            continue
        layers.append([name, depth, rgb, a])
    return layers


def shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    x0s, x1s = max(0, dx), min(w, w + dx)
    y0s, y1s = max(0, dy), min(h, h + dy)
    x0d, x1d = max(0, -dx), min(w, w - dx)
    y0d, y1d = max(0, -dy), min(h, h - dy)
    out[y0s:y1s, x0s:x1s] = img[y0d:y1d, x0d:x1d]
    return out


def squash_eye(rgb, a, blink, cy):
    """blink(0..1)で目レイヤーをcy中心に縦縮小→背後の肌が見え閉じる。"""
    if blink <= 0.02:
        return rgb, a
    sc = max(0.05, 1.0 - blink)
    h, w = a.shape
    M = np.float32([[1, 0, 0], [0, sc, cy * (1 - sc)]])
    a2 = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    rgb2 = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rgb2, a2


def main() -> int:
    ap = argparse.ArgumentParser(description="Live2D的パペット")
    ap.add_argument("--st-dir", default=ST)
    ap.add_argument("--port", type=int, default=5007)
    ap.add_argument("--k-yaw", type=float, default=2.2, help="yaw→水平パララックス係数(px/度)")
    ap.add_argument("--k-pitch", type=float, default=1.6, help="pitch→垂直パララックス")
    ap.add_argument("--k-roll", type=float, default=1.0, help="roll→回転係数(度/度)")
    ap.add_argument("--scale", type=float, default=0.6, help="表示縮小")
    ap.add_argument("--bg", default="240,240,245")
    args = ap.parse_args()

    layers = load_layers(args.st_dir)
    if not layers:
        print("ERROR: レイヤーが読めません"); return 1
    print(f"[live2d] {len(layers)}層: {', '.join(l[0] for l in layers)}")
    H, W = layers[0][2].shape[:2]
    bg = np.array([int(x) for x in args.bg.split(",")], np.float32)
    pivot = (W / 2, H * 0.8)  # 首あたりを回転中心

    rx = EyeTrackReceiver(port=args.port).start()
    print(f"[live2d] 頭ポーズ受信 :{args.port}  (face_pose_server.py を起動)。q で終了")

    # スムージング
    sy = sp = sr = 0.0
    sbl = sbr = 0.0
    win = "live2d puppet (q quit)"
    last = time.perf_counter(); fps = 0.0
    while True:
        yaw, pitch, roll = rx.get_head()
        bl, br, _ = rx.get_blink()
        lx, ly = rx.get_look()
        a = 0.4  # EMA
        sy = (1 - a) * sy + a * yaw; sp = (1 - a) * sp + a * pitch; sr = (1 - a) * sr + a * roll
        sbl = (1 - a) * sbl + a * min(1.0, bl * 1.4); sbr = (1 - a) * sbr + a * min(1.0, br * 1.4)

        canvas = np.tile(bg, (H, W, 1))
        for name, depth, rgb, al in layers:
            dx = int(round(args.k_yaw * sy * depth))
            dy = int(round(args.k_pitch * sp * depth))
            r, aa = rgb, al
            if name in EYE_LAYERS:
                bk = sbl if name else sbr  # 左右別は近似(両目同時が殆ど)
                r, aa = squash_eye(rgb, al, max(sbl, sbr), EYE_CY)
            if name == "irides":
                dx += int(round(lx * 18)); dy += int(round(-ly * 12))
            if dx or dy:
                r = shift(r, dx, dy); aa = shift(aa, dx, dy)
            am = aa[:, :, None]
            canvas = canvas * (1 - am) + r * am

        out = canvas.astype(np.uint8)
        if abs(sr) > 0.5:  # roll 全体回転
            M = cv2.getRotationMatrix2D(pivot, args.k_roll * sr, 1.0)
            out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderValue=tuple(int(x) for x in bg))
        if args.scale != 1.0:
            out = cv2.resize(out, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

        now = time.perf_counter(); fps = 0.9 * fps + 0.1 / max(1e-3, now - last); last = now
        cv2.putText(out, f"yaw{sy:+.0f} pitch{sp:+.0f} roll{sr:+.0f} fps{fps:.0f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 200), 2)
        cv2.imshow(win, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
    rx.stop(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
