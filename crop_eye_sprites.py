#!/usr/bin/env python3
"""gen_eye_sprites.py の出力(閉じ目/半目フル画像)を目スプライト(RGBA)に切り出す。

全フレーム(1280x960)サイズの RGBA を作り、RGB=生成画像、α=目領域の羽根付き楕円。
ランタイムは元動画にこの α×まばたき量 で合成する(位置合わせ不要・1:1)。

  uv run python crop_eye_sprites.py
出力: workspace/gura/eye/closed.png, half.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# base_face(1280x960) の目box (cx,cy,hw,hh)。合成αはまつ毛も含め少し大きめ。
EYES = [(486, 428, 114, 80), (742, 428, 114, 80)]  # ループ実測で目全体をカバー(外角まで)


def build_alpha(w: int, h: int, feather: int = 21) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for cx, cy, hw, hh in EYES:
        cv2.ellipse(m, (cx, cy), (hw, hh), 0, 0, 360, 255, -1)
    m = cv2.GaussianBlur(m, (feather, feather), 0)  # 縁を羽根化
    return m


def color_match(gen: np.ndarray, base: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """生成画像(VAEで色がずれる)を、目以外の不変領域でbaseに平均色合わせする。"""
    if base is None or base.shape != gen.shape:
        return gen
    ref = alpha < 10  # 目以外(=本来baseと同一のはず)
    if ref.sum() < 1000:
        return gen
    g = gen.astype(np.float32); b = base.astype(np.float32)
    for c in range(3):
        off = float(b[:, :, c][ref].mean() - g[:, :, c][ref].mean())
        g[:, :, c] += off
    return np.clip(g, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description="目スプライト(RGBA)切り出し")
    ap.add_argument("--gen-dir", default="workspace/gura/eye_gen")
    ap.add_argument("--out-dir", default="workspace/gura/eye")
    ap.add_argument("--base", default="workspace/gura/base_face.png", help="色合わせ基準")
    ap.add_argument("--states", default="closed,half")
    args = ap.parse_args()

    gen = Path(args.gen_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    base = cv2.imread(args.base)
    made = []
    for st in [s.strip() for s in args.states.split(",") if s.strip()]:
        src = gen / f"{st}_full.png"
        if not src.is_file():
            print(f"  skip {st}: {src} なし"); continue
        bgr = cv2.imread(str(src))
        h, w = bgr.shape[:2]
        alpha = build_alpha(w, h)
        bgr = color_match(bgr, base, alpha)  # 元絵に色味を合わせる
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        dst = out / f"{st}.png"
        cv2.imwrite(str(dst), rgba)
        made.append(str(dst))
        print(f"  OK {st} -> {dst}  (α被覆 {100*(alpha>10).mean():.1f}%)")
    if not made:
        print("ERROR: 生成画像が無い。先に gen_eye_sprites.py"); return 1
    print("\nランタイム: --eye-sprite-dir workspace/gura/eye で自動使用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
