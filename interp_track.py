#!/usr/bin/env python3
"""mouth_track.npz を任意フレーム数に線形補間する。

体ループ動画をフレーム補間(30→60fps等)で コマ数 を増やすと、口トラック
(frame_idx→quad) も同じフレーム数に合わせないと get_quad の % で位置がズレる。
quad/valid/confidence を時間方向に線形補間し、fps を更新して書き出す。

使い方:
  uv run python interp_track.py workspace/gura/mouth_track.npz \\
      --out workspace/gura/mouth_track_60.npz --frames 357 --fps 60
  # または動画に合わせる:
  uv run python interp_track.py in.npz --out out.npz --like-video loop_60.mp4
"""
from __future__ import annotations

import argparse

import numpy as np


def interp_track(src: dict, n_out: int, fps_out: float) -> dict:
    quad = src["quad"].astype(np.float64)        # (N,4,2)
    valid = src["valid"].astype(np.uint8)        # (N,)
    conf = src["confidence"].astype(np.float64)  # (N,)
    n_in = quad.shape[0]
    if n_in < 2:
        raise ValueError("入力トラックのフレーム数が不足")

    # 出力フレーム i を 入力レンジ[0, n_in-1] に線形マップ（ループ始終端を揃える）
    pos = np.linspace(0.0, n_in - 1, n_out)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    frac = (pos - lo)[:, None, None]

    quad_out = quad[lo] * (1.0 - frac) + quad[hi] * frac
    conf_out = conf[lo] * (1.0 - (pos - lo)) + conf[hi] * (pos - lo)
    # valid は両隣が有効なときのみ有効（補間で無効を跨がない）
    valid_out = (valid[lo].astype(bool) & valid[hi].astype(bool)).astype(np.uint8)

    out = {k: src[k] for k in src.files} if hasattr(src, "files") else dict(src)
    out["quad"] = quad_out.astype(np.float32)
    out["valid"] = valid_out
    out["confidence"] = conf_out.astype(np.float32)
    out["fps"] = np.array(float(fps_out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="mouth_track.npz をフレーム補間")
    ap.add_argument("src", help="入力 mouth_track.npz")
    ap.add_argument("--out", required=True, help="出力 npz")
    ap.add_argument("--frames", type=int, default=0, help="出力フレーム数")
    ap.add_argument("--fps", type=float, default=0.0, help="出力fps(メタ更新用)")
    ap.add_argument("--like-video", default="", help="この動画のフレーム数/fpsに合わせる")
    args = ap.parse_args()

    d = np.load(args.src, allow_pickle=True)
    n_out, fps_out = args.frames, args.fps
    if args.like_video:
        import cv2
        cap = cv2.VideoCapture(args.like_video)
        n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_out = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        print(f"[like-video] {args.like_video}: frames={n_out} fps={fps_out}")
    if n_out <= 0:
        print("ERROR: --frames か --like-video を指定してください")
        return 1
    if fps_out <= 0:
        fps_out = float(d["fps"]) if "fps" in d.files else 30.0

    out = interp_track(d, n_out, fps_out)
    np.savez(args.out, **out)
    print(f"=== 保存: {args.out} ===")
    print(f"  frames: {d['quad'].shape[0]} -> {out['quad'].shape[0]}  fps={fps_out}")
    print(f"  valid_rate: {100*out['valid'].mean():.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
