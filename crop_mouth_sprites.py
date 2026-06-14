"""5枚の口inpaint画像 → 194x194 RGBA 口スプライト (mouth/{open,closed,half,e,u}.png)。

【精巧化方式】pad大(1.8)+mask_scale大(0.85)で大きめにパッチ取得→GrabCutで
口の実際の形に沿った精巧な前景マスクを推定→フェザー付きRGBA。
楕円マスク（口の形に追従しない）の精度問題を改善。

Phase1 (comfyui_api/generate_mouth_sprites.py) が出力した
  mouth_<shape>_00001_.png (open/closed/half/e/u の5枚)
を入力とする。

MotionPNGTuber ディレクトリで uv run すること（.venv の anime-face-detector/cv2 を使用）。
"""
import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

from motionpngtuber.mouth_sprite_extractor import (
    load_track_data,
    warp_frame_to_norm,
    make_ellipse_mask,
    feather_mask,
)

SHAPES = ["open", "closed", "half", "e", "u", "aa", "o"]
FILENAME_TMPL = "mouth_{shape}_00001_.png"


def extract_mouth_sprite_refined(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    unified_w: int,
    unified_h: int,
    feather_px: int = 15,
    mask_scale: float = 0.85,
    grabcut_iter: int = 5,
) -> np.ndarray:
    """大きめ楕円で初期マスク→GrabCutで口の実際の形に精巧化→RGBA。

    Args:
        frame_bgr: 入力フレーム(BGR)
        quad: 口のquad(4,2)。pad大で口周りを広く取る前提。
        unified_w/h: 出力サイズ
        feather_px: フェザー幅
        mask_scale: 初期楕円マスクのサイズ係数(大きめ0.85推奨)
        grabcut_iter: GrabCut反復回数
    Returns:
        rgba: (H,W,4) uint8
    """
    patch_bgr = warp_frame_to_norm(frame_bgr, quad, unified_w, unified_h)
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)

    # 初期マスク: 大きめ楕円
    rx = int((unified_w * mask_scale) * 0.5)
    ry = int((unified_h * mask_scale) * 0.5)
    init_ellipse = make_ellipse_mask(unified_w, unified_h, rx, ry)

    # GrabCut用マスク: 楕円内=恐らく前景, 楕円外=背景
    gc = np.zeros((unified_h, unified_w), np.uint8)
    gc[init_ellipse > 0] = cv2.GC_PR_FGD
    gc[init_ellipse == 0] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(patch_rgb, gc, None, bgd, fgd, grabcut_iter, cv2.GC_INIT_WITH_MASK)
        refined = np.where(
            (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except Exception:
        # GrabCut失敗時は楕円マスクにフォールバック
        refined = init_ellipse

    mask_f = feather_mask(refined, feather_px)
    rgba = np.zeros((unified_h, unified_w, 4), dtype=np.uint8)
    rgba[:, :, :3] = patch_rgb
    rgba[:, :, 3] = (mask_f * 255).astype(np.uint8)
    return rgba


def main() -> int:
    ap = argparse.ArgumentParser(description="口inpaint5枚 → 口スプライトRGBA切り抜き(GrabCut精巧化)")
    ap.add_argument("--input-dir", required=True, help="mouth_inpaint ディレクトリ（5枚）")
    ap.add_argument("--out", required=True, help="出力 mouth/ ディレクトリ")
    ap.add_argument("--sprite-size", type=int, default=194, help="スプライト一辺（px）")
    ap.add_argument("--feather", type=int, default=15, help="フェザー幅（px）")
    ap.add_argument("--pad", type=float, default=1.8, help="口quadのパディング係数（大きめ1.8推奨: GrabCut余地）")
    ap.add_argument("--mask-scale", type=float, default=0.85, help="初期楕円マスク係数（大きめ0.85推奨）")
    ap.add_argument("--device", default="cpu", help="検出デバイス（RTX5070Tiはcpu必須）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- 1. 5枚を読込（順序固定）---
    frames = []
    for shape in SHAPES:
        p = os.path.join(args.input_dir, FILENAME_TMPL.format(shape=shape))
        img = cv2.imread(p)
        if img is None:
            print(f"ERROR: 読込失敗: {p}")
            return 1
        frames.append(img)
        print(f"  loaded {shape}: {p}  {img.shape}")
    h, w = frames[0].shape[:2]

    # --- 2. 5フレーム動画を生成 ---
    tmpdir = tempfile.mkdtemp(prefix="crop_mouth_")
    video = os.path.join(tmpdir, "mouth_5frames.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video, fourcc, 10.0, (w, h))
    if not writer.isOpened():
        print("ERROR: VideoWriter のオープン失敗（mp4v コーデック不足）")
        return 1
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"  5フレーム動画: {video}")

    # --- 3. face_track_anime_detector.py で口 quad 検出 ---
    npz = os.path.join(tmpdir, "track.npz")
    detector = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_track_anime_detector.py")
    print(f"  口 quad 検出中 (--device {args.device}, --pad {args.pad}) ...")
    ret = subprocess.run(
        [sys.executable, detector,
         "--video", video, "--out", npz, "--device", args.device,
         "--pad", str(args.pad)],
        capture_output=True, text=True,
    )
    if ret.returncode != 0 or not os.path.exists(npz):
        print("ERROR: 口 quad 検出失敗")
        print("--- stdout ---"); print(ret.stdout)
        print("--- stderr ---"); print(ret.stderr)
        return 1
    print("  検出完了")

    # --- 4. npz から quad 取得 ---
    quads, valid, conf = load_track_data(npz, w, h)
    if len(quads) < len(SHAPES):
        print(f"ERROR: quad が不足: {len(quads)} < {len(SHAPES)}")
        return 1

    # --- 5. 各形状を切り抜き（GrabCut精巧化）---
    print(f"\n=== 切り抜き(GrabCut): {args.sprite_size}x{args.sprite_size} pad={args.pad} mask_scale={args.mask_scale} ===")
    for i, shape in enumerate(SHAPES):
        if not valid[i]:
            print(f"  WARNING: {shape} の口検出 valid=False。近傍フレームの quad で代用。")
            for j in range(len(SHAPES)):
                if valid[j]:
                    quads[i] = quads[j]
                    break
        quad = quads[i]
        rgba = extract_mouth_sprite_refined(
            frames[i], quad, args.sprite_size, args.sprite_size,
            feather_px=args.feather, mask_scale=args.mask_scale,
        )
        out_path = os.path.join(args.out, f"{shape}.png")
        Image.fromarray(rgba, mode="RGBA").save(out_path)
        print(f"  {shape}: {out_path}  shape={rgba.shape}  conf={float(conf[i]):.2f}")

    print(f"\n=== 完成: {args.out}/{{{','.join(SHAPES)}}}.png ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
