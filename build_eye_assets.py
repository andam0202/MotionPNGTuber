#!/usr/bin/env python3
"""see-through の層分解出力から、まばたき用アセットを構築する。

see-through(別repo)が base_face を 1280x1280(y+160パディング)に分解した層から:
  - eyeless.png : 目を消したのっぺらぼう肌パッチ(RGBA)。ループ動画に重ねて目を消す。
  - open.png    : 開き目スプライト(eyewhite+irides+eyelash 合成, RGBA)。

消去/開き目のαは目レイヤー(eyewhite+irides+eyelash)のα和から作るので、目だけを
正確に扱える。出力は base/ループ(1280x960)整合。

  uv run python build_eye_assets.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ST_DIR = "/mnt/c/Users/mao0202/Documents/GitHub/see-through/data/output_marigold/gura_base"
Y_OFFSET = 160  # 1280x1280 層 -> base(1280x960): [160:1120]
EYE_LAYERS = ["eyewhite", "irides", "eyelash"]


def crop960(im: np.ndarray) -> np.ndarray:
    return im[Y_OFFSET:Y_OFFSET + 960]


def load_layer(d: Path, name: str) -> np.ndarray:
    im = cv2.imread(str(d / f"{name}.png"), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(d / f"{name}.png")
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return crop960(im)


def main() -> int:
    ap = argparse.ArgumentParser(description="see-through層→まばたきアセット")
    ap.add_argument("--st-dir", default=ST_DIR, help="see-through gura_base 出力dir")
    ap.add_argument("--out-dir", default="workspace/gura/eye")
    ap.add_argument("--erase-dilate", type=int, default=9, help="消去マスクの膨張(px)")
    ap.add_argument("--erase-feather", type=int, default=15, help="消去マスクの羽根(px)")
    args = ap.parse_args()

    d = Path(args.st_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    face = load_layer(d, "face")          # のっぺらぼう肌(目/鼻/口inpaint済)
    layers = {n: load_layer(d, n) for n in EYE_LAYERS}

    # 目領域α = 3レイヤーのα和
    eye_a = np.zeros(face.shape[:2], np.float32)
    for n in EYE_LAYERS:
        eye_a = np.maximum(eye_a, layers[n][:, :, 3].astype(np.float32) / 255.0)

    # --- 開き目スプライト(eyewhite→irides→eyelash の順で重ねる) ---
    open_rgb = np.zeros_like(face[:, :, :3])
    open_a = np.zeros(face.shape[:2], np.float32)
    for n in EYE_LAYERS:
        la = layers[n][:, :, 3].astype(np.float32) / 255.0
        for c in range(3):
            open_rgb[:, :, c] = (open_rgb[:, :, c] * (1 - la) + layers[n][:, :, c] * la).astype(np.uint8)
        open_a = np.maximum(open_a, la)
    open_rgba = np.dstack([open_rgb, (open_a * 255).astype(np.uint8)])
    cv2.imwrite(str(out / "open.png"), open_rgba)

    # --- 閉じ目(キャラ本来のまつ毛を下げて閉じまつ毛にする・画風一致/対称) ---
    # ComfyUIのinpaintはまつ毛が薄く出るため、see-throughのeyelash層を流用。
    lash = layers["eyelash"]
    sh = 35  # まつ毛を下げる量(px)。上まつ毛→閉じ目の位置へ
    M = np.float32([[1, 0, 0], [0, 1, sh]])
    la = cv2.warpAffine(lash[:, :, 3].astype(np.float32) / 255.0, M, (face.shape[1], face.shape[0]))
    lr = cv2.warpAffine(lash[:, :, :3].astype(np.float32), M, (face.shape[1], face.shape[0]))
    closed_rgb = face[:, :, :3].astype(np.float32).copy()
    am = la[:, :, None]
    closed_rgb = (closed_rgb * (1 - am) + lr * am).astype(np.uint8)
    closed_alpha = cv2.GaussianBlur((eye_a > 0.05).astype(np.uint8) * 255, (13, 13), 0)
    cv2.imwrite(str(out / "closed.png"), np.dstack([closed_rgb, closed_alpha]))

    # --- のっぺらぼう肌パッチ(目領域を膨張+羽根化して消去用に) ---
    em = (eye_a > 0.05).astype(np.uint8) * 255
    if args.erase_dilate > 0:
        k = np.ones((args.erase_dilate, args.erase_dilate), np.uint8)
        em = cv2.dilate(em, k)
    f = args.erase_feather | 1
    em = cv2.GaussianBlur(em, (f, f), 0)
    eyeless_rgba = np.dstack([face[:, :, :3], em])
    cv2.imwrite(str(out / "eyeless.png"), eyeless_rgba)

    print(f"=== 保存: {out} ===")
    print(f"  open.png     開き目  非透明{int((open_a>0.05).sum())}px")
    print(f"  closed.png   閉じ目(まつ毛下げsh{sh})")
    print(f"  eyeless.png  のっぺらぼう肌パッチ 被覆{int((em>10).sum())}px")
    print("ランタイム: --eye-sprite-dir で eyeless.png/open.png/closed.png を使用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
