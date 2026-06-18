#!/usr/bin/env python3
"""ComfyUI で base_face の目領域だけを inpaint し、閉じ目/半目を生成する。

webカメラのまばたき(連続値)でアバターを瞬きさせる本格版スプライト。base_face の
目boxにマスクを当て、checkpoint(obsessionIllustrious_vPred)で「closed eyes」を
inpaint。顔の同一性は目以外を保持するので維持される。生成後 crop_eye_sprites で
RGBA スプライトに切り出す。

  uv run python gen_eye_sprites.py --host 172.27.112.1:8001

依存なし(urllib)。出力: workspace/gura/eye_gen/<state>_full.png
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np

BASE_POSITIVE = ("gawr gura, 1girl, (light blue hair:1.3), gradient hair, two-tone hair, "
                 "(shark teeth:1.4), sharp triangular teeth, ahoge, (face focus), (portrait), "
                 "white t-shirt, blue hoodie, (shark hood:1.3)")
BASE_NEGATIVE = ("2girl, peoples, lowres, worst quality, low quality, bad anatomy, bad hands, "
                 "text, error, jpeg artifacts, signature, watermark, blurry, "
                 "(realistic:1.3), (photorealistic:1.3), (3d:1.2)")
CKPT = "ill\\obsessionIllustrious_vPredV11.safetensors"

# 目の表情ごとの追加プロンプト
STATES = {
    "closed":   "(closed eyes:1.35), eyes closed, ^_^, gentle smile, eyelid, eyelashes, smooth skin",
    "half":     "(half-closed eyes:1.3), sleepy half-lidded eyes, droopy eyes",
}

# base_face(1280x960) で実測した目box: (cx,cy,hw,hh)
EYES = [(487, 408, 92, 64), (735, 408, 92, 64)]  # マスクは少し大きめ


def build_mask(w: int, h: int, only: int | None = None) -> np.ndarray:
    """目マスク。only=0/1 で片目だけ(順次inpaint用)、None で両目。"""
    m = np.zeros((h, w), np.uint8)
    for i, (cx, cy, hw, hh) in enumerate(EYES):
        if only is not None and i != only:
            continue
        cv2.ellipse(m, (cx, cy), (hw, hh), 0, 0, 360, 255, -1)
    return m


def http_post(url: str, data: bytes, headers: dict) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def upload_image(host: str, name: str, img_bgr_or_gray: np.ndarray) -> str:
    """PNG を multipart で /upload/image にアップロード。返り値=ComfyUI上の名前。"""
    ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    if not ok:
        raise RuntimeError("png encode failed")
    boundary = "----comfymask" + uuid.uuid4().hex
    body = io.BytesIO()
    def w(s): body.write(s if isinstance(s, bytes) else s.encode())
    w(f"--{boundary}\r\n")
    w(f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n')
    w("Content-Type: image/png\r\n\r\n")
    w(buf.tobytes()); w("\r\n")
    for k, v in (("type", "input"), ("overwrite", "true")):
        w(f"--{boundary}\r\n")
        w(f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n')
    w(f"--{boundary}--\r\n")
    resp = http_post(f"http://{host}/upload/image", body.getvalue(),
                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(resp).get("name", name)


def build_workflow(img_name: str, mask_name: str, extra_pos: str, seed: int, cfg: float = 6.0) -> dict:
    """目領域 inpaint の最小ワークフロー(API形式)。vPred のため v_prediction を明示。"""
    pos = f"{extra_pos}, {BASE_POSITIVE}"
    neg = f"(open eyes:1.2), wide eyes, white eyes, blank eyes, {BASE_NEGATIVE}"
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "ModelSamplingDiscrete",
              "inputs": {"model": ["1", 0], "sampling": "v_prediction", "zsnr": False}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}},
        "5": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "6": {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}},
        "7": {"class_type": "VAEEncodeForInpaint",
              "inputs": {"pixels": ["5", 0], "vae": ["1", 2], "mask": ["6", 0], "grow_mask_by": 8}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0],
                         "latent_image": ["7", 0], "seed": seed, "steps": 30, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "eye_inpaint"}},
    }


def run_workflow(host: str, wf: dict) -> list[dict]:
    payload = json.dumps({"prompt": wf, "client_id": uuid.uuid4().hex}).encode()
    resp = http_post(f"http://{host}/prompt", payload, {"Content-Type": "application/json"})
    pid = json.loads(resp)["prompt_id"]
    print(f"  queued prompt_id={pid}")
    for _ in range(300):  # 最大~5分
        time.sleep(1.0)
        hist = json.loads(http_get(f"http://{host}/history/{pid}"))
        if pid in hist:
            outs = hist[pid].get("outputs", {})
            imgs = []
            for node in outs.values():
                imgs += node.get("images", [])
            return imgs
    raise TimeoutError("ComfyUI 生成がタイムアウト")


def main() -> int:
    ap = argparse.ArgumentParser(description="ComfyUIで閉じ目/半目をinpaint生成")
    ap.add_argument("--host", default="172.27.112.1:8001")
    ap.add_argument("--base", default="workspace/gura/base_face.png")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--states", default="closed,half")
    ap.add_argument("--out-dir", default="workspace/gura/eye_gen")
    ap.add_argument("--mirror", action="store_true", default=True,
                    help="左目だけ生成し右目はミラー複製(左右対称・確実)。--no-mirror で両目生成")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    args = ap.parse_args()

    base = cv2.imread(args.base)
    if base is None:
        print(f"ERROR: base が読めません: {args.base}"); return 1
    h, w = base.shape[:2]
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    from urllib.parse import urlencode

    def run_and_fetch(img_np, mask_np, extra_pos, seed) -> np.ndarray:
        img_n = upload_image(args.host, "gura_cur.png", img_np)
        mask_n = upload_image(args.host, "gura_cur_mask.png", mask_np)
        wf = build_workflow(img_n, mask_n, extra_pos, seed, args.cfg)
        imgs = run_workflow(args.host, wf)
        if not imgs:
            raise RuntimeError("出力なし")
        info = imgs[0]
        q = urlencode({"filename": info["filename"], "subfolder": info.get("subfolder", ""),
                       "type": info.get("type", "output")})
        raw = http_get(f"http://{args.host}/view?{q}")
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return arr

    def mirror_l2r(img: np.ndarray) -> np.ndarray:
        """よく閉じた左目(EYES[0])を水平反転して右目(EYES[1])へ複製(左右対称化)。"""
        (lx, ly, _, _) = EYES[0]
        (rx, ry, _, _) = EYES[1]
        rw, rh = 118, 86  # まつ毛を含む十分な範囲(αで楕円に絞られる)
        src = img[ly - rh:ly + rh, lx - rw:lx + rw]
        if src.size == 0:
            return img
        flip = cv2.flip(src, 1)
        out = img.copy()
        out[ry - rh:ry + rh, rx - rw:rx + rw] = flip
        return out

    print(f"=== ComfyUI {args.host} / base {w}x{h} / 左目inpaint→右目はミラー ===")
    for st in [s.strip() for s in args.states.split(",") if s.strip() in STATES]:
        print(f"--- {st}: {STATES[st]} ---")
        try:
            if args.mirror:
                print("  左目(EYES[0])を inpaint...")
                cur = run_and_fetch(base.copy(), build_mask(w, h, only=0), STATES[st], args.seed)
                cur = mirror_l2r(cur)
                print("  右目=左目のミラー複製")
            else:
                cur = base.copy()
                for i in range(len(EYES)):
                    print(f"  目{i}を inpaint...")
                    cur = run_and_fetch(cur, build_mask(w, h, only=i), STATES[st], args.seed + i)
        except Exception as e:
            print(f"  !! {st}: {e}"); continue
        dst = outdir / f"{st}_full.png"
        cv2.imwrite(str(dst), cur)
        print(f"  OK -> {dst}")

    print("\n次: crop_eye_sprites.py で目スプライトに切り出し")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
