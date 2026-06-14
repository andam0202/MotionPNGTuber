#!/usr/bin/env python3
"""
オフライン口パク合成（Rhubarb Lip Sync viseme ベース）。

音声ファイル全体を Rhubarb で解析して viseme 系列（時刻付き）を取得し、
口消し動画 + mouth_track へ各フレームの viseme に対応する口スプライトを
合成して、音声付きの口パク動画を書き出す。

リアルタイム版（loop_lipsync_runtime_*.py）の音量ベース3段階分類と違い、
音素認識に基づくため母音・子音の口形が音に追従する（先読みできるオフライン専用）。

例:
  python render_offline_viseme.py \
    --audio workspace/dxm/dxm_loop3s.mp4 \
    --loop-video workspace/dxm/loop_mouthless.mp4 \
    --track workspace/dxm/mouth_track.npz \
    --mouth-dir workspace/dxm/mouth \
    --out workspace/dxm/out_viseme.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from motionpngtuber.lipsync_core import (
    MouthTrack,
    alpha_blit_rgb_safe,
    load_mouth_sprites,
    warp_rgba_to_quad,
)

# Rhubarb の Preston Blair 系 viseme → 既存口スプライトのマッピング（既定）
#   A: M/B/P 閉口        B: 控えめ開き(い/え/子音)   C: 開き(あ系中)
#   D: 大きく開く(あ)     E: 丸め(お)                F: すぼめ(う/F/V)
#   G: F/V 歯唇音         H: L                       X: 休止/無音
DEFAULT_VISEME_MAP = {
    "A": "closed",
    "B": "half",
    "C": "open",
    "D": "aa",    # 大きく開く（専用スプライト。無ければopenにフォールバック）
    "E": "o",     # お/丸く開く（専用スプライト。無ければuにフォールバック）
    "F": "o",     # う/F/V → 丸口で代用（u専用は小口でGrabCut切り抜きが破片化するため）
    "G": "o",
    "H": "half",
    "X": "closed",
}

# viseme別の開口ストレッチ（縦方向、上端固定で下方向に拡大＝開けると顎が下がる動き）。
# スプライト内の口の開きが控えめでも、開口visemeで口を大きく見せる。
# D(大きく開く)を最大に、X/A(閉)は等倍。--open-boost で全体に乗算。
DEFAULT_VISEME_STRETCH = {
    "A": 1.00,
    "X": 1.00,
    "B": 1.20,
    "C": 1.30,   # open（開きすぎ緩和のため抑えめ）
    "D": 1.05,   # aa は専用スプライトが大開きなので控えめ
    "E": 1.05,   # o も専用スプライト
    "F": 1.10,
    "G": 1.10,
    "H": 1.20,
}


def run_rhubarb(
    rhubarb_bin: str,
    wav_path: str,
    out_json: str,
    recognizer: str,
    dialog_path: str = "",
) -> None:
    cmd = [rhubarb_bin, "-r", recognizer, "-f", "json", "-o", out_json]
    if dialog_path:
        cmd += ["--dialogFile", dialog_path]
    cmd.append(wav_path)
    print("[rhubarb]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_viseme_cues(json_path: str) -> list[tuple[float, float, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cues = []
    for c in data.get("mouthCues", []):
        cues.append((float(c["start"]), float(c["end"]), str(c["value"])))
    cues.sort(key=lambda x: x[0])
    return cues


def parse_viseme_map(spec: str) -> dict[str, str]:
    m = dict(DEFAULT_VISEME_MAP)
    if spec:
        for kv in spec.split(","):
            kv = kv.strip()
            if not kv or "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            m[k.strip().upper()] = v.strip()
    return m


def crossfade_rgba_premult(prev_patch: np.ndarray, cur_patch: np.ndarray, t: float) -> np.ndarray:
    """2つのRGBAパッチをプリマルチプライドαでクロスフェード（暗いフリンジ防止）。

    t=0 で prev、t=1 で cur。サイズ・配置が一致する前提（同一frameのquadへwarp）。
    """
    t = float(np.clip(t, 0.0, 1.0))
    a_prev = prev_patch[..., 3:4].astype(np.float32) / 255.0
    a_cur = cur_patch[..., 3:4].astype(np.float32) / 255.0
    w_prev = a_prev * (1.0 - t)
    w_cur = a_cur * t
    out_a = w_prev + w_cur
    rgb = (
        prev_patch[..., :3].astype(np.float32) * w_prev
        + cur_patch[..., :3].astype(np.float32) * w_cur
    )
    safe_a = np.where(out_a > 1e-6, out_a, 1.0)
    rgb = rgb / safe_a
    out = np.empty_like(cur_patch)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(out_a[..., 0] * 255.0, 0, 255).astype(np.uint8)
    return out


def _to_union(
    prev: np.ndarray, px: int, py: int,
    cur: np.ndarray, cx: int, cy: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """配置・サイズが異なる2パッチを共通キャンバスに載せる。返り値 (cp, cc, x0, y0)。"""
    ph, pw = prev.shape[:2]
    ch, cw = cur.shape[:2]
    x0 = min(px, cx)
    y0 = min(py, cy)
    x1 = max(px + pw, cx + cw)
    y1 = max(py + ph, cy + ch)
    W, H = x1 - x0, y1 - y0
    cp = np.zeros((H, W, 4), dtype=np.uint8)
    cc = np.zeros((H, W, 4), dtype=np.uint8)
    cp[py - y0:py - y0 + ph, px - x0:px - x0 + pw] = prev
    cc[cy - y0:cy - y0 + ch, cx - x0:cx - x0 + cw] = cur
    return cp, cc, x0, y0


def union_crossfade(
    prev: np.ndarray, px: int, py: int,
    cur: np.ndarray, cx: int, cy: int,
    t: float,
) -> tuple[np.ndarray, int, int]:
    """配置・サイズが異なる2パッチを共通キャンバスでαクロスフェード。"""
    cp, cc, x0, y0 = _to_union(prev, px, py, cur, cx, cy)
    return crossfade_rgba_premult(cp, cc, t), x0, y0


def _warp_by_flow(img: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
    """オプティカルフロー flow を scale 倍した変位で img を warp。"""
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mapx = (gx + flow[..., 0] * scale).astype(np.float32)
    mapy = (gy + flow[..., 1] * scale).astype(np.float32)
    return cv2.remap(
        img, mapx, mapy, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )


def morph_rgba(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """2つのRGBAパッチ(同サイズ)を形状モーフィングで中間生成。

    口の不透明領域(アルファ)からオプティカルフローを計算し、a を前方へ、
    b を後方へ warp してから合成。αブレンドの「二重口ゴースト」を避け、
    口の輪郭が実際に動いて中間形状になる。
    """
    t = float(np.clip(t, 0.0, 1.0))
    fa = a[..., 3]
    fb = b[..., 3]
    # アルファ(口の形)でフロー推定。Farneback(numpy/cv2標準、追加依存なし)
    flow_ab = cv2.calcOpticalFlowFarneback(fa, fb, None, 0.5, 4, 21, 3, 7, 1.5, 0)
    flow_ba = cv2.calcOpticalFlowFarneback(fb, fa, None, 0.5, 4, 21, 3, 7, 1.5, 0)
    wa = _warp_by_flow(a, flow_ab, t)
    wb = _warp_by_flow(b, flow_ba, 1.0 - t)
    return crossfade_rgba_premult(wa, wb, t)


def union_morph(
    prev: np.ndarray, px: int, py: int,
    cur: np.ndarray, cx: int, cy: int,
    t: float,
) -> tuple[np.ndarray, int, int]:
    """配置・サイズが異なる2パッチを共通キャンバスで形状モーフィング。"""
    cp, cc, x0, y0 = _to_union(prev, px, py, cur, cx, cy)
    return morph_rgba(cp, cc, t), x0, y0


def stretch_quad_down(quad: np.ndarray, fy: float) -> np.ndarray:
    """quadを上端固定で下方向に縦拡大（口を開けると下顎が下がる動きを再現）。"""
    if abs(fy - 1.0) < 1e-3:
        return quad
    q = quad.astype(np.float32).copy()
    top = float(q[:, 1].min())
    q[:, 1] = top + (q[:, 1] - top) * float(fy)
    return q


def compose_patch(
    mouth: dict[str, np.ndarray],
    shape: str,
    quad: np.ndarray | None,
    fixed_x: int,
    fixed_y: int,
    stretch_y: float = 1.0,
    jaw_drop_px: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    """口スプライトを quad へwarp（quad=Noneなら固定位置）して (patch, x0, y0) を返す。

    jaw_drop_px: 口パッチ全体を下方向にずらす量[px]。開口量に比例させると
    顎が下がる錯覚を作れる（顔メッシュ変形の簡易代替）。
    """
    spr = mouth.get(shape, mouth["closed"])
    if quad is None:
        h = int(spr.shape[0] * stretch_y)
        spr2 = cv2.resize(spr, (spr.shape[1], h), interpolation=cv2.INTER_LINEAR) if h != spr.shape[0] else spr
        x = int(fixed_x - spr2.shape[1] // 2)
        y = int(fixed_y - spr.shape[0] // 2 + jaw_drop_px)  # 上端固定 + 顎ドロップ
        return spr2, x, int(y)
    q = stretch_quad_down(quad, stretch_y)
    if jaw_drop_px:
        q = q.astype(np.float32).copy()
        q[:, 1] += float(jaw_drop_px)
    return warp_rgba_to_quad(spr, q)


def main() -> int:
    ap = argparse.ArgumentParser(description="オフライン口パク合成(Rhubarb viseme)")
    ap.add_argument("--audio", required=True, help="音声ソース(wav/mp4/任意。ffmpegでWAV化)")
    ap.add_argument("--loop-video", required=True, help="口消し動画(背景)")
    ap.add_argument("--track", required=True, help="mouth_track.npz")
    ap.add_argument("--mouth-dir", required=True, help="口スプライトディレクトリ")
    ap.add_argument("--out", required=True, help="出力mp4")
    ap.add_argument("--rhubarb", default="", help="rhubarbバイナリパス(省略時は自動探索)")
    ap.add_argument("--recognizer", default="phonetic", choices=["phonetic", "pocketSphinx"],
                    help="phonetic=言語非依存(日本語等推奨) / pocketSphinx=英語")
    ap.add_argument("--dialog", default="", help="台本テキスト(pocketSphinx時に精度向上)")
    ap.add_argument("--viseme-json", default="", help="既存のRhubarb JSONを使う(再解析しない)")
    ap.add_argument("--viseme-map", default="", help='上書き例: "B=e,H=open"')
    ap.add_argument("--open-boost", type=float, default=1.0,
                    help="開口ストレッチの全体倍率(>1で口を大きく)")
    ap.add_argument("--fps", type=float, default=0.0, help="出力fps(0=動画fps)")
    ap.add_argument("--blend-sec", type=float, default=0.07,
                    help="viseme境界の遷移秒(0=無効。大きいほど滑らか)")
    ap.add_argument("--blend-mode", default="morph", choices=["morph", "alpha"],
                    help="morph=オプティカルフローで形状補間(中間口形) / alpha=単純クロスフェード")
    ap.add_argument("--jaw-drop", type=float, default=0.35,
                    help="開口量に比例した顎ドロップ係数(口パッチを下にずらし顎が動く錯覚。0=無効)")
    ap.add_argument("--fixed-x", type=int, default=640)
    ap.add_argument("--fixed-y", type=int, default=480)
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    rhubarb_bin = args.rhubarb
    if not rhubarb_bin:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, "tools", "Rhubarb-Lip-Sync-1.14.0-Linux", "rhubarb")
        rhubarb_bin = cand if os.path.isfile(cand) else "rhubarb"

    tmpdir = tempfile.mkdtemp(prefix="viseme_")
    wav_path = os.path.join(tmpdir, "audio.wav")

    # 1) 音声をRhubarb向けに16k mono WAV化
    print(f"[ffmpeg] extracting WAV -> {wav_path}")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", args.audio,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
        check=True,
    )

    # 2) viseme系列を取得
    viseme_json = args.viseme_json
    if not viseme_json:
        viseme_json = os.path.join(tmpdir, "viseme.json")
        run_rhubarb(rhubarb_bin, wav_path, viseme_json, args.recognizer, args.dialog)
    cues = load_viseme_cues(viseme_json)
    if not cues:
        print("[error] viseme cue が空です", file=sys.stderr)
        return 2
    audio_dur = cues[-1][1]
    vmap = parse_viseme_map(args.viseme_map)
    print(f"[info] viseme cues: {len(cues)}  audio_dur={audio_dur:.2f}s  map={vmap}")

    # 3) 背景動画とトラック・スプライトを読み込み
    cap = cv2.VideoCapture(args.loop_video)
    if not cap.isOpened():
        print(f"[error] 動画を開けません: {args.loop_video}", file=sys.stderr)
        return 2
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vfps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    vtotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out_fps = args.fps if args.fps > 0 else vfps

    # 背景フレームを全部メモリに（ループ参照のため。短尺動画前提）
    frames: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)  # BGR
    cap.release()
    if not frames:
        print("[error] 背景フレームが読めません", file=sys.stderr)
        return 2
    vtotal = len(frames)

    track = MouthTrack.load(args.track, vw, vh, policy="hold")
    mouth = load_mouth_sprites(args.mouth_dir, vw, vh)

    # 4) フレーム生成
    out_total = int(math.ceil(audio_dur * out_fps))
    silent_path = os.path.join(tmpdir, "silent.mp4")
    writer = cv2.VideoWriter(silent_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (vw, vh))

    cue_j = 0
    def shape_and_stretch(vis: str) -> tuple[str, float]:
        sh = vmap.get(vis, "closed")
        st = 1.0 + (DEFAULT_VISEME_STRETCH.get(vis, 1.0) - 1.0) * args.open_boost
        return sh, st

    for oi in range(out_total):
        t = oi / out_fps
        # 現在の cue を前進検索
        while cue_j + 1 < len(cues) and t >= cues[cue_j + 1][0]:
            cue_j += 1
        cur_shape, cur_str = shape_and_stretch(cues[cue_j][2])

        # 背景フレーム（15fps等の動画を出力fpsに合わせて最近傍リサンプル）
        vidx = int(t * vfps) % vtotal
        bg = frames[vidx].copy()  # BGR
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        quad = track.get_quad(vidx) if track is not None else None
        qh = float(quad[:, 1].max() - quad[:, 1].min()) if quad is not None else float(mouth["closed"].shape[0])
        cur_jaw = (cur_str - 1.0) * args.jaw_drop * qh
        patch, x0, y0 = compose_patch(mouth, cur_shape, quad, args.fixed_x, args.fixed_y, cur_str, cur_jaw)

        # viseme境界の両側クロスフェード（入口=前cue / 出口=次cue の近い方と合成）
        if args.blend_sec > 0:
            bs = args.blend_sec
            d_prev = t - cues[cue_j][0]
            d_next = cues[cue_j][1] - t
            nb = None
            w = 0.0
            if cue_j > 0 and d_prev < bs:
                nb, w = cue_j - 1, 0.5 * (1.0 - d_prev / bs)
            elif cue_j + 1 < len(cues) and d_next < bs:
                nb, w = cue_j + 1, 0.5 * (1.0 - d_next / bs)
            if nb is not None and w > 1e-3:
                nb_shape, nb_str = shape_and_stretch(cues[nb][2])
                if nb_shape != cur_shape or abs(nb_str - cur_str) > 1e-3:
                    nb_jaw = (nb_str - 1.0) * args.jaw_drop * qh
                    p_nb, nx, ny = compose_patch(
                        mouth, nb_shape, quad, args.fixed_x, args.fixed_y, nb_str, nb_jaw
                    )
                    if args.blend_mode == "morph":
                        patch, x0, y0 = union_morph(patch, int(x0), int(y0), p_nb, int(nx), int(ny), w)
                    else:
                        patch, x0, y0 = union_crossfade(patch, int(x0), int(y0), p_nb, int(nx), int(ny), w)

        alpha_blit_rgb_safe(bg_rgb, patch, int(x0), int(y0))
        writer.write(cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"[info] silent video: {silent_path}  frames={out_total} fps={out_fps}")

    # 5) 元音声をmux
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-i", silent_path, "-i", args.audio,
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", args.out],
        check=True,
    )
    print(f"[done] -> {args.out}")

    if not args.keep_temp:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print(f"[info] temp kept: {tmpdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
