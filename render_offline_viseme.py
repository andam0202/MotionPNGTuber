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
    "D": "open",
    "E": "u",
    "F": "u",
    "G": "u",
    "H": "half",
    "X": "closed",
}

# viseme別の開口ストレッチ（縦方向、上端固定で下方向に拡大＝開けると顎が下がる動き）。
# スプライト内の口の開きが控えめでも、開口visemeで口を大きく見せる。
# D(大きく開く)を最大に、X/A(閉)は等倍。--open-boost で全体に乗算。
DEFAULT_VISEME_STRETCH = {
    "A": 1.00,
    "X": 1.00,
    "B": 1.25,
    "C": 1.55,
    "D": 1.90,
    "E": 1.20,
    "F": 1.15,
    "G": 1.15,
    "H": 1.30,
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
) -> tuple[np.ndarray, int, int]:
    """口スプライトを quad へwarp（quad=Noneなら固定位置）して (patch, x0, y0) を返す。"""
    spr = mouth.get(shape, mouth["closed"])
    if quad is None:
        h = int(spr.shape[0] * stretch_y)
        spr2 = cv2.resize(spr, (spr.shape[1], h), interpolation=cv2.INTER_LINEAR) if h != spr.shape[0] else spr
        x = int(fixed_x - spr2.shape[1] // 2)
        y = int(fixed_y - spr.shape[0] // 2)  # 上端固定
        return spr2, x, y
    return warp_rgba_to_quad(spr, stretch_quad_down(quad, stretch_y))


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
    ap.add_argument("--blend-sec", type=float, default=0.045,
                    help="viseme切替の時間クロスフェード秒(0=無効)")
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
    for oi in range(out_total):
        t = oi / out_fps
        # 現在の cue を前進検索
        while cue_j + 1 < len(cues) and t >= cues[cue_j + 1][0]:
            cue_j += 1
        cur_vis = cues[cue_j][2]
        cur_shape = vmap.get(cur_vis, "closed")
        cur_str = 1.0 + (DEFAULT_VISEME_STRETCH.get(cur_vis, 1.0) - 1.0) * args.open_boost

        # 背景フレーム（15fps等の動画を出力fpsに合わせて最近傍リサンプル）
        vidx = int(t * vfps) % vtotal
        bg = frames[vidx].copy()  # BGR
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        quad = track.get_quad(vidx) if track is not None else None
        patch, x0, y0 = compose_patch(mouth, cur_shape, quad, args.fixed_x, args.fixed_y, cur_str)

        # viseme切替の時間クロスフェード
        if args.blend_sec > 0 and cue_j > 0:
            elapsed = t - cues[cue_j][0]
            if elapsed < args.blend_sec:
                prev_vis = cues[cue_j - 1][2]
                prev_shape = vmap.get(prev_vis, "closed")
                prev_str = 1.0 + (DEFAULT_VISEME_STRETCH.get(prev_vis, 1.0) - 1.0) * args.open_boost
                if prev_shape != cur_shape or abs(prev_str - cur_str) > 1e-3:
                    p_prev, px0, py0 = compose_patch(
                        mouth, prev_shape, quad, args.fixed_x, args.fixed_y, prev_str
                    )
                    if p_prev.shape == patch.shape and px0 == x0 and py0 == y0:
                        alpha = elapsed / args.blend_sec
                        patch = crossfade_rgba_premult(p_prev, patch, alpha)

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
