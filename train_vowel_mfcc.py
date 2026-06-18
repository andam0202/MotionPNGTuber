#!/usr/bin/env python3
"""MFCC 母音分類器の学習データ収録＋学習。

あ/い/う/え/お を各 N 回ずつ録音し、MFCC 特徴を全フレーム抽出して
kNN を学習、vowel_mfcc.npz に保存する。フォルマント2点判定より
a/o/u の分離が大きく改善する（個人の声に最適化）。

使い方（WSLマイク）:
  env PULSE_SERVER=unix:/mnt/wslg/PulseServer \\
    uv run python train_vowel_mfcc.py --device 4 --reps 3

実行後、loop_lipsync ... --lipsync-mode mfcc が自動で vowel_mfcc.npz を読む。
"""
from __future__ import annotations

import argparse
import os
import queue
import time

import numpy as np
import sounddevice as sd

os.environ.setdefault("PULSE_SERVER", "unix:/mnt/wslg/PulseServer")

from motionpngtuber.vowel_mfcc import KNNVowel, frame_mfccs

VOWELS = [("a", "あ"), ("i", "い"), ("u", "う"), ("e", "え"), ("o", "お")]


def record_stream(dur: float, sr: int, device: int, channels: int = 1) -> np.ndarray:
    """InputStream(コールバック)で dur 秒録音（sd.rec は WSLg でハングする）。"""
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def cb(indata, frames, time_info, status):  # noqa: ANN001
        q.put(indata.copy())

    frames: list[np.ndarray] = []
    blocksize = max(256, int(sr * 0.02))
    with sd.InputStream(samplerate=sr, channels=channels, blocksize=blocksize,
                        dtype="float32", callback=cb, device=device, latency="low"):
        t0 = time.time()
        while time.time() - t0 < dur:
            try:
                frames.append(q.get(timeout=0.5))
            except queue.Empty:
                pass
    if not frames:
        return np.zeros(0, dtype=np.float32)
    data = np.concatenate(frames)
    return data[:, 0] if data.ndim > 1 else data


def _build_and_save(samples: dict[str, list[np.ndarray]], out: str, k: int) -> int:
    train: dict[str, np.ndarray] = {}
    for kk, v in samples.items():
        if v:
            train[kk] = np.concatenate(v, axis=0)
    if len(train) < 3:
        print("ERROR: 学習できる母音が少なすぎます。マイク音量/デバイスを確認。")
        return 1
    model = KNNVowel.train(train, k=k)
    model.save(out)
    print(f"=== 保存: {out} ===")
    for lab in model.labels:
        cnt = int(np.sum(model.y == model.labels.index(lab)))
        print(f"  {lab}: 学習サンプル {cnt}")
    print(f"\n母音: {model.labels}  /  kNN k={model.k}")
    print("loop_lipsync ... --lipsync-mode mfcc で自動的に読み込まれます。")
    return 0


def train_from_raw(args) -> int:  # noqa: ANN001
    if not os.path.isfile(args.raw):
        print(f"ERROR: キャッシュ {args.raw} がありません。先に録音してください。")
        return 1
    d = np.load(args.raw)
    sr = int(d["sr"][0])
    print(f"キャッシュ {args.raw} から再学習 (sr={sr})")
    samples: dict[str, list[np.ndarray]] = {k: [] for k, _ in VOWELS}
    for name in d.files:
        if name == "sr":
            continue
        key = name.split("_")[0]
        x = d[name]
        n = len(x)
        mid = x[int(n * 0.2):int(n * 0.8)]
        feats = frame_mfccs(mid, sr)
        if feats.shape[0] >= 3 and key in samples:
            samples[key].append(feats)
    for k, _ in VOWELS:
        tot = sum(f.shape[0] for f in samples[k])
        print(f"  {k}: 有声フレーム {tot}")
    return _build_and_save(samples, args.out, args.k)


def main() -> int:
    ap = argparse.ArgumentParser(description="MFCC母音分類器の学習")
    ap.add_argument("--device", type=int, default=4, help="入力デバイス番号(マイク)")
    ap.add_argument("--out", default="vowel_mfcc.npz", help="保存先(.npz)")
    ap.add_argument("--dur", type=float, default=1.6, help="各録音の秒数")
    ap.add_argument("--reps", type=int, default=3, help="各母音の繰り返し回数")
    ap.add_argument("--k", type=int, default=7, help="kNN の近傍数")
    ap.add_argument("--raw", default="vowel_mfcc_raw.npz", help="生録音キャッシュの保存/読込先")
    ap.add_argument("--from-raw", action="store_true",
                    help="録音せずキャッシュ(--raw)から特徴を再抽出して学習し直す")
    args = ap.parse_args()

    if args.from_raw:
        return train_from_raw(args)

    try:
        dev = sd.query_devices(args.device, "input")
        sr = int(dev["default_samplerate"])
    except Exception as e:
        print(f"ERROR: デバイス{args.device}を開けません: {e}")
        print(sd.query_devices())
        return 1

    print(f"デバイス: {dev['name']}  sr={sr}")
    print(f"各母音を {args.reps} 回ずつ録音します。少しずつ高さ/長さを変えて発声すると頑健になります。\n")

    samples: dict[str, list[np.ndarray]] = {k: [] for k, _ in VOWELS}
    raw_store: dict[str, np.ndarray] = {"sr": np.array([sr])}
    for rep in range(args.reps):
        print(f"=== ラウンド {rep + 1}/{args.reps} ===", flush=True)
        for key, jp in VOWELS:
            print(f"次は [{jp}]。準備...", flush=True)
            for c in (3, 2, 1):
                print(f"  {c}...", flush=True)
                time.sleep(0.7)
            print(f"  ●録音中（{args.dur:.1f}秒）! 「{jp}ー」", flush=True)
            x = record_stream(args.dur, sr, args.device)
            rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
            if x.size:
                raw_store[f"{key}_{rep}"] = x.astype(np.float32)
                n = len(x)
                mid = x[int(n * 0.2):int(n * 0.8)]  # 中央60%
                feats = frame_mfccs(mid, sr)
            else:
                feats = np.zeros((0, 12))
            if feats.shape[0] < 3:
                print(f"  [警告] {jp}: 特徴が取れず（rms={rms:.5f}）スキップ\n")
                continue
            samples[key].append(feats)
            print(f"  {jp}: 有声フレーム{feats.shape[0]} (rms={rms:.4f})\n")

    np.savez(args.raw, **raw_store)
    print(f"(生録音を {args.raw} にキャッシュしました。--from-raw で再学習できます)")

    return _build_and_save(samples, args.out, args.k)


if __name__ == "__main__":
    raise SystemExit(main())
