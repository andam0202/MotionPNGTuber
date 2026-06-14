#!/usr/bin/env python3
"""フォルマント母音キャリブレーション。

あ/い/う/え/お を順に発声してもらい、あなたの声の F1/F2 を実測して
formant_calib.json に保存する。formant モードの口パク精度（個人差の吸収）が上がる。

使い方（WSLマイク）:
  env PULSE_SERVER=unix:/mnt/wslg/PulseServer \\
    uv run python calibrate_formant.py --device 4

実行後、loop_lipsync ... --lipsync-mode formant が自動で formant_calib.json を読む。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import sounddevice as sd

os.environ.setdefault("PULSE_SERVER", "unix:/mnt/wslg/PulseServer")

from motionpngtuber.formant_vowel import estimate_formants

VOWELS = [
    ("a", "あ"),
    ("i", "い"),
    ("u", "う"),
    ("e", "え"),
    ("o", "お"),
]


def measure_vowel(x: np.ndarray, sr: int) -> tuple[float, float] | None:
    """録音波形の中央部から F1/F2 の中央値を測る。"""
    n = len(x)
    mid = x[int(n * 0.25):int(n * 0.75)]  # 立ち上がり/減衰を避け中央のみ
    if len(mid) < sr // 10:
        return None
    hop = int(sr * 0.04)  # 40ms窓
    f1s, f2s = [], []
    for i in range(0, len(mid) - hop, hop // 2):
        fmts = estimate_formants(mid[i:i + hop], sr)
        if len(fmts) >= 2:
            f1s.append(fmts[0])
            f2s.append(fmts[1])
    if len(f1s) < 3:
        return None
    return float(np.median(f1s)), float(np.median(f2s))


def main() -> int:
    ap = argparse.ArgumentParser(description="フォルマント母音キャリブレーション")
    ap.add_argument("--device", type=int, default=4, help="入力デバイス番号(マイク)")
    ap.add_argument("--out", default="formant_calib.json", help="保存先JSON")
    ap.add_argument("--dur", type=float, default=2.0, help="各母音の録音秒数")
    args = ap.parse_args()

    try:
        dev = sd.query_devices(args.device, "input")
        sr = int(dev["default_samplerate"])
    except Exception as e:
        print(f"ERROR: デバイス{args.device}を開けません: {e}")
        print("利用可能デバイス:")
        print(sd.query_devices())
        return 1

    print(f"デバイス: {dev['name']}  sr={sr}")
    print("各母音をカウントダウン後に一定の高さで伸ばして発声してください。\n")

    calib: dict[str, list[float]] = {}
    for key, jp in VOWELS:
        print(f"次は [{jp}]。「{jp}ー」と伸ばす準備を...", flush=True)
        for c in (3, 2, 1):
            print(f"  {c}...", flush=True)
            time.sleep(0.8)
        print(f"  ●録音中（{args.dur:.0f}秒）! 「{jp}ー」", flush=True)
        rec = sd.rec(int(args.dur * sr), samplerate=sr, channels=1, device=args.device, dtype="float32")
        sd.wait()
        x = rec[:, 0]
        res = measure_vowel(x, sr)
        if res is None:
            print(f"  [警告] {jp}: フォルマントを測定できませんでした（音量不足？）。スキップ。\n")
            continue
        f1, f2 = res
        calib[key] = [round(f1, 1), round(f2, 1)]
        print(f"  {jp}({key}): F1={f1:.0f}Hz  F2={f2:.0f}Hz\n")

    if len(calib) < 3:
        print("ERROR: 測定できた母音が少なすぎます。マイク音量・デバイスを確認してください。")
        return 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"=== 保存: {args.out} ===")
    print(json.dumps(calib, ensure_ascii=False, indent=2))
    print("\nformant モードで自動的に読み込まれます。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
