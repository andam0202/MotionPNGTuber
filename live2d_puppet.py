#!/usr/bin/env python3
"""Live2D的フェイストラッキング・パペット（see-throughレイヤー＋webカメラ頭ポーズ）。

see-through で分解した意味レイヤー(髪/顔/目/口…)を z順に重ね、webカメラの頭の向きで
2.5D的に動かす:
  - roll  → 全体を回転
  - yaw/pitch → レイヤー別パララックス平行移動(手前ほど大きく動く)
  - まばたき → 目レイヤー(虹彩/白目)を縦に潰し、背後の顔(のっぺらぼう肌)を見せて閉じる
  - 視線 → 虹彩レイヤーを平行移動
  - リップシンク → 口レイヤーを jawOpen(webカメラ) で縦に開く

頭ポーズ等は face_pose_server.py が UDP 送信(既定 port 5007)。
レイヤーは BGR のまま扱い表示する(色反転しないよう cvtColor しない)。

  env PULSE_SERVER=unix:/mnt/wslg/PulseServer uv run python live2d_puppet.py
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from motionpngtuber.eye_track_udp import EyeTrackReceiver

ST = "/mnt/c/Users/mao0202/Documents/GitHub/see-through/data/output_marigold/gura_base"
Y_OFF = 160

Z_ORDER = [
    ("wings", 0.0), ("tail", 0.0), ("back hair", 0.05),
    ("bottomwear", 0.1), ("legwear", 0.1), ("footwear", 0.1),
    ("topwear", 0.15), ("handwear", 0.15), ("neck", 0.2), ("neckwear", 0.2),
    ("ears", 0.35), ("earwear", 0.35), ("face", 0.4),
    ("eyewhite", 0.45), ("irides", 0.47), ("eyelash", 0.48),
    ("eyebrow", 0.5), ("nose", 0.5), ("mouth", 0.5), ("eyewear", 0.52),
    ("headwear", 0.6), ("front hair", 0.8), ("objects", 0.5),
]
EYE_LAYERS = {"eyewhite", "irides", "eyelash"}  # まつ毛/目輪郭も閉じる
EYE_CY = 436      # 目領域中心y(1280x960基準)
# 口リップシンク: ループ用に作った良質な口スプライト(aa=大開き等)を流用
MOUTH_DIR = "/mnt/c/Users/mao0202/Documents/GitHub/MotionPNGTuber/workspace/gura/mouth"
MOUTH_ANCHOR = (660, 660)   # see-through口中心(1280x960)
MOUTH_SCALE = 0.85          # 194スプライトの配置倍率


class Layer:
    __slots__ = ("name", "depth", "rgb", "a", "x0", "y0")

    def __init__(self, name, depth, rgb, a, x0, y0):
        self.name = name; self.depth = depth; self.rgb = rgb; self.a = a
        self.x0 = x0; self.y0 = y0  # クロップ左上(描画解像度)


def load_layers(st_dir: str, rs: float):
    """各層を描画解像度(rs)に縮小し、α>0のbboxで切り出して返す。"""
    layers = []
    for name, depth in Z_ORDER:
        p = os.path.join(st_dir, f"{name}.png")
        if not os.path.isfile(p):
            continue
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None or im.shape[2] < 4:
            continue
        im = im[Y_OFF:Y_OFF + 960]
        if rs != 1.0:
            im = cv2.resize(im, None, fx=rs, fy=rs, interpolation=cv2.INTER_AREA)
        a = im[:, :, 3].astype(np.float32) / 255.0
        if a.max() < 0.02:
            continue
        ys, xs = np.where(a > 0.02)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        rgb = im[y0:y1, x0:x1, :3].astype(np.float32)
        ac = a[y0:y1, x0:x1]
        layers.append(Layer(name, depth, rgb, ac, x0, y0))
    return layers


def load_mouth_states(rs: float):
    """ループ口スプライト(closed/half/aa)を口アンカーに配置し、Layerとして返す。"""
    states = {}
    ax, ay = MOUTH_ANCHOR[0] * rs, MOUTH_ANCHOR[1] * rs
    for name in ("closed", "half", "aa", "u", "o", "e", "open"):
        p = os.path.join(MOUTH_DIR, f"{name}.png")
        if not os.path.isfile(p):
            continue
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None or im.shape[2] < 4:
            continue
        sc = MOUTH_SCALE * rs
        im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        a = im[:, :, 3].astype(np.float32) / 255.0
        ys, xs = np.where(a > 0.02)
        if not len(xs):
            continue
        x0c, x1c, y0c, y1c = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        rgb = im[y0c:y1c, x0c:x1c, :3].astype(np.float32)
        ac = a[y0c:y1c, x0c:x1c]
        # スプライト中心をアンカーへ
        gx = int(ax - (x0c + x1c) / 2)
        gy = int(ay - (y0c + y1c) / 2)
        states[name] = Layer(name, 0.5, rgb, ac, x0c + gx, y0c + gy)
    return states


def pick_mouth(jaw, pucker, funnel, smile, gain, states):
    """口ブレンドシェイプ→母音口形(closed/half/aa/u/o/e)。フェイストラッキングの口形を反映。"""
    jg = jaw * gain
    # 実測: この話者は mouthFunnel がほぼ0で、お/う は mouthPucker の大小で分かれる
    # (お=puck~0.9, う=puck~0.1-0.5)。funnel は補助。
    pk = max(pucker, funnel)
    if jg < 0.12 and pk < 0.25 and smile < 0.30:
        name = "closed"
    elif pk > 0.62:
        name = "o"          # 強いすぼめ/丸め = お
    elif pk > 0.25:
        name = "u"          # 中程度のすぼめ = う
    elif jg > 0.50:
        name = "aa"
    elif smile > 0.35:
        name = "e"
    elif jg > 0.18:
        name = "half"
    else:
        name = "closed"
    if name in states:
        return name
    # フォールバック
    for fb in (name, "half", "aa", "open", "closed"):
        if fb in states:
            return fb
    return next(iter(states), None)


def _blend(canvas, rgb, a, x0, y0):
    """rgb(クロップ)を a で canvas の (x0,y0) に合成。はみ出しはクリップ。"""
    H, W = canvas.shape[:2]
    h, w = a.shape
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sx0, sy0 = cx0 - x0, cy0 - y0
    am = a[sy0:sy0 + (cy1 - cy0), sx0:sx0 + (cx1 - cx0)][:, :, None]
    reg = canvas[cy0:cy1, cx0:cx1]
    reg *= (1 - am)
    reg += rgb[sy0:sy0 + (cy1 - cy0), sx0:sx0 + (cx1 - cx0)] * am


def _squash(rgb, a, amount, cy_local):
    """amount(0..1)で縦縮小(cy_local中心)。目=閉じ/口=逆に開く時に使用。"""
    if abs(amount) < 0.02:
        return rgb, a
    sc = max(0.05, 1.0 - amount)
    h, w = a.shape
    M = np.float32([[1, 0, 0], [0, sc, cy_local * (1 - sc)]])
    a2 = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR)
    r2 = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR)
    return r2, a2


def main() -> int:
    ap = argparse.ArgumentParser(description="Live2D的パペット")
    ap.add_argument("--st-dir", default=ST)
    ap.add_argument("--port", type=int, default=5007)
    ap.add_argument("--k-yaw", type=float, default=2.2)
    ap.add_argument("--k-pitch", type=float, default=1.6)
    ap.add_argument("--k-roll", type=float, default=1.0)
    ap.add_argument("--render-scale", type=float, default=0.55, help="描画解像度(小=速い)")
    ap.add_argument("--mouth-gain", type=float, default=1.8, help="jawOpen→口開き倍率")
    ap.add_argument("--blink-thresh", type=float, default=0.42, help="この正規化blink以上で閉眼にきちっと切替")
    ap.add_argument("--bg", default="245,240,240")  # BGR
    args = ap.parse_args()

    rs = args.render_scale
    layers = load_layers(args.st_dir, rs)
    if not layers:
        print("ERROR: レイヤーが読めません"); return 1
    mouth_states = load_mouth_states(rs)
    # 閉じ目イラスト(ComfyUI生成, のっぺらぼう+実位置) を読み込み(rs縮小・bbox切出し)
    eye_closed = None
    cp = "/mnt/c/Users/mao0202/Documents/GitHub/MotionPNGTuber/workspace/gura/eye/closed.png"
    _ce = cv2.imread(cp, cv2.IMREAD_UNCHANGED)
    if _ce is not None and _ce.shape[2] == 4:
        if rs != 1.0:
            _ce = cv2.resize(_ce, None, fx=rs, fy=rs, interpolation=cv2.INTER_AREA)
        _a = _ce[:, :, 3].astype(np.float32) / 255.0
        _ys, _xs = np.where(_a > 0.02)
        if len(_xs):
            _x0, _x1, _y0, _y1 = _xs.min(), _xs.max() + 1, _ys.min(), _ys.max() + 1
            eye_closed = (_ce[_y0:_y1, _x0:_x1, :3].astype(np.float32), _a[_y0:_y1, _x0:_x1], _x0, _y0)
    print(f"[live2d] {len(layers)}層 @rs{rs}: {', '.join(l.name for l in layers)}")
    print(f"[live2d] 口リップシンク: {list(mouth_states)}  閉じ目イラスト: {'有' if eye_closed else '無'}")
    # canvasサイズ = レイヤーの最大範囲
    H = max(l.y0 + l.a.shape[0] for l in layers)
    W = max(l.x0 + l.a.shape[1] for l in layers)
    bg = np.array([int(x) for x in args.bg.split(",")], np.float32)
    pivot = (W / 2, H * 0.85)
    eye_cy_r = EYE_CY * rs

    rx = EyeTrackReceiver(port=args.port).start()
    print(f"[live2d] 頭ポーズ受信 :{args.port}  q で終了")
    sy = sp = sr = sbl = 0.0
    sjaw = spuck = sfun = ssmi = 0.0
    eye_is_closed = False  # 離散スイッチ状態(ヒステリシス)
    last = time.perf_counter(); fps = 0.0
    win = "live2d puppet (q quit)"
    while True:
        yaw, pitch, roll = rx.get_head()
        bl, br, _ = rx.get_blink()
        lx, ly = rx.get_look()
        jaw, pucker, funnel, smile = rx.get_mouth()
        e = 0.45
        sy += e * (yaw - sy); sp += e * (pitch - sp); sr += e * (roll - sr)
        blink = min(1.0, max(bl, br) * 1.4)
        sbl += e * (blink - sbl)
        sjaw += e * (jaw - sjaw); spuck += e * (pucker - spuck)
        sfun += e * (funnel - sfun); ssmi += e * (smile - ssmi)
        mouth_name = pick_mouth(sjaw, spuck, sfun, ssmi, args.mouth_gain, mouth_states)
        # 閉眼を離散切替(ヒステリシスでチラつき防止)
        if eye_is_closed:
            eye_is_closed = sbl > (args.blink_thresh - 0.12)
        else:
            eye_is_closed = sbl > args.blink_thresh

        canvas = np.empty((H, W, 3), np.float32); canvas[:] = bg
        for L in layers:
            dx = int(round(args.k_yaw * sy * L.depth * rs))
            dy = int(round(args.k_pitch * sp * L.depth * rs))
            if L.name == "mouth" and mouth_states:
                ms = mouth_states.get(mouth_name)
                if ms is None:
                    continue
                _blend(canvas, ms.rgb, ms.a, ms.x0 + dx, ms.y0 + dy)
                continue
            rgb, a = L.rgb, L.a
            if L.name == "irides":
                dx += int(round(lx * 18 * rs)); dy += int(round(-ly * 12 * rs))
            # 離散スイッチ: 閉眼判定時は開き目層を描かず、閉じ目イラストに切替
            if L.name in EYE_LAYERS and eye_closed is not None and eye_is_closed:
                if L.name == "eyelash":
                    ecr, eca, ex0, ey0 = eye_closed
                    _blend(canvas, ecr, eca, ex0 + dx, ey0 + dy)
                continue
            _blend(canvas, rgb, a, L.x0 + dx, L.y0 + dy)

        out = canvas.astype(np.uint8)
        if abs(sr) > 0.5:
            M = cv2.getRotationMatrix2D(pivot, args.k_roll * sr, 1.0)
            out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderValue=tuple(int(x) for x in bg))
        now = time.perf_counter(); fps = 0.9 * fps + 0.1 / max(1e-3, now - last); last = now
        cv2.putText(out, f"yaw{sy:+.0f} pitch{sp:+.0f} roll{sr:+.0f} fps{fps:.0f}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 220), 2)
        # 口形状キャリブレーション用に実値を表示(お/うの閾値調整に使う)
        cv2.putText(out, f"jaw{sjaw:.2f} puck{spuck:.2f} fun{sfun:.2f} smi{ssmi:.2f} ->{mouth_name}",
                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 40, 40), 2)
        cv2.imshow(win, out)  # 既にBGR(色反転しない)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
    rx.stop(); cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
