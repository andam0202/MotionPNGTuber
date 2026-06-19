#!/usr/bin/env python3
"""webカメラ→顔トラッキング(頭の向き＋まばたき＋視線)をUDP送信。Live2D的駆動用。

Windows venv(.venv-win, mediapipe)で実行。FaceLandmarker の
output_facial_transformation_matrixes から頭の yaw/pitch/roll を取り、
ARKitブレンドシェイプから blink/gaze を取り、JSONでWSLへ送る。

  <.venv-win>/python.exe face_pose_server.py --host <WSL_IP> --port 5007 [--preview]
依存: mediapipe, opencv-python, numpy。モデルは mediapipe-live/models/face_landmarker.task。
"""
import argparse
import json
import math
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = Path(r"C:\Users\mao0202\Documents\GitHub\idoladeus\projects\mediapipe-live\models\face_landmarker.task")


def euler_from_matrix(m: np.ndarray):
    """4x4変換行列の回転部→(yaw,pitch,roll)度。符号は経験的に調整可。"""
    R = m[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    else:
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw = 0.0
        roll = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
    return yaw, pitch, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5007)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.6, min_tracking_confidence=0.6))

    cap = None
    for bename, be in (("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)):
        c = cv2.VideoCapture(args.cam, be)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        ok, fr = c.read()
        if ok and fr is not None:
            print(f"[face] カメラ{args.cam} backend={bename} で開けました {fr.shape[1]}x{fr.shape[0]}")
            cap = c
            break
        c.release()
    if cap is None:
        print(f"[face] ERROR: カメラ{args.cam}を開けません(全backend失敗)"); return 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)
    print(f"[face] 顔トラッキング送信 -> {args.host}:{args.port}  (preview窓 q で終了)")
    t0 = time.time(); i = 0; sent = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int((time.time() - t0) * 1000) + i
        res = face.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        pkt = {"face": None, "head": None}
        if res and res.face_blendshapes:
            pkt["face"] = {c.category_name: round(c.score, 4) for c in res.face_blendshapes[0]}
        if res and res.facial_transformation_matrixes:
            m = np.array(res.facial_transformation_matrixes[0])
            yaw, pitch, roll = euler_from_matrix(m)
            pkt["head"] = {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2),
                           "tx": round(float(m[0, 3]), 2), "ty": round(float(m[1, 3]), 2)}
        try:
            sock.sendto(json.dumps(pkt).encode(), addr); sent += 1
        except OSError:
            pass
        if args.preview:
            h = pkt["head"]
            msg = (f"yaw{h['yaw']:+.0f} pitch{h['pitch']:+.0f} roll{h['roll']:+.0f}"
                   if h else "no face")
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("face_pose (q quit)", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
        i += 1
    cap.release()
    if args.preview:
        cv2.destroyAllWindows()
    print(f"[face] 終了 sent={sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
