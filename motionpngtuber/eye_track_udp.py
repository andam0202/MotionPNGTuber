"""webカメラのアイトラッキング状態を UDP/JSON で受信する（WSL側）。

Windows の pose_server.py(mediapipe FaceLandmarker, output_face_blendshapes=True)が
`--host=<WSL_IP> --port=<PORT>` で送る ARKit ブレンドシェイプ(eyeBlinkLeft/Right,
eyeLookIn/Out/Up/Down 等)を受信し、最新のまばたき/視線状態をスレッドセーフに保持する。

  Windows webcam → pose_server.py ──UDP/JSON──▶ EyeTrackReceiver(WSL) → ランタイムが瞬き反映

face は pose_server が face_every 間隔で送るため None の回がある。最後の有効値を保つ。
"""
from __future__ import annotations

import json
import socket
import threading
import time


class EyeTrackReceiver:
    """UDP を待ち受け、最新のまばたき/視線スコアを保持するデーモン受信器。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 5006) -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._blink_l = 0.0
        self._blink_r = 0.0
        self._look_x = 0.0   # +右 / -左（被写体視点）
        self._look_y = 0.0   # +上 / -下
        self._last_face_t = 0.0   # 最後に face を受けた時刻
        self._last_pkt_t = 0.0    # 最後に何かを受けた時刻
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> "EyeTrackReceiver":
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.settimeout(0.5)
        self._sock = s
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            now = time.time()
            self._last_pkt_t = now
            try:
                pkt = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            face = pkt.get("face")
            if not face:
                continue
            bl = float(face.get("eyeBlinkLeft", 0.0))
            br = float(face.get("eyeBlinkRight", 0.0))
            # 視線: ARKit の look(In/Out/Up/Down) を左右上下に合成（任意・将来用）
            lx = (float(face.get("eyeLookOutLeft", 0.0)) - float(face.get("eyeLookInLeft", 0.0))
                  + float(face.get("eyeLookInRight", 0.0)) - float(face.get("eyeLookOutRight", 0.0))) * 0.5
            ly = (float(face.get("eyeLookUpLeft", 0.0)) + float(face.get("eyeLookUpRight", 0.0))
                  - float(face.get("eyeLookDownLeft", 0.0)) - float(face.get("eyeLookDownRight", 0.0))) * 0.5
            with self._lock:
                self._blink_l, self._blink_r = bl, br
                self._look_x, self._look_y = lx, ly
                self._last_face_t = now

    def get_blink(self) -> tuple[float, float, float]:
        """(左まばたき, 右まばたき, 最後のface受信からの経過秒)。0=開, 1=閉。"""
        with self._lock:
            age = time.time() - self._last_face_t if self._last_face_t else 1e9
            return self._blink_l, self._blink_r, age

    def get_look(self) -> tuple[float, float]:
        with self._lock:
            return self._look_x, self._look_y

    def connected(self, timeout: float = 1.0) -> bool:
        return (time.time() - self._last_pkt_t) < timeout if self._last_pkt_t else False

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass


def _main() -> int:
    """単体テスト: 受信したまばたき/視線をライブ表示。"""
    import argparse
    ap = argparse.ArgumentParser(description="アイトラUDP受信テスト")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5006)
    args = ap.parse_args()
    rx = EyeTrackReceiver(args.host, args.port).start()
    print(f"[eye-rx] listening {args.host}:{args.port}  (Ctrl-C で終了)")
    try:
        while True:
            time.sleep(0.1)
            bl, br, age = rx.get_blink()
            lx, ly = rx.get_look()
            status = "OK" if age < 1.0 else f"(face未受信 {age:.0f}s)"
            bar_l = "#" * int(bl * 20)
            bar_r = "#" * int(br * 20)
            print(f"\r blinkL={bl:.2f}[{bar_l:<20}] R={br:.2f}[{bar_r:<20}] look=({lx:+.2f},{ly:+.2f}) {status}   ",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\n[eye-rx] stop")
        rx.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
