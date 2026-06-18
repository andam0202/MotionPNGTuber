#!/usr/bin/env bash
# webカメラのアイトラッキング(まばたき)を WSL の MotionPNGTuber ランタイムへ送る。
#
#   webカメラ → pose_server.py(.venv-win, MediaPipe) ──UDP──▶ WSL ランタイム(--eye-track udp)
#
# WSL2 と Windows は localhost が別スタックなので、pose_server を WSL の実IP宛に送らせる。
# 流用元: idoladeus/projects/mediapipe-live(.venv-win と face_landmarker.task 構築済み)。
#
# 使い方:
#   1) 先に WSL で  ... loop_lipsync_..._auto.py --eye-track udp  を起動しておく
#   2) bash start_eye_tracker.sh            # プレビュー窓つきで起動(q で終了)
#   環境変数: PORT=5006  CAM=0  MODEL=lite|full|heavy  PREVIEW=1
set -u
ML="/mnt/c/Users/mao0202/Documents/GitHub/idoladeus/projects/mediapipe-live"
PYW="$ML/.venv-win/Scripts/python.exe"
SRV="$ML/scripts/pose_server.py"
PORT="${PORT:-5006}"
WSLIP="$(hostname -I | awk '{print $1}')"

[ -x "$PYW" ] || { echo "!! Windows venv が無い: $PYW"; echo "   先に: bash $ML/scripts/setup_win_env.sh"; exit 1; }
[ -s "$ML/models/face_landmarker.task" ] || { echo "!! face_landmarker.task が無い → setup_win_env.sh"; exit 1; }

PREVIEW_FLAG=""; [ "${PREVIEW:-1}" = "1" ] && PREVIEW_FLAG="--preview"
echo "=== webカメラ → WSL($WSLIP:$PORT) へまばたき送信 ==="
echo "    model=${MODEL:-lite} cam=${CAM:-0}  (プレビュー窓 q で終了)"
exec "$PYW" "$(wslpath -w "$SRV")" \
  --host="$WSLIP" --port="$PORT" --cam="${CAM:-0}" \
  --model="${MODEL:-lite}" --face-every=1 $PREVIEW_FLAG
