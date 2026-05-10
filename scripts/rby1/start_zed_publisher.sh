#!/bin/bash
# Start the ZED ZMQ publisher for all 3 RBY1 cameras.
#
# Camera serial numbers:
#   head_cam        = 32938613     (ZED 2i, USB stereo)
#   right_wrist_cam = 314996496    (ZED XOne UHD, GMSL mono)
#   left_wrist_cam  = 311689153    (ZED XOne UHD, GMSL mono)
#
# Note: ZED XOne UHD cameras (the wrist cams) only support hd1080 and hd1200.
# hd720 and lower resolutions are NOT supported on those cameras, so the
# global default here is hd1080 (works for both wrist cams).
#
# Per-camera resolution override: append ':<resolution>' as an optional 5th
# field in each CameraSpec below (e.g. 'head_cam:stereo:32938613:5555:hd720').
# Cameras without a 5th field fall back to the global --resolution value.
# The head cam (ZED 2i) captures at hd720 here to reduce ZED compute while
# the wrist cams stay at hd1080 (their only supported resolution).
#
# Usage:
#   ./start_zed_publisher.sh                    # defaults: hd1080 @ 10 fps, resize to 640x480
#   ./start_zed_publisher.sh 15                 # hd1080 @ 15 fps, resize to 640x480
#   ./start_zed_publisher.sh 10 hd1200          # hd1200 @ 10 fps, resize to 640x480
#   ./start_zed_publisher.sh 10 hd1080 640 480  # explicit resize
#   ./start_zed_publisher.sh 10 hd1080 0        # disable resize (publish full 1920x1080)
#   FPS=15 RES=hd1080 ./start_zed_publisher.sh

set -e

FPS=${1:-${FPS:-10}}
RESOLUTION=${2:-${RES:-hd1080}}
RESIZE_W=${3:-640}
RESIZE_H=${4:-480}

CAMERAS=(
    "head_cam:stereo:32938613:5555:hd720"
    "right_wrist_cam:mono:306470987:5556"
    "left_wrist_cam:mono:301119863:5557"
)

echo "Starting ZED publisher @ ${RESOLUTION}, ${FPS} fps"
echo "  head_cam        (stereo, S/N 32938613)   -> tcp://*:5555"
echo "  right_wrist_cam (mono,   S/N 306470987) -> tcp://*:5556"
echo "  left_wrist_cam  (mono,   S/N 301119863) -> tcp://*:5557"
echo ""

RESIZE_ARGS=()
if [ "$RESIZE_W" != "0" ]; then
    RESIZE_ARGS=(--resize "$RESIZE_W" "$RESIZE_H")
fi

exec conda run -n policy python /data/objsearch/rby1_policy_learning/scripts/rby1/zed_zmq_publisher.py \
    --cameras "${CAMERAS[@]}" \
    --resolution "$RESOLUTION" \
    --fps "$FPS" \
    "${RESIZE_ARGS[@]}"
