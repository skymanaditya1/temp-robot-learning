#!/bin/bash
# Start the non-policy ZED ZMQ publisher for the RBY1 head camera.
#
# Unlike start_zed_publisher.sh, this script:
#   - Publishes ONLY the head camera (stereo-only; mono not supported here).
#   - Publishes RGB + right RGB + depth + intrinsics at native resolution
#     (no --resize), for downstream pixel -> 3D unprojection.
#   - Runs a real depth mode (default: neural).
#
# Camera serial number:
#   head_camera = 32938613   (ZED 2i, USB stereo)
#
# Port matches rby1_standalone's head_camera ZMQ port (5558).
#
# Usage:
#   ./start_no_policy_zed_publisher.sh                       # hd1080 @ 10 fps, depth=neural
#   ./start_no_policy_zed_publisher.sh 15                    # hd1080 @ 15 fps
#   ./start_no_policy_zed_publisher.sh 10 hd1200             # hd1200 @ 10 fps
#   ./start_no_policy_zed_publisher.sh 10 hd1080 performance # override depth mode
#   FPS=15 RES=hd1080 DEPTH=ultra ./start_no_policy_zed_publisher.sh

set -e

FPS=${1:-${FPS:-1}}
RESOLUTION=${2:-${RES:-hd1080}}
DEPTH_MODE=${3:-${DEPTH:-neural}}

CAMERAS=(
    "head_camera:stereo:32938613:5558"
)

echo "Starting non-policy ZED publisher @ ${RESOLUTION}, ${FPS} fps, depth=${DEPTH_MODE}"
echo "  head_camera (stereo, S/N 32938613) -> tcp://*:5558"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec conda run --no-capture-output -n policy python "${SCRIPT_DIR}/non_policy_zed_zmq_publisher.py" \
    --cameras "${CAMERAS[@]}" \
    --resolution "$RESOLUTION" \
    --fps "$FPS" \
    --depth-mode "$DEPTH_MODE"
