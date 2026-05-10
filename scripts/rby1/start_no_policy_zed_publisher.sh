#!/bin/bash
# Start the non-policy ZED ZMQ publisher for the RBY1 head camera.
#
# Uses the unified zed_zmq_publisher.py with profile=rgbd:
#   - Stereo-only (head cam, ZED 2i USB).
#   - Publishes RGB + right RGB + depth + intrinsics at native resolution
#     (no resize), for downstream pixel -> 3D unprojection.
#   - Real depth mode (default: neural).
#   - Publish rate throttled to 1 Hz (decoupled from camera grab rate).
#
# Camera serial number:
#   head_camera = 32938613   (ZED 2i, USB stereo)
#
# Port matches rby1_standalone's head_camera ZMQ port (5558).
#
# Usage:
#   ./start_no_policy_zed_publisher.sh                       # hd1080, depth=neural
#   ./start_no_policy_zed_publisher.sh hd1200                # hd1200
#   ./start_no_policy_zed_publisher.sh hd1080 performance    # override depth mode
#   RES=hd1080 DEPTH=ultra ./start_no_policy_zed_publisher.sh

set -e

RESOLUTION=${1:-${RES:-hd1080}}
DEPTH_MODE=${2:-${DEPTH:-neural}}

# rgbd profile: name:type:serial:port:resolution:profile
CAMERAS=(
    "head_camera:stereo:32938613:5558:${RESOLUTION}:rgbd"
)

echo "Starting non-policy ZED publisher @ ${RESOLUTION}, 1 Hz publish, depth=${DEPTH_MODE}"
echo "  head_camera (stereo, S/N 32938613, rgbd) -> tcp://*:5558"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec conda run --no-capture-output -n policy python "${SCRIPT_DIR}/zed_zmq_publisher.py" \
    --cameras "${CAMERAS[@]}" \
    --resolution "$RESOLUTION" \
    --depth-mode "$DEPTH_MODE" \
    --publish-hz 1
