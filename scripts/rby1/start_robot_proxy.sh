#!/bin/bash
# Start the RBY1 robot proxy daemon on the Jetson.
#
# Exposes the RBY1 (gRPC arm + Dynamixel grippers) over ZMQ so a workstation
# can drive rollout remotely. Keeps the only hardware connection on this host.
#
# Prerequisites (not started by this script):
#   - RBY1 powered on and reachable at 192.168.30.1:50051
#   - No master-arm teleop server running (this script uses SDK commanding mode)
#
# Usage:
#   ./start_robot_proxy.sh
#
# Env-var overrides (defaults shown):
#   CONDA_ENV=policy_new
#   STATE_PORT=5560
#   ACTION_PORT=5561
#   STATE_RATE_HZ=100
#   ROBOT_ADDRESS=192.168.30.1:50051
#   GRIPPER_CURRENT_CAP=5.0
#   WITH_TORSO=false   # set to "true" to include torso joints
#   WITH_HEAD=false    # set to "true" to include head joints

set -e

CONDA_ENV="${CONDA_ENV:-policy_new}"
STATE_PORT="${STATE_PORT:-5560}"
ACTION_PORT="${ACTION_PORT:-5561}"
STATE_RATE_HZ="${STATE_RATE_HZ:-100}"
ROBOT_ADDRESS="${ROBOT_ADDRESS:-192.168.30.1:50051}"
GRIPPER_CURRENT_CAP="${GRIPPER_CURRENT_CAP:-5.0}"

EXTRA_ARGS=()
if [ "${WITH_TORSO:-false}" = "true" ]; then
    EXTRA_ARGS+=(--with-torso)
fi
if [ "${WITH_HEAD:-false}" = "true" ]; then
    EXTRA_ARGS+=(--with-head)
fi

# Mirror all output to a logfile too — `conda run --no-capture-output` plus a
# pipeline can buffer Python's stderr in some environments, so a persistent
# tail-able log is the cheapest way to keep visibility.
LOG_FILE="${LOG_FILE:-/tmp/robot_proxy.log}"

echo "Starting RBY1 robot proxy:"
echo "  conda env         : ${CONDA_ENV}"
echo "  robot address     : ${ROBOT_ADDRESS}"
echo "  state PUB port    : ${STATE_PORT}  (pubs at ${STATE_RATE_HZ} Hz)"
echo "  action PULL port  : ${ACTION_PORT}"
echo "  gripper current   : ${GRIPPER_CURRENT_CAP} A"
echo "  with_torso/head   : ${WITH_TORSO:-false} / ${WITH_HEAD:-false}"
echo "  log file          : ${LOG_FILE}"
echo ""

# `python -u` and PYTHONUNBUFFERED=1 force unbuffered stdout/stderr so logs
# stream live to both the terminal and the logfile.
PYTHONUNBUFFERED=1 conda run -n "${CONDA_ENV}" --no-capture-output python -u \
    /data/objsearch/rby1_policy_learning/scripts/rby1/robot_proxy.py \
    --robot-address "${ROBOT_ADDRESS}" \
    --state-port "${STATE_PORT}" \
    --action-port "${ACTION_PORT}" \
    --state-rate-hz "${STATE_RATE_HZ}" \
    --gripper-current-cap "${GRIPPER_CURRENT_CAP}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
