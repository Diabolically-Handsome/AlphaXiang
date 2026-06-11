#!/bin/bash
# Launch v13 dense baseline first, then strategy-token arm with the same defaults.

set -euo pipefail

REPO="/mnt/c/Users/Laure/Desktop/AlphaXiang Transformer"
RUN_ONE="$REPO/tools/_run_v13_200m_train_arm.sh"

if [ ! -f "$RUN_ONE" ]; then
    echo "missing v13 train arm script: $RUN_ONE" >&2
    exit 1
fi

V13_ARM=dense bash "$RUN_ONE"
V13_ARM=strategy bash "$RUN_ONE"
