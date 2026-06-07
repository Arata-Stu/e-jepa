#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export RUN_RANDOM="${RUN_RANDOM:-0}"
export RUN_ADAPTIVE_AREA="${RUN_ADAPTIVE_AREA:-0}"
export RUN_STRATEGIC="${RUN_STRATEGIC:-0}"
export RUN_STRATEGIC_SMALL="${RUN_STRATEGIC_SMALL:-1}"

exec ./scripts/experiments/pretrain_jepa_mask_240x320_ablation.sh "$@"
