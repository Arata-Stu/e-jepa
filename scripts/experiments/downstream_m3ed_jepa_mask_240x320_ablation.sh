#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export DOWNSTREAM_TASK="${DOWNSTREAM_TASK:-m3ed_semantic}"

exec ./scripts/experiments/downstream_jepa_mask_240x320_ablation.sh "$@"
