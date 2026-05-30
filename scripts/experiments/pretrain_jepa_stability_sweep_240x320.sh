#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DSEC_ROOT="${DSEC_ROOT:-/media/apollo-22/AT_2TB/dataset/t_1/DSEC_voxels_semantic_20s_tbin1}"
MPX_ROOT="${MPX_ROOT:-/media/apollo-22/AT_2TB/dataset/t_1/1mpx_voxels_20s_tbin1}"
EVENTSCAPE_ROOT="${EVENTSCAPE_ROOT:-/media/apollo-22/AT_2TB/dataset/t_1/EventScape_voxels_tbin1/Town01-03_train}"

DATASETS="[${DSEC_ROOT}/train,${MPX_ROOT}/train,${EVENTSCAPE_ROOT}]"
DATASET_FPCS="${DATASET_FPCS:-[8,8,8]}"
DATASET_WEIGHTS="${DATASET_WEIGHTS:-[0.15,0.7,0.15]}"

SWEEP_EPOCHS="${SWEEP_EPOCHS:-1000}"
SWEEP_IPE_SCALE="${SWEEP_IPE_SCALE:-1.0}"
SWEEP_START_LR="${SWEEP_START_LR:-1.0e-6}"
SWEEP_FINAL_LR="${SWEEP_FINAL_LR:-1.0e-6}"
SWEEP_WARMUP="${SWEEP_WARMUP:-40}"
PRETRAIN_NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-16}"
VIS_INTERVAL="${VIS_INTERVAL:-200}"
VIS_MAX_TEMPORAL_SLICES="${VIS_MAX_TEMPORAL_SLICES:-8}"

RUN_SMALL_NO_CONTEXT_LOWLR="${RUN_SMALL_NO_CONTEXT_LOWLR:-1}"
RUN_ADAPTIVE_NO_CONTEXT_LOWLR="${RUN_ADAPTIVE_NO_CONTEXT_LOWLR:-1}"
RUN_SMALL_CONTEXT_TINY_LAMBDA="${RUN_SMALL_CONTEXT_TINY_LAMBDA:-1}"
RUN_SMALL_CONTEXT_LOW_LAMBDA="${RUN_SMALL_CONTEXT_LOW_LAMBDA:-1}"
RUN_ADAPTIVE_CONTEXT_LOW_LAMBDA="${RUN_ADAPTIVE_CONTEXT_LOW_LAMBDA:-1}"

COMMON_ARGS=(
  "data.datasets=${DATASETS}"
  "data.dataset_fpcs=${DATASET_FPCS}"
  "data.datasets_weights=${DATASET_WEIGHTS}"
  "data.batch_size=64"
  "data.num_workers=${PRETRAIN_NUM_WORKERS}"
  "data.crop_size=[240,320]"
  "data_aug.preserve_input_size=false"
  "data_aug.pad_to_hw=null"
  "data_aug.allowed_input_hw=null"
  "data_aug.random_resize_scale=[1.0,1.0]"
  "data_aug.random_resize_aspect_ratio=[1.3333333333,1.3333333333]"
  "model=vit_tiny_2_1"
  "model.in_chans=2"
  "loss.weight_distance_loss=false"
  "optimization.epochs=${SWEEP_EPOCHS}"
  "optimization.ipe_scale=${SWEEP_IPE_SCALE}"
  "optimization.start_lr=${SWEEP_START_LR}"
  "optimization.final_lr=${SWEEP_FINAL_LR}"
  "optimization.warmup=${SWEEP_WARMUP}"
  "optimization.clip_grad=1.0"
  "meta.use_tqdm=true"
  "meta.vis_interval=${VIS_INTERVAL}"
  "meta.vis_max_temporal_slices=${VIS_MAX_TEMPORAL_SLICES}"
)

run_jepa() {
  local name="$1"
  shift
  echo "==== Running ${name} ===="
  python3 scripts/train/run_train.py \
    "folder=outputs/${name}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

if [[ "${RUN_SMALL_NO_CONTEXT_LOWLR}" == "1" ]]; then
  run_jepa \
    "stage1_jepa_diag_01_smallmask_no_context_lr5e-5" \
    "mask=stage1_event_small" \
    "loss.predict_all=false" \
    "model.lambda_value_vid=0.0" \
    "optimization.lr=5.0e-5"
fi

if [[ "${RUN_ADAPTIVE_NO_CONTEXT_LOWLR}" == "1" ]]; then
  run_jepa \
    "stage1_jepa_diag_02_adaptive_no_context_lr5e-5" \
    "mask=stage1_event_activity_adaptive" \
    "loss.predict_all=false" \
    "model.lambda_value_vid=0.0" \
    "optimization.lr=5.0e-5"
fi

if [[ "${RUN_SMALL_CONTEXT_TINY_LAMBDA}" == "1" ]]; then
  run_jepa \
    "stage1_jepa_diag_03_smallmask_context_lambda0.02_lr5e-5" \
    "mask=stage1_event_small" \
    "loss.predict_all=true" \
    "model.lambda_value_vid=0.02" \
    "optimization.lr=5.0e-5"
fi

if [[ "${RUN_SMALL_CONTEXT_LOW_LAMBDA}" == "1" ]]; then
  run_jepa \
    "stage1_jepa_diag_04_smallmask_context_lambda0.05_lr1e-4" \
    "mask=stage1_event_small" \
    "loss.predict_all=true" \
    "model.lambda_value_vid=0.05" \
    "optimization.lr=1.0e-4"
fi

if [[ "${RUN_ADAPTIVE_CONTEXT_LOW_LAMBDA}" == "1" ]]; then
  run_jepa \
    "stage1_jepa_diag_05_adaptive_context_lambda0.05_lr1e-4" \
    "mask=stage1_event_activity_adaptive" \
    "loss.predict_all=true" \
    "model.lambda_value_vid=0.05" \
    "optimization.lr=1.0e-4"
fi
