#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TBIN="${TBIN:-1}"
if [[ -z "${IN_CHANS:-}" ]]; then
  IN_CHANS=$((2 * TBIN))
fi

if [[ -z "${DSEC_ROOT:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    DSEC_ROOT="/media/apollo-22/AT_2TB/dataset/t_1/DSEC_voxels_semantic_20s_tbin1"
  else
    DSEC_ROOT="/media/apollo-22/AT_2TB/dataset/t_1/DSEC_voxels_semantic_20s"
  fi
fi
if [[ -z "${MPX_ROOT:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    MPX_ROOT="/media/apollo-22/AT_2TB/dataset/t_1/1mpx_voxels_20s_tbin1"
  else
    MPX_ROOT="/media/apollo-22/AT_2TB/dataset/t_1/1mpx_voxels_20s"
  fi
fi
if [[ -z "${EVENTSCAPE_ROOT:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    EVENTSCAPE_ROOT="/media/apollo-22/AT_2TB/dataset/t_1/EventScape_voxels_tbin1/Town01-03_train"
  else
    EVENTSCAPE_ROOT="/media/apollo-22/AT_2TB/dataset/t_1/EventScape_voxels/Town01-03_train"
  fi
fi

DATASETS="[${DSEC_ROOT}/train,${MPX_ROOT}/train,${EVENTSCAPE_ROOT}]"
DATASET_FPCS="${DATASET_FPCS:-[8,8,8]}"
DATASET_WEIGHTS="${DATASET_WEIGHTS:-[0.15,0.7,0.15]}"

if [[ -z "${JEPA_FOLDER:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    JEPA_FOLDER="outputs/stage1_jepa_1gpu_mix_2ch_240x320_w015_070_015"
  else
    JEPA_FOLDER="outputs/stage1_jepa_1gpu_mix_${IN_CHANS}ch_240x320_tbin${TBIN}_w015_070_015"
  fi
fi
if [[ -z "${MAE_FOLDER:-}" ]]; then
  if [[ "${TBIN}" == "1" ]]; then
    MAE_FOLDER="outputs/stage1_mae_1gpu_mix_2ch_240x320_w015_070_015"
  else
    MAE_FOLDER="outputs/stage1_mae_1gpu_mix_${IN_CHANS}ch_240x320_tbin${TBIN}_w015_070_015"
  fi
fi

RUN_JEPA="${RUN_JEPA:-1}"
RUN_MAE="${RUN_MAE:-1}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-5000}"
PRETRAIN_IPE_SCALE="${PRETRAIN_IPE_SCALE:-1.25}"
PRETRAIN_START_LR="${PRETRAIN_START_LR:-1.0e-6}"
PRETRAIN_LR="${PRETRAIN_LR:-1.0e-4}"
PRETRAIN_FINAL_LR="${PRETRAIN_FINAL_LR:-1.0e-6}"
PRETRAIN_WARMUP="${PRETRAIN_WARMUP:-40}"
PRETRAIN_NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-16}"
VIS_INTERVAL="${VIS_INTERVAL:-500}"
VIS_MAX_TEMPORAL_SLICES="${VIS_MAX_TEMPORAL_SLICES:-8}"

COMMON_DATA_ARGS=(
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
)

COMMON_OPT_ARGS=(
  "optimization.epochs=${PRETRAIN_EPOCHS}"
  "optimization.ipe_scale=${PRETRAIN_IPE_SCALE}"
  "optimization.start_lr=${PRETRAIN_START_LR}"
  "optimization.lr=${PRETRAIN_LR}"
  "optimization.final_lr=${PRETRAIN_FINAL_LR}"
  "optimization.warmup=${PRETRAIN_WARMUP}"
  "optimization.clip_grad=1.0"
  "meta.use_tqdm=true"
  "meta.vis_interval=${VIS_INTERVAL}"
  "meta.vis_max_temporal_slices=${VIS_MAX_TEMPORAL_SLICES}"
)

if [[ "${RUN_JEPA}" == "1" ]]; then
  python3 scripts/train/run_train.py \
    "folder=${JEPA_FOLDER}" \
    "${COMMON_DATA_ARGS[@]}" \
    "model=vit_tiny_2_1" \
    "model.in_chans=${IN_CHANS}" \
    "model.lambda_value_vid=0.1" \
    "mask=stage1_event_activity_adaptive" \
    "loss.predict_all=true" \
    "loss.weight_distance_loss=false" \
    "${COMMON_OPT_ARGS[@]}"
fi

if [[ "${RUN_MAE}" == "1" ]]; then
  python3 scripts/mae/run_mae.py \
    "folder=${MAE_FOLDER}" \
    "model=vit_tiny_mae" \
    "model.in_chans=${IN_CHANS}" \
    "${COMMON_DATA_ARGS[@]}" \
    "optimization.weight_decay=0.04" \
    "optimization.final_weight_decay=0.04" \
    "${COMMON_OPT_ARGS[@]}"
fi
