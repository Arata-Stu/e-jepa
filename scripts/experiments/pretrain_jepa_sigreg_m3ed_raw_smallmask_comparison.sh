#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

M3ED_ROOT="${M3ED_ROOT:-/mnt/ssd-4tb/dataset/m3ed}"
TBIN="${TBIN:-10}"
IN_CHANS="${IN_CHANS:-$((2 * TBIN))}"

RUN_STOPGRAD="${RUN_STOPGRAD:-1}"
RUN_SIGREG="${RUN_SIGREG:-1}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1000}"
PRETRAIN_IPE_SCALE="${PRETRAIN_IPE_SCALE:-1.0}"
PRETRAIN_START_LR="${PRETRAIN_START_LR:-1.0e-6}"
PRETRAIN_LR="${PRETRAIN_LR:-5.0e-5}"
PRETRAIN_FINAL_LR="${PRETRAIN_FINAL_LR:-1.0e-6}"
PRETRAIN_WARMUP="${PRETRAIN_WARMUP:-40}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8}"
PRETRAIN_NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-8}"
PRETRAIN_SAVE_EVERY_FREQ="${PRETRAIN_SAVE_EVERY_FREQ:-250}"

SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.09}"
SIGREG_NUM_PROJ="${SIGREG_NUM_PROJ:-1024}"
SIGREG_PROJECTION_CHUNK_SIZE="${SIGREG_PROJECTION_CHUNK_SIZE:-64}"
SIGREG_MAX_TOKENS="${SIGREG_MAX_TOKENS:-512}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RUN_PREFIX="${RUN_PREFIX:-m3ed_raw_jepa_smallmask}"

COMMON_ARGS=(
  "data=m3ed_raw"
  "data.datasets=[${M3ED_ROOT}]"
  "data.dataset_fpcs=[16]"
  "data.batch_size=${PRETRAIN_BATCH_SIZE}"
  "data.num_workers=${PRETRAIN_NUM_WORKERS}"
  "data.t_bins=${TBIN}"
  "data.crop_size=[240,320]"
  "data_aug.preserve_input_size=false"
  "data_aug.pad_to_hw=null"
  "data_aug.allowed_input_hw=null"
  "data_aug.random_resize_scale=[1.0,1.0]"
  "data_aug.random_resize_aspect_ratio=[1.3333333333,1.3333333333]"
  "model=vit_tiny_2_1"
  "model.in_chans=${IN_CHANS}"
  "model.lambda_value_vid=0.0"
  "model.lambda_progressive=false"
  "mask=stage1_event_sigreg_small"
  "loss.predict_all=false"
  "loss.weight_distance_loss=false"
  "optimization.epochs=${PRETRAIN_EPOCHS}"
  "optimization.ipe_scale=${PRETRAIN_IPE_SCALE}"
  "optimization.start_lr=${PRETRAIN_START_LR}"
  "optimization.lr=${PRETRAIN_LR}"
  "optimization.final_lr=${PRETRAIN_FINAL_LR}"
  "optimization.warmup=${PRETRAIN_WARMUP}"
  "optimization.clip_grad=1.0"
  "meta.use_tqdm=true"
  "meta.tensorboard=true"
  "meta.save_every_freq=${PRETRAIN_SAVE_EVERY_FREQ}"
)

EXTRA_ARGS=("$@")

run_comparison() {
  local run_name="$1"
  local collapse_mode="$2"
  shift 2
  echo "==== Running ${run_name}: collapse_prevention=${collapse_mode} ===="
  python3 scripts/train/run_train.py \
    "folder=${OUTPUT_ROOT}/${RUN_PREFIX}_${run_name}" \
    "${COMMON_ARGS[@]}" \
    "loss.collapse_prevention=${collapse_mode}" \
    "$@" \
    "${EXTRA_ARGS[@]}"
}

if [[ "${RUN_STOPGRAD}" == "1" ]]; then
  run_comparison "stopgrad_ema" "stopgrad_ema"
fi

if [[ "${RUN_SIGREG}" == "1" ]]; then
  run_comparison \
    "sigreg_w${SIGREG_WEIGHT}" \
    "sigreg" \
    "loss.sigreg.weight=${SIGREG_WEIGHT}" \
    "loss.sigreg.num_proj=${SIGREG_NUM_PROJ}" \
    "loss.sigreg.projection_chunk_size=${SIGREG_PROJECTION_CHUNK_SIZE}" \
    "loss.sigreg.max_tokens=${SIGREG_MAX_TOKENS}"
fi
