#!/bin/bash
#SBATCH --job-name=rayzer-dl3dv
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/rykerlin/slurm/rayzer_infer_%A_%a.out

set -euo pipefail

CONFIG="${CONFIG:-configs/rayzer_dl3dv.yaml}"
REPO_ROOT="${REPO_ROOT:-/home/rykerlin/Vision-Probe}"
DATASET_PATH="${DATASET_PATH:?DATASET_PATH is required}"
MODEL_PATH="${MODEL_PATH:-model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt}"
SAVE_ROOT="${SAVE_ROOT:-/scratch/rykerlin/rayzer_camera_tokens}"
VENV_PATH="${VENV_PATH:-/home/rykerlin/Vision-Probe/.venv}"
SEED="${SEED:?SEED is required}"
CHUNK_SIZE="${CHUNK_SIZE:-200}"
TOTAL_SCENES="${TOTAL_SCENES:-$(wc -l < "${DATASET_PATH}")}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}"

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
SCENE_START=$((ARRAY_ID * CHUNK_SIZE))
SCENE_END=$((SCENE_START + CHUNK_SIZE))
if (( SCENE_END > TOTAL_SCENES )); then
  SCENE_END=$TOTAL_SCENES
fi

if (( SCENE_START >= SCENE_END )); then
  echo "Skipping empty shard: seed=${SEED}, scenes=[${SCENE_START},${SCENE_END})"
  exit 0
fi

mkdir -p /scratch/rykerlin/slurm

cd "${REPO_ROOT}"

if [[ -d "${VENV_PATH}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

python inference_dl3dv_all.py --config "${CONFIG}" \
  training.dataset_name=data.dataset_dl3dv.Dataset \
  training.dataset_path="${DATASET_PATH}" \
  training.seed="${SEED}" \
  training.scene_start="${SCENE_START}" \
  training.scene_end="${SCENE_END}" \
  inference.if_inference=true \
  inference.batch_size_per_gpu="${INFERENCE_BATCH_SIZE}" \
  inference.view_idx_file_path=null \
  inference.model_path="${MODEL_PATH}" \
  inference.save_root="${SAVE_ROOT}"
