#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#
alias python=python3

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"
cd "${CURRENT_DIR}"

# Set up the working directories.
DATASET_FOLDER="autovision"
MODEL_NAME="xception_65"
EXP_FOLDER="exp/${MODEL_NAME}"
DATASET_PATH="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/tfrecord"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Train several iterations.
NUM_ITERATIONS=90000
python "${WORK_DIR}"/train.py \
  --dataset="${DATASET_FOLDER}" \
  --logtostderr \
  --train_split="train" \
  --model_variant="${MODEL_NAME}" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size="385,385" \
  --train_batch_size=6 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/xception_65/model.ckpt-90000" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET_PATH}" \
  --img_channels=1 \
  --max_ckpt=5 \
  --num_clones=1 \
  --base_learning_rate=0.00001 \
  --init_exclude="[]" \
  --train_only="[]" \
  --aug_labels="[11]" \
  --aug_weights="[3]"

# Run evaluation.
python "${WORK_DIR}"/eval.py \
  --dataset="${DATASET_FOLDER}" \
  --logtostderr \
  --eval_split="val" \
  --model_variant="${MODEL_NAME}" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size="513,513" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${DATASET_PATH}" \
  --img_channels=1 \
  --eval_scales=1.00 \
  --max_number_of_evaluations=1

# Visualize the results.
python "${WORK_DIR}"/vis.py \
  --dataset="${DATASET_FOLDER}" \
  --logtostderr \
  --vis_split="val" \
  --model_variant="${MODEL_NAME}" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size="513,513" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${DATASET_PATH}" \
  --img_channels=1 \
  --max_number_of_iterations=1

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="${MODEL_NAME}" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=15 \
  --crop_size=513 \
  --crop_size=513 \
  --img_channels=1 \
  --inference_scales=1.0

python "${WORK_DIR}"/export_saved_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_DIR}/saved_model" \
  --model_variant="${MODEL_NAME}" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=15 \
  --crop_size=513 \
  --crop_size=513 \
  --img_channels=1 \
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
