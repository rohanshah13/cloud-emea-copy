#!/bin/bash
# Copyright 2020 Google and DeepMind.
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

# Script to train a model on SQuAD v1.1 or the English TyDiQA-GoldP train data.

REPO=$PWD
GPU=${1:-0}
MODEL=${2:-bert-base-multilingual-cased}
SRC=${3:-squad}
TGT=${4:-xquad}
DATA_DIR=${5:-"$REPO/data/"}
OUT_DIR=${6:-"$REPO/output/"}

export CUDA_VISIBLE_DEVICES=$GPU
TASK="squad"
BATCH_SIZE=4
GRAD_ACC=8

SEED=3

MAXL=384
LR=3e-4
NUM_EPOCHS=15

ADAPTER_LANG="en"
ADAPTER_NAME="en/wiki@ukp"

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlm-roberta"
  LR=1e-4
  GRAD_ACC=4
fi

# Model path where trained model should be stored
MODEL_PATH=$OUT_DIR/$SRC/my_${MODEL}_LR${LR}_EPOCH${NUM_EPOCHS}_maxlen${MAXL}_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}_s${SEED}
mkdir -p $MODEL_PATH

# Train either on the SQuAD or TyDiQa-GoldP English train file
if [ $SRC == 'squad' ]; then
  TASK_DATA_DIR=${DATA_DIR}/squad
  TRAIN_FILE=${TASK_DATA_DIR}/train-v1.1.json
  PREDICT_FILE=${TASK_DATA_DIR}/dev-v1.1.json
else
  TASK_DATA_DIR=${DATA_DIR}/tydiqa
  TRAIN_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-train/tydiqa.en.train.json
  PREDICT_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-dev/tydiqa.goldp.en.dev.json
fi

# train
CUDA_VISIBLE_DEVICES=$GPU
nohup python third_party/my_run_squad.py \
  --model_type ${MODEL_TYPE} \
  --model_name_or_path ${MODEL} \
  --seed $SEED \
  --do_train \
  --data_dir ${TASK_DATA_DIR} \
  --train_file ${TRAIN_FILE} \
  --predict_file ${PREDICT_FILE} \
  --per_gpu_train_batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --num_train_epochs ${NUM_EPOCHS} \
  --max_seq_length $MAXL \
  --doc_stride 128 \
  --save_steps 1000 \
  --overwrite_output_dir \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --warmup_steps 500 \
  --output_dir ${MODEL_PATH} \
  --weight_decay 0.0001 \
  --threads 8 \
  --train_lang en \
  --eval_lang en \
  --eval_all_checkpoints \
  --eval_patience -1 \
  --train_adapter \
  --adapter_config pfeiffer \
  --task_name "qna" \
  --load_lang_adapter $ADAPTER_NAME \
  --language $ADAPTER_LANG \
  --lang_adapter_config pfeiffer \
  --log_file $MODEL_PATH/train.log \
  --save_only_best_checkpoint $LC > $MODEL_PATH/detailed_train.log \

# predict
# bash scripts/predict_qa.sh $MODEL $MODEL_PATH $TGT $GPU $DATA_DIR
