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
OUT_DIR=${6:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU
TASK="squad"
BATCH_SIZE=4
GRAD_ACC=8

SEEDS=( 1 2 3 )

#Top 10 based on English results for each seed
ADAPTER_LANGS=( "en,zh,es,ja,it,ta,ru,el,tr,ar,vi" "ru,en,el,ta,de,it,es,id,th,et,zh" "en,ru,zh,ja,el,es,id,it,de,ta,ar" )
ADAPTER_NAMES=( "en/wiki@ukp,zh/wiki@ukp,es/wiki@ukp,ja/wiki@ukp,it/wiki@ukp,ta/wiki@ukp,ru/wiki@ukp,el/wiki@ukp,tr/wiki@ukp,ar/wiki@ukp,vi/wiki@ukp" "ru/wiki@ukp,en/wiki@ukp,el/wiki@ukp,ta/wiki@ukp,de/wiki@ukp,it/wiki@ukp,es/wiki@ukp,id/wiki@ukp,th/wiki@ukp,et/wiki@ukp,zh/wiki@ukp" "en/wiki@ukp,ru/wiki@ukp,zh/wiki@ukp,ja/wiki@ukp,el/wiki@ukp,es/wiki@ukp,id/wiki@ukp,it/wiki@ukp,de/wiki@ukp,ta/wiki@ukp,ar/wiki@ukp" )

#Top 5 based on English results for each seed
# ADAPTER_LANGS=( "en,zh,es,ja,it" "ru,en,el,ta,de" "en,ru,zh,ja,el" )
# ADAPTER_NAMES=( "en/wiki@ukp,zh/wiki@ukp,es/wiki@ukp,ja/wiki@ukp,it/wiki@ukp" "ru/wiki@ukp,en/wiki@ukp,el/wiki@ukp,ta/wiki@ukp,de/wiki@ukp" "en/wiki@ukp,ru/wiki@ukp,zh/wiki@ukp,ja/wiki@ukp,el/wiki@ukp" )

#Top 3 based on English results for each seed
# ADAPTER_LANGS=( "en,zh,es" "ru,en,el" "en,ru,zh" )
# ADAPTER_NAMES=( "en/wiki@ukp,zh/wiki@ukp,es/wiki@ukp" "ru/wiki@ukp,en/wiki@ukp,el/wiki@ukp" "en/wiki@ukp,ru/wiki@ukp,zh/wiki@ukp" )

MAXL=384
LR=3e-4
NUM_EPOCHS=15

TRAIN_LANG="en"
PREDICT_LANGS="zh"
# PREDICT_LANGS="en"

TASK_ADAPTER_NAME="qna"

K=10

OUTPUT_DIR="$OUT_DIR/$TASK/my-${MODEL}-MaxLen${MAXL}_${TASK_ADAPTER_NAME}_ensemble_en_top${K}_target/"

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
MODEL_PATH=output/$SRC/my_${MODEL}_LR${LR}_EPOCH${NUM_EPOCHS}_maxlen${MAXL}_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}_s${SEED}
mkdir -p $OUTPUT_DIR
echo $OUTPUT_DIR
# Train either on the SQuAD or TyDiQa-GoldP English train file
if [ $SRC == 'squad' ]; then
  TASK_DATA_DIR=${DATA_DIR}/${TGT}
  TRAIN_FILE=${TASK_DATA_DIR}/train-v1.1.json
  PREDICT_FILE=${TASK_DATA_DIR}/dev-v1.1.json
else
  TASK_DATA_DIR=${DATA_DIR}/tydiqa
  TRAIN_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-train/tydiqa.en.train.json
  PREDICT_FILE=${TASK_DATA_DIR}/tydiqa-goldp-v1.1-dev/tydiqa.goldp.en.dev.json
fi

# train
CUDA_VISIBLE_DEVICES=$GPU

for i in 0 1 2
do
SEED=${SEEDS[i]}
ADAPTER_LANG=${ADAPTER_LANGS[i]}
ADAPTER_NAME=${ADAPTER_NAMES[i]}
MY_TASK_ADAPTER="output/${SRC}/my_${MODEL}_LR${LR}_EPOCH${NUM_EPOCHS}_maxlen${MAXL}_batchsize${BATCH_SIZE}_gradacc${GRAD_ACC}_s${SEED}/checkpoint-best/qna"
nohup python third_party/my_run_squad.py \
  --model_type ${MODEL_TYPE} \
  --model_name_or_path ${MODEL} \
  --seed $SEED \
  --do_predict \
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
  --output_dir ${OUTPUT_DIR} \
  --weight_decay 0.0001 \
  --threads 8 \
  --train_lang $TRAIN_LANG \
  --predict_langs $PREDICT_LANGS \
  --eval_all_checkpoints \
  --eval_patience -1 \
  --test_adapter \
  --adapter_config pfeiffer \
  --task_name $TASK_ADAPTER_NAME \
  --predict_task_adapter $MY_TASK_ADAPTER \
  --load_lang_adapter $ADAPTER_NAME \
  --language $ADAPTER_LANG \
  --lang_adapter_config pfeiffer \
  --adapter_weight "equal" \
  --calc_weight_step 0 \
  --log_file $OUTPUT_DIR/train_$PREDICT_LANGS.log \
  --save_only_best_checkpoint $LC >> $OUTPUT_DIR/detailed_train_$PREDICT_LANGS.log
done