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

REPO=$PWD
GPU=${1:-0}
MODEL=${2:-bert-base-multilingual-cased}
DATA_DIR=${3:-"$REPO/data/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU
TASK='udpos'
# LANGS="be,bg,da,ta,fo,uk"
LANGS="mr,bho,ta,fo,no,da,be,uk,bg"
TRAIN_LANGS="en"

NUM_EPOCHS=50
MAX_LENGTH=128
SEED=0

#Just switch the commented out lines to change the adapters used
# LANG_ADAPTER_NAME="am/wiki@ukp,bn/wiki@ukp,cs/wiki@ukp,de/wiki@ukp,el/wiki@ukp,es/wiki@ukp,et/wiki@ukp,eu/wiki@ukp,fi/wiki@ukp,fr/wiki@ukp,hi/wiki@ukp,hu/wiki@ukp,hy/wiki@ukp,id/wiki@ukp,ja/wiki@ukp,jv/wiki@ukp,ko/wiki@ukp,la/wiki@ukp,lv/wiki@ukp,mi/wiki@ukp,my/wiki@ukp,pt/wiki@ukp,ru/wiki@ukp,tr/wiki@ukp,vi/wiki@ukp,zh/wiki@ukp"
# ADAPTER_LANG="am,bn,cs,de,el,es,et,eu,fi,fr,hi,hu,hy,id,ja,jv,ko,la,lv,mi,my,pt,ru,tr,vi,zh"

LANG_ADAPTER_NAME="am/wiki@ukp,ar/wiki@ukp,bh/wiki@ukp,bn/wiki@ukp,bxr/wiki@ukp,cdo/wiki@ukp,cs/wiki@ukp,de/wiki@ukp,el/wiki@ukp,en/wiki@ukp,es/wiki@ukp,et/wiki@ukp,eu/wiki@ukp,fa/wiki@ukp,fi/wiki@ukp,fr/wiki@ukp,gn/wiki@ukp,hi/wiki@ukp,ht/wiki@ukp,hu/wiki@ukp,hy/wiki@ukp,id/wiki@ukp,ilo/wiki@ukp,is/wiki@ukp,ja/wiki@ukp,jv/wiki@ukp,ka/wiki@ukp,ko/wiki@ukp,kv/wiki@ukp,la/wiki@ukp,lv/wiki@ukp,mhr/wiki@ukp,mi/wiki@ukp,my/wiki@ukp,myv/wiki@ukp,pt/wiki@ukp,qu/wiki@ukp,ru/wiki@ukp,se/wiki@ukp,sw/wiki@ukp,tk/wiki@ukp,tr/wiki@ukp,vi/wiki@ukp,wo/wiki@ukp,xmf/wiki@ukp,zh/wiki@ukp,zh_yue/wiki@ukp"
ADAPTER_LANG="am,ar,bh,bn,bxr,cdo,cs,de,el,en,es,et,eu,fa,fi,fr,gn,hi,ht,hu,hy,id,ilo,is,ja,jv,ka,ko,kv,la,lv,mhr,mi,my,myv,pt,qu,ru,se,sw,tk,tr,vi,wo,xmf,zh,zh_yue"
# AW="0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464"
AW="equal_en"
EN_WEIGHT="0.1"

TASK_ADAPTER_NAME="udpos"

DATA_DIR=$DATA_DIR/${TASK}/${TASK}_processed_maxlen${MAX_LENGTH}/
OUTPUT_DIR="$OUT_DIR/${TASK}/my-${MODEL}-MaxLen${MAX_LENGTH}_${TASK_ADAPTER_NAME}_${ADAPTER_LANG}_ensemble_aw${AW}_en${EN_WEIGHT}/"
# TASK_ADAPTER="outputs/ner/"
# TASK_ADAPTER="output/panx/bert-base-multilingual-cased-LR1e-4-epoch100-MaxLen128-TrainLangen_en_s12/checkpoint-best/ner/"

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=8
  GRAD_ACC=4
fi

mkdir -p $OUTPUT_DIR

for SEED in 1 2 3
do
MY_TASK_ADAPTER="output/${TASK}/my-bert-base-multilingual-cased-LR1e-4-epoch${NUM_EPOCHS}-MaxLen128-TrainLangen_en_s${SEED}/checkpoint-best/${TASK_ADAPTER_NAME}/"

nohup time python third_party/my_run_tag.py \
  --predict_save_prefix "" \
  --calc_weight_step 0 \
  --per_gpu_eval_batch_size  32 \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps 1000 \
  --seed $SEED \
  --do_predict_dev \
  --predict_langs $LANGS \
  --train_langs $TRAIN_LANGS \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --eval_patience -1 \
  --overwrite_output_dir \
  --test_adapter \
  --adapter_config pfeiffer \
  --task_name $TASK_ADAPTER_NAME \
  --predict_task_adapter $MY_TASK_ADAPTER \
  --lang_adapter_config pfeiffer \
  --save_only_best_checkpoint $LC \
  --load_lang_adapter $LANG_ADAPTER_NAME \
  --adapter_weight $AW \
  --en_weight $EN_WEIGHT \
  --language $ADAPTER_LANG >> $OUTPUT_DIR/detailed_train.log 
done