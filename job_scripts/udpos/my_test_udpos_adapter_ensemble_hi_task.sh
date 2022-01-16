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
# LANGS="bn,ta,bho"
TRAIN_LANGS="en"

LANGS_ARRAY=( "mr,bho,ta" "fo,no,da" "be,uk,bg" "af,bm,yo")


NUM_EPOCHS=50
MAX_LENGTH=128
# SEED=0

# LANG_ADAPTER_NAME="am/wiki@ukp,bn/wiki@ukp,cs/wiki@ukp,de/wiki@ukp,el/wiki@ukp,es/wiki@ukp,et/wiki@ukp,eu/wiki@ukp,fi/wiki@ukp,fr/wiki@ukp,hi/wiki@ukp,hu/wiki@ukp,hy/wiki@ukp,id/wiki@ukp,ja/wiki@ukp,jv/wiki@ukp,ko/wiki@ukp,la/wiki@ukp,lv/wiki@ukp,mi/wiki@ukp,my/wiki@ukp,pt/wiki@ukp,ru/wiki@ukp,tr/wiki@ukp,vi/wiki@ukp,zh/wiki@ukp"
# ADAPTER_LANG="am,bn,cs,de,el,es,et,eu,fi,fr,hi,hu,hy,id,ja,jv,ko,la,lv,mi,my,pt,ru,tr,vi,zh"
# AW="0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464,0.038461538461538464"
LANG_ADAPTER_NAMES=( "en/wiki@ukp,hi/wiki@ukp,ar/wiki@ukp"  "en/wiki@ukp,is/wiki@ukp,de/wiki@ukp" "en/wiki@ukp,ru/wiki@ukp" "en/wiki@ukp,sw/wiki@ukp,am/wiki@ukp" )
ADAPTERS_LANGS=( "en,hi,ar" "en,is,de" "en,ru" "en,sw,am" )
AWS=( "0.33,0.33,0.33" "0.33,0.33,0.33" "0.5,0.5" "0.33,0.33,0.33" )

TASK_ADAPTER_NAME="udpos"

DATA_DIR=$DATA_DIR/${TASK}/${TASK}_processed_maxlen${MAX_LENGTH}/
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

for i in 0
do
LANGS=${LANGS_ARRAY[i]}
LANG_ADAPTER_NAME=${LANG_ADAPTER_NAMES[i]}
ADAPTER_LANG=${ADAPTERS_LANGS[i]}
AW=${AWS[i]}
OUTPUT_DIR="$OUT_DIR/${TASK}/my-${MODEL}-MaxLen${MAX_LENGTH}_${TASK_ADAPTER_NAME}_${ADAPTER_LANG}_ensemble_hi_task/"
mkdir -p $OUTPUT_DIR
echo $OUTPUT_DIR
for SEED in 1 2 3
do
MY_TASK_ADAPTER="output/${TASK}/my-bert-base-multilingual-cased-LR1e-4-epoch${NUM_EPOCHS}-MaxLen128-TrainLanghi_hi_s${SEED}/checkpoint-best/${TASK}/"

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
  --do_predict \
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
  --language $ADAPTER_LANG >> $OUTPUT_DIR/detailed_train.log
done
done