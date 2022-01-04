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
TASK='panx'
# LANGS="fo,no,da"
LANGS="mr"
# REM_LANGS="be"
TRAIN_LANGS="en"

# LANGS_ARRAY=( "mr,bho,ta" "fo,no,da" "be,uk,bg" )
# LANGS="da"

NUM_EPOCHS=100
MAX_LENGTH=128
# SEED=12

# LANG_ADAPTER_NAME="is/wiki@ukp"
# ADAPTER_LANG="is"
ADAPTER_LANGS1=( "ar" "ar" "bxr" "cdo" "cs" "de" "el" "en" "es" "et" "eu" "fa" "fi" "fr" "gn" "hi" "ht" "hu" "hy" "id" "ilo" "is" "ja" "jv" "ka" "ko" "kv" "la" "lv" "mhr" "mi" "my" "myv" "pt" "qu" "ru" "se" "sw" "tk" "tr" "vi" "wo" "xmf" "zh" "zh_yue" )
LANG_ADAPTER_NAMES1=( "ar/wiki@ukp" "ar/wiki@ukp" "bxr/wiki@ukp" "cdo/wiki@ukp" "cs/wiki@ukp" "de/wiki@ukp" "el/wiki@ukp" "en/wiki@ukp" "es/wiki@ukp" "et/wiki@ukp" "eu/wiki@ukp" "fa/wiki@ukp" "fi/wiki@ukp" "fr/wiki@ukp" "gn/wiki@ukp" "hi/wiki@ukp" "ht/wiki@ukp" "hu/wiki@ukp" "hy/wiki@ukp" "id/wiki@ukp" "ilo/wiki@ukp" "is/wiki@ukp" "ja/wiki@ukp" "jv/wiki@ukp" "ka/wiki@ukp" "ko/wiki@ukp" "kv/wiki@ukp" "la/wiki@ukp" "lv/wiki@ukp" "mhr/wiki@ukp" "mi/wiki@ukp" "my/wiki@ukp" "myv/wiki@ukp" "pt/wiki@ukp" "qu/wiki@ukp" "ru/wiki@ukp" "se/wiki@ukp" "sw/wiki@ukp" "tk/wiki@ukp" "tr/wiki@ukp" "vi/wiki@ukp" "wo/wiki@ukp" "xmf/wiki@ukp" "zh/wiki@ukp" "zh_yue/wiki@ukp" )

ADAPTER_LANGS2=( "et" "eu" "fa" "fi" "fr" "gn" "hi" "ht" "hu" )
LANG_ADAPTER_NAMES2=( "et/wiki@ukp" "eu/wiki@ukp" "fa/wiki@ukp" "fi/wiki@ukp" "fr/wiki@ukp" "gn/wiki@ukp" "hi/wiki@ukp" "ht/wiki@ukp" "hu/wiki@ukp" )

ADAPTER_LANGS3=( "hy" "id" "ilo" "is" "ja" "jv" "ka" "ko" "kv" )
LANG_ADAPTER_NAMES3=( "hy/wiki@ukp" "id/wiki@ukp" "ilo/wiki@ukp" "is/wiki@ukp" "ja/wiki@ukp" "jv/wiki@ukp" "ka/wiki@ukp" "ko/wiki@ukp" "kv/wiki@ukp" )

ADAPTER_LANGS4=( "ka" "ko" "kv" "la" "lv" "mhr" )
LANG_ADAPTER_NAMES4=( "ka/wiki@ukp" "ko/wiki@ukp" "kv/wiki@ukp" "la/wiki@ukp" "lv/wiki@ukp" "mhr/wiki@ukp" )

ADAPTER_LANGS5=( "se" "sw" "tk" "tr" "vi" "wo" )
LANG_ADAPTER_NAMES5=( "mi/wiki@ukp" "my/wiki@ukp" "myv/wiki@ukp" "pt/wiki@ukp" "qu/wiki@ukp" "ru/wiki@ukp" )

ADAPTER_LANGS6=( "xmf" "zh" "zh_yue" )
LANG_ADAPTER_NAMES6=( "xmf/wiki@ukp" "zh/wiki@ukp" "zh_yue/wiki@ukp" )

TASK_ADAPTER_NAME="ner"

DATA_DIR=$DATA_DIR/${TASK}/${TASK}_processed_maxlen${MAX_LENGTH}/
#This is where you choose the english adapter, and average over seeds
# TASK_ADAPTER="outputs/ner/"

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

for i in {0..0}
do
# LANGS=${LANGS_ARRAY[i]}
LANG_ADAPTER_NAME=${LANG_ADAPTER_NAMES1[i]}
ADAPTER_LANG=${ADAPTER_LANGS1[i]}
OUTPUT_DIR="$OUT_DIR/${TASK}/cloud-test-weights-100-${MODEL}-MaxLen${MAX_LENGTH}_${TASK_ADAPTER_NAME}_${ADAPTER_LANG}/"
mkdir -p $OUTPUT_DIR
echo $OUTPUT_DIR
for SEED in 1 2 3
do
MY_TASK_ADAPTER="output/${TASK}/my-bert-base-multilingual-cased-LR1e-4-epoch${NUM_EPOCHS}-MaxLen128-TrainLangen_en_s${SEED}/checkpoint-best/${TASK_ADAPTER_NAME}/"

nohup time python third_party/my_run_tag_editable.py \
  --predict_save_prefix "" \
  --calc_weight_step 0 \
  --per_gpu_eval_batch_size  1 \
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
  --log_file $OUTPUT_DIR/train_100.log \
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
  --get_attr \
  --language $ADAPTER_LANG >> $OUTPUT_DIR/detailed_train_100.log
done
done