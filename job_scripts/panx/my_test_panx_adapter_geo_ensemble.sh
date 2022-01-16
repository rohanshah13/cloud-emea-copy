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
# LANGS="be,uk,bg"
TRAIN_LANGS="en"

LANGS_ARRAY=( "mr" "bn" "ta" "fo" "no" "da" "be" "uk" "bg" )

NUM_EPOCHS=100
MAX_LENGTH=128
SEED=0

# LANG_ADAPTER_NAME="en/wiki@ukp,ru/wiki@ukp"
# ADAPTER_LANG="en,ru"
# AW="0.5,0.5"

LANG_ADAPTER_NAMES=( "hi/wiki@ukp,my/wiki@ukp,fa/wiki@ukp,tk/wiki@ukp,vi/wiki@ukp,zh/wiki@ukp,ka/wiki@ukp,zh_yue/wiki@ukp,hy/wiki@ukp,ru/wiki@ukp" "hi/wiki@ukp,my/wiki@ukp,fa/wiki@ukp,vi/wiki@ukp,zh/wiki@ukp,tk/wiki@ukp,zh_yue/wiki@ukp,ru/wiki@ukp,bxr/wiki@ukp,cdo/wiki@ukp" "hi/wiki@ukp,my/wiki@ukp,fa/wiki@ukp,vi/wiki@ukp,tk/wiki@ukp,jv/wiki@ukp,zh_yue/wiki@ukp,id/wiki@ukp,zh/wiki@ukp,am/wiki@ukp" "is/wiki@ukp,en/wiki@ukp,se/wiki@ukp,et/wiki@ukp,fi/wiki@ukp,fr/wiki@ukp,kv/wiki@ukp,cs/wiki@ukp,de/wiki@ukp,eu/wiki@ukp" "et/wiki@ukp,lv/wiki@ukp,fi/wiki@ukp,en/wiki@ukp,se/wiki@ukp,cs/wiki@ukp,de/wiki@ukp,fr/wiki@ukp,is/wiki@ukp,hu/wiki@ukp" "cs/wiki@ukp,en/wiki@ukp,de/wiki@ukp,fr/wiki@ukp,lv/wiki@ukp,et/wiki@ukp,hu/wiki@ukp,fi/wiki@ukp,la/wiki@ukp,eu/wiki@ukp" "lv/wiki@ukp,et/wiki@ukp,cs/wiki@ukp,hu/wiki@ukp,de/wiki@ukp,el/wiki@ukp,fi/wiki@ukp,myv/wiki@ukp,mhr/wiki@ukp,tr/wiki@ukp" "hu/wiki@ukp,el/wiki@ukp,cs/wiki@ukp,de/wiki@ukp,lv/wiki@ukp,tr/wiki@ukp,la/wiki@ukp,et/wiki@ukp,xmf/wiki@ukp,myv/wiki@ukp" "el/wiki@ukp,hu/wiki@ukp,tr/wiki@ukp,la/wiki@ukp,cs/wiki@ukp,de/wiki@ukp,xmf/wiki@ukp,lv/wiki@ukp,hy/wiki@ukp,ka/wiki@ukp" )
ADAPTERS_LANGS=( "hi,my,fa,tk,vi,zh,ka,zh_yue,hy,ru" "hi,my,fa,vi,zh,tk,zh_yue,ru,bxr,cdo" "hi,my,fa,vi,tk,jv,zh_yue,id,zh,am" "is,en,se,et,fi,fr,kv,cs,de,eu" "et,lv,fi,en,se,cs,de,fr,is,hu" "cs,en,de,fr,lv,et,hu,fi,la,eu" "lv,et,cs,hu,de,el,fi,myv,mhr,tr" "hu,el,cs,de,lv,tr,la,et,xmf,myv" "el,hu,tr,la,cs,de,xmf,lv,hy,ka" )

TASK_ADAPTER_NAME="ner"

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

OUTPUT_DIR="$OUT_DIR/${TASK}/my-${MODEL}-MaxLen${MAX_LENGTH}_${TASK_ADAPTER_NAME}_geo_ensemble/"
mkdir -p $OUTPUT_DIR
echo $OUTPUT_DIR

for i in 1
do 
LANGS=${LANGS_ARRAY[i]}
LANG_ADAPTER_NAME=${LANG_ADAPTER_NAMES[i]}
ADAPTER_LANG=${ADAPTERS_LANGS[i]}
for SEED in 1 2 3
do
MY_TASK_ADAPTER="output/panx/my-bert-base-multilingual-cased-LR1e-4-epoch100-MaxLen128-TrainLangen_en_s${SEED}/checkpoint-best/ner/"

nohup python third_party/my_run_tag.py \
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
  --adapter_weight "equal" \
  --language $ADAPTER_LANG >> $OUTPUT_DIR/detailed_train.log
done
done