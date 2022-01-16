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
LANGS="gn"
# LANGS="mr,bn,no"
# LANGS="af,bm,yo"
# LANGS="bn"
TRAIN_LANGS="en"

NUM_EPOCHS=100
MAX_LENGTH=128
SEED=0


#Top 10 based on english results
# LANG_ADAPTER_NAME='en/wiki@ukp,pt/wiki@ukp,id/wiki@ukp,tr/wiki@ukp,cs/wiki@ukp,vi/wiki@ukp,eu/wiki@ukp,zh_yue/wiki@ukp,fa/wiki@ukp,es/wiki@ukp'
# ADAPTER_LANG="en,pt,id,tr,cs,vi,eu,zh_yue,fa,es"
SEEDS=( 1 2 3 )
K=10

#Top 10 based on english results for each seed
LANG_ADAPTER_NAMES=( 'en/wiki@ukp,pt/wiki@ukp,id/wiki@ukp,cs/wiki@ukp,tr/wiki@ukp,eu/wiki@ukp,zh_yue/wiki@ukp,vi/wiki@ukp,fr/wiki@ukp,gn/wiki@ukp' 'en/wiki@ukp,pt/wiki@ukp,id/wiki@ukp,tr/wiki@ukp,cs/wiki@ukp,vi/wiki@ukp,eu/wiki@ukp,fa/wiki@ukp,zh_yue/wiki@ukp,gn/wiki@ukp' 'en/wiki@ukp,pt/wiki@ukp,id/wiki@ukp,tr/wiki@ukp,vi/wiki@ukp,fa/wiki@ukp,eu/wiki@ukp,zh_yue/wiki@ukp,cs/wiki@ukp,gn/wiki@ukp' )
ADAPTER_LANGS=( "en,pt,id,cs,tr,eu,zh_yue,vi,fr,gn" "en,pt,id,tr,cs,vi,eu,fa,zh_yue,gn" "en,pt,id,tr,vi,fa,eu,zh_yue,cs,gn" )

#Top 5 based on english results for each seed
# LANG_ADAPTER_NAMES=( 'en/wiki@ukp,pt/wiki@ukp,id/wiki@ukp,cs/wiki@ukp,tr/wiki@ukp' 'en/wiki@ukp,pt/wiki@ukp,id/wiki@ukp,tr/wiki@ukp,cs/wiki@ukp' 'en/wiki@ukp,pt/wiki@ukp,id/wiki@ukp,tr/wiki@ukp,vi/wiki@ukp' )
# ADAPTER_LANGS=( "en,pt,id,cs,tr" "en,pt,id,tr,cs" "en,pt,id,tr,vi" )

#Top 3 based on english results for each seed
# LANG_ADAPTER_NAMES=( 'en/wiki@ukp,pt/wiki@ukp,id/wiki@ukp' 'en/wiki@ukp,pt/wiki@ukp,id/wiki@ukp' 'en/wiki@ukp,pt/wiki@ukp,id/wiki@ukp' )
# ADAPTER_LANGS=( "en,pt,id" "en,pt,id" "en,pt,id" )

#Top 1 based on english results for each seed
# LANG_ADAPTER_NAMES=( 'pt/wiki@ukp' 'pt/wiki@ukp' 'pt/wiki@ukp' )
# ADAPTER_LANGS=( "pt" "pt" "pt" )


TASK_ADAPTER_NAME="ner"

EN_WEIGHT="_uniform"
DATA_DIR=$DATA_DIR/${TASK}/${TASK}_processed_maxlen${MAX_LENGTH}/
OUTPUT_DIR="$OUT_DIR/${TASK}/my-${MODEL}-MaxLen${MAX_LENGTH}_${TASK_ADAPTER_NAME}_ensemble_en_top${K}_madx/"

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

echo $OUTPUT_DIR
for i in 0 1 2
do
SEED=${SEEDS[i]}
LANG_ADAPTER_NAME=${LANG_ADAPTER_NAMES[i]}
ADAPTER_LANG=${ADAPTER_LANGS[i]}
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