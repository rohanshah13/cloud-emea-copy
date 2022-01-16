# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, 
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from typing import Optional
from dataclasses import dataclass, field
from scipy.spatial import distance


from transformers import (
  WEIGHTS_NAME,
  AdamW,
  AdapterConfig,
  AdapterType,
  MultiLingAdapterArguments,
  HfArgumentParser,
  AlbertConfig,
  AlbertForQuestionAnswering,
  AlbertTokenizer,
  BertConfig,
  BertForQuestionAnswering,
  BertTokenizer,
  DistilBertConfig,
  DistilBertForQuestionAnswering,
  DistilBertTokenizer,
  XLMConfig,
  XLMForQuestionAnswering,
  XLMTokenizer,
  XLNetConfig,
  XLNetForQuestionAnswering,
  XLNetTokenizer,
  get_linear_schedule_with_warmup,
  XLMRobertaTokenizer,
  XLMRobertaForQuestionAnswering,
  XLMRobertaConfig
)

from transformers.data.metrics.squad_metrics import (
  compute_predictions_log_probs,
  compute_predictions_logits,
  squad_evaluate,
)

# from xlm_roberta import XLMRobertaForQuestionAnswering, XLMRobertaConfig

from processors.squad import (
  SquadResult,
  SquadV1Processor,
  SquadV2Processor,
  squad_convert_examples_to_features
)

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter

DEFAULT_LANGUAGES = {
  'mr': 'hi',
  'bn': 'hi',
  'ta': 'ta',
  'fo': 'fo',
  'no': 'da',
  'da': 'da',
  'be': 'be',
  'uk': 'uk',
  'bg': 'bg'
}
logger = logging.getLogger(__name__)

# ALL_MODELS = sum(
  # (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)),
  # (),
# )

MODEL_CLASSES = {
  "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
  "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
  "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
  "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
  "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
  "xlm-roberta": (XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer)
}


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
  return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer, lang2id=None):
  """ Train the model """
  if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter()

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
  else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
  )

  # Check if saved optimizer or scheduler states exist
  if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    os.path.join(args.model_name_or_path, "scheduler.pt")
  ):
    # Load in optimizer and scheduler states
    optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

  # multi-gpu training (should be after apex fp16 initialization)
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  # Distributed training (should be after apex fp16 initialization)
  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
      model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )

  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  logger.info(
    "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    args.train_batch_size
    * args.gradient_accumulation_steps
    * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
  )
  logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  logger.info("  Total optimization steps = %d", t_total)

  global_step = 1
  epochs_trained = 0
  steps_trained_in_current_epoch = 0
  # Check if continuing training from a checkpoint
  if os.path.exists(args.model_name_or_path):
    try:
      # set global_step to gobal_step of last saved checkpoint from model path
      checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
      global_step = int(checkpoint_suffix)
      epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
      steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

      logger.info("  Continuing training from checkpoint, will skip to saved global_step")
      logger.info("  Continuing training from epoch %d", epochs_trained)
      logger.info("  Continuing training from global step %d", global_step)
      logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    except ValueError:
      logger.info("  Starting fine-tuning.")

  tr_loss, logging_loss = 0.0, 0.0
  best_score = 0.0
  model.zero_grad()
  train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
  )
  # Added here for reproductibility
  set_seed(args)

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):

      # Skip past any already trained steps if resuming training
      if steps_trained_in_current_epoch > 0:
        steps_trained_in_current_epoch -= 1
        continue

      model.train()
      batch = tuple(t.to(args.device) for t in batch)

      inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": None if args.model_type in ["xlm", "xlm-roberta", "distilbert"] else batch[2],
        "start_positions": batch[3],
        "end_positions": batch[4],
      }

      if args.model_type in ["xlnet", "xlm"]:
        inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
        if args.version_2_with_negative:
          inputs.update({"is_impossible": batch[7]})
      if args.model_type == "xlm":
        inputs["langs"] = batch[7]
      outputs = model(**inputs)
      # model outputs are always tuple in transformers (see doc)
      loss = outputs[0]

      if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

      if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()

      tr_loss += loss.item()
      if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1

        # Log metrics
        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          # Only evaluate when single GPU otherwise metrics may not average well
          if args.local_rank == -1 and args.evaluate_during_training:
            results = evaluate(args, model, tokenizer, language=args.train_lang, lang2id=lang2id)
            for key, value in results.items():
              tb_writer.add_scalar("eval_{}".format(key), value, global_step)
          tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
          tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
          logging_loss = tr_loss

        # Save model checkpoint
        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
          if args.save_only_best_checkpoint:
            result = evaluate(args, model, tokenizer, prefix=global_step, language=args.train_lang, lang2id=lang2id)
            if result["f1"] > best_score:
              logger.info("result['f1']={} > best_score={}".format(result["f1"], best_score))
              best_score = result["f1"]
              # Save the best model checkpoint
              output_dir = os.path.join(args.output_dir, "checkpoint-best")
              best_checkpoint = output_dir
              if not os.path.exists(output_dir):
                os.makedirs(output_dir)
              # Take care of distributed/parallel training
              model_to_save = model.module if hasattr(model, "module") else model
              if args.do_save_adapters:
                model_to_save.save_all_adapters(output_dir)
              if args.do_save_adapter_fusions:
                model_to_save.save_all_adapter_fusions(output_dir)
              if args.do_save_full_model:
                model_to_save.save_pretrained(output_dir)
              torch.save(args, os.path.join(output_dir, "training_args.bin"))
              logger.info("Saving the best model checkpoint to %s", output_dir)
              
              torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
              torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
              logger.info("Saving optimizer and scheduler states to %s", output_dir)
              
              logger.info("Reset patience to 0")
              patience = 0
            else:
              patience += 1
              logger.info("Hit patience={}".format(patience))
              if args.eval_patience > 0 and patience > args.eval_patience:
                logger.info("early stop! patience={}".format(patience))
                epoch_iterator.close()
                train_iterator.close()
                if args.local_rank in [-1, 0]:
                  tb_writer.close()
                return global_step, tr_loss / global_step
          else:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
              os.makedirs(output_dir)
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            if args.do_save_adapters:
              model_to_save.save_all_adapters(output_dir)
            if args.do_save_adapter_fusions:
              model_to_save.save_all_adapter_fusions(output_dir)
            if args.do_save_full_model:
              model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
            
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

      if args.max_steps > 0 and global_step > args.max_steps:
        epoch_iterator.close()
        break
    if args.max_steps > 0 and global_step > args.max_steps:
      train_iterator.close()
      break

  if args.local_rank in [-1, 0]:
    tb_writer.close()

  return global_step, tr_loss / global_step

def jaccard_sim(vec1, vec2):
    intersection = 0
    union = 0
    for i in range(len(vec1)):
        if vec1[i] == '--' or vec2[i] == '--':
            continue
        if vec1[i] == 1 or vec2[i] == 1:
            union += 1
        if vec1[i] == 1 and vec2[i] == 1:
            intersection += 1
    return intersection/union

def get_sim(lang1, lang2):
  features = l2v.get_features(f'{DEFAULT_LANGUAGES[lang1]} {lang2}', 'learned')
  similarity = 1 - distance.cosine(features[DEFAULT_LANGUAGES[lang1]], features[lang2])
  return similarity

def get_syntax_sim(lang1, lang2):
  features = l2v.get_features(f'{lang1} {lang2}', "syntax_wals|syntax_sswl|syntax_ethnologue")
  similarity = jaccard_sim(features[lang1], features[lang2])
  return similarity

def calc_l2v_weights(args, lang, lang_adapter_names):
  adapter_weight = []
  for adapter_lang in lang_adapter_names:
    if args.en_weight is not None and adapter_lang == 'en':
      continue
    if args.lang_to_vec == 'learned':
      adapter_weight.append(get_sim(lang, adapter_lang))
    elif args.lang_to_vec == 'syntax':
      adapter_weight.append(get_syntax_sim(lang, adapter_lang))
    else:
      logger.info('INVALID FEATURE TYPE')
      exit()
  logger.info(adapter_weight)
  adapter_weight = torch.FloatTensor(adapter_weight)
  adapter_weight = torch.nn.functional.softmax(adapter_weight/args.temperature).tolist()
  if args.en_weight is not None:
    adapter_weight = [(1 - args.en_weight)*aw for aw in adapter_weight]
    en_index = lang_adapter_names.index('en')
    adapter_weight.insert(en_index, args.en_weight)
  return adapter_weight

def calc_weight_multi(args, model, batch, lang_adapter_names, task_name, adapter_weights, step=10, lang=None):
  inputs = {"input_ids": batch[0],
        "attention_mask": batch[1],
        "return_sequence_out": True,
        "labels": batch[3]}
  # logger.info(f'Language Adapters are {lang_adapter_names}')
  adapter_weights = [torch.FloatTensor([0.5 for _ in range(len(lang_adapter_names))]).to(args.device) for _ in range(13)]
  if args.lang_to_vec:
    logger.info(lang)
    logger.info(lang_adapter_names)
    adapter_weights = calc_l2v_weights(lang, lang_adapter_names, args.en_weight)
    logger.info(adapter_weights)
  for step_no in range(step):
    for w in adapter_weights: w.requires_grad = True
    if args.lang_to_vec and step_no == 0:
      normed_adapter_weights = adapter_weights
    else:
      normed_adapter_weights = [torch.nn.functional.softmax(w) for w in adapter_weights]
    # logger.info(f'Initial Adapter Weights = {normed_adapter_weights}')
    model.set_active_adapters([lang_adapter_names, [task_name]])
    inputs["adapter_names"] = [lang_adapter_names, [task_name]]

    inputs["adapter_weights"] = normed_adapter_weights
    outputs = model(**inputs)

    loss, logits, orig_sequence_output = outputs[:3]
    kept_logits = outputs[-1]
    entropy = torch.nn.functional.softmax(kept_logits, dim=1)*torch.nn.functional.log_softmax(kept_logits, dim=1)
    entropy = -entropy.sum() / kept_logits.size(0)
    grads = torch.autograd.grad(entropy, adapter_weights)
    #print(adapter_weights)
    #print(grads)
    #print(grads)
    for i, w in enumerate(adapter_weights):
      adapter_weights[i] = adapter_weights[i].data - 10*grads[i].data

def evaluate(args, model, tokenizer, prefix="", language='en', lang2id=None, adapter_weight=None, mode='train', lang_adapter_names=None, task_name=None, calc_weight_step=0):
  dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True,
                              language=language, lang2id=lang2id)

  if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    os.makedirs(args.output_dir)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

  # Note that DistributedSampler samples randomly
  eval_sampler = SequentialSampler(dataset)
  eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

  # multi-gpu evaluate
  if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

  # Eval!
  logger.info("***** Running evaluation {} *****".format(prefix))
  logger.info("  Num examples = %d", len(dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)

  all_results = []
  start_time = timeit.default_timer()

  counter = 0
  for batch in tqdm(eval_dataloader, desc="Evaluating"):
    counter += 1
    logger.info(f'Batch Number = {counter}')
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)

    if calc_weight_step > 0:
      adapter_weight = calc_weight_multi(args, model, batch, lang_adapter_names, task_name, adapter_weight, calc_weight_step, lang=language)
    with torch.no_grad():
      inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": None if args.model_type in ["xlm", "distilbert", "xlm-roberta"] else batch[2],
        "adapter_weights": adapter_weight,
      }
      example_indices = batch[3]

      # XLNet and XLM use more arguments for their predictions
      if args.model_type in ["xlnet", "xlm"]:
        inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
      if args.model_type == "xlm":
        inputs["langs"] = batch[6]
      outputs = model(**inputs)

    for i, example_index in enumerate(example_indices):
      eval_feature = features[example_index.item()]
      unique_id = int(eval_feature.unique_id)

      output = [to_list(output[i]) for output in outputs]

      # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
      # models only use two.
      if len(output) >= 5:
        start_logits = output[0]
        start_top_index = output[1]
        end_logits = output[2]
        end_top_index = output[3]
        cls_logits = output[4]

        result = SquadResult(
          unique_id,
          start_logits,
          end_logits,
          start_top_index=start_top_index,
          end_top_index=end_top_index,
          cls_logits=cls_logits,
        )

      else:
        start_logits, end_logits = output
        result = SquadResult(unique_id, start_logits, end_logits)

      all_results.append(result)

  evalTime = timeit.default_timer() - start_time
  logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

  # Compute predictions
  output_prediction_file = os.path.join(args.output_dir, "predictions_{}_{}.json".format(language, prefix))
  output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}_{}.json".format(language, prefix))

  if args.version_2_with_negative:
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
  else:
    output_null_log_odds_file = None

  # XLNet and XLM use a more complex post-processing procedure
  if args.model_type in ["xlnet", "xlm"]:
    start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
    end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

    predictions = compute_predictions_log_probs(
      examples,
      features,
      all_results,
      args.n_best_size,
      args.max_answer_length,
      output_prediction_file,
      output_nbest_file,
      output_null_log_odds_file,
      start_n_top,
      end_n_top,
      args.version_2_with_negative,
      tokenizer,
      args.verbose_logging,
    )
  else:
    predictions = compute_predictions_logits(
      examples,
      features,
      all_results,
      args.n_best_size,
      args.max_answer_length,
      args.do_lower_case,
      output_prediction_file,
      output_nbest_file,
      output_null_log_odds_file,
      args.verbose_logging,
      args.version_2_with_negative,
      args.null_score_diff_threshold,
      tokenizer,
    )

  # Compute the F1 and exact scores.
  results = squad_evaluate(examples, predictions)
  logger.info(f'Results = {results}')
  return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False,
              language='en', lang2id=None):
  if args.do_predict:
    args.predict_file = f'xquad.{language}.json'
  logger.info(f'Predict File = {args.predict_file}')
  if args.local_rank not in [-1, 0] and not evaluate:
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    torch.distributed.barrier()

  # Load data features from cache or dataset file
  input_dir = args.data_dir if args.data_dir else "."
  cached_features_file = os.path.join(
    input_dir,
    "cached_{}_{}_{}_{}".format(
      os.path.basename(args.predict_file) if evaluate else os.path.basename(args.train_file),
      list(filter(None, args.model_name_or_path.split("/"))).pop(),
      str(args.max_seq_length),
      str(language)
    ),
  )

  # Init features and dataset from cache if it exists
  if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
    logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
  else:
    logger.info("Creating features from dataset file at %s", input_dir)

    if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
      try:
        import tensorflow_datasets as tfds
      except ImportError:
        raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

      if args.version_2_with_negative:
        logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

      tfds_examples = tfds.load("squad")
      examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate, language=language)
    else:
      processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
      if evaluate:
        examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file, language=language)
      else:
        examples = processor.get_train_examples(args.data_dir, filename=args.train_file, language=language)

    features, dataset = squad_convert_examples_to_features(
      examples=examples,
      tokenizer=tokenizer,
      max_seq_length=args.max_seq_length,
      doc_stride=args.doc_stride,
      max_query_length=args.max_query_length,
      is_training=not evaluate,
      return_dataset="pt",
      threads=args.threads,
      lang2id=lang2id
    )
    logger.info(f'Local Rank = {args.local_rank}')
    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file %s", cached_features_file)
      torch.save({"features": features, "dataset": dataset}, cached_features_file)

  if args.local_rank == 0 and not evaluate:
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    torch.distributed.barrier()

  if output_examples:
    return dataset, examples, features
  return dataset

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    max_seq_length: Optional[int] = field(
        default=384, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    #Added these
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets."})
    predict_file: Optional[str] = field(default=None, metadata={"help": "The input evaluation file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets."})
    eval_test_set: Optional[bool] = field(default=False)

    do_train: Optional[bool] = field(default=False )
    do_eval: Optional[bool] = field(default=False )
    do_predict: Optional[bool] = field(default=False )
    do_adapter_predict: Optional[bool] = field(default=False )
    do_predict_dev: Optional[bool] = field(default=False )
    do_predict_train: Optional[bool] = field(default=False )
    init_checkpoint: Optional[str] = field(default=None )
    evaluate_during_training: Optional[bool] = field(default=False )
    do_lower_case: Optional[bool] = field(default=False )
    few_shot: Optional[int] = field(default=-1 )
    per_gpu_train_batch_size: Optional[int] = field(default=8)
    per_gpu_eval_batch_size: Optional[int] = field(default=8)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=5e-5)
    weight_decay: Optional[float] = field(default=0.0)
    adam_epsilon: Optional[float] = field(default=1e-8)
    max_grad_norm: Optional[float] = field(default=1.0)
    num_train_epochs: Optional[float] = field(default=3.0)
    max_steps: Optional[int] = field(default=-1)
    save_steps: Optional[int] = field(default=50)
    warmup_steps: Optional[int] = field(default=0)
    logging_steps: Optional[int] = field(default=50)
    save_only_best_checkpoint: Optional[bool] = field(default=False)
    eval_all_checkpoints: Optional[bool] = field(default=False)
    no_cuda: Optional[bool] = field(default=False)
    overwrite_output_dir: Optional[bool] = field(default=False)
    overwrite_cache: Optional[bool] = field(default=False)
    seed: Optional[int] = field(default=42)
    fp16: Optional[bool] = field(default=False)
    fp16_opt_level: Optional[str] = field(default="O1")
    local_rank: Optional[int] = field(default=-1)
    server_ip: Optional[str] = field(default="")
    server_port: Optional[str] = field(default="")
    eval_lang: Optional[str] = field(default="en", metadata={"help": "The language of the dev data"}) #!!!
    predict_langs: Optional[str] = field(default="en", metadata={"help": "The language of the test data"})
    train_lang: Optional[str] = field(default="en", metadata={"help": "The language of the training data"})
    log_file: Optional[str] = field(default=None)
    eval_patience: Optional[int] = field(default=-1)
    bpe_dropout: Optional[float] = field(default=0)
    do_save_adapter_fusions: Optional[bool] = field(default=False)
    task_name: Optional[str] = field(default="xnli")
    
    threads: Optional[int] = field(default=1, metadata={"help": "multiple threads for converting example to features"})
    version_2_with_negative: Optional[bool] = field(default=False, metadata={"help": "If true, the SQuAD examples contain some that do not have an answer."})
    verbose_logging: Optional[bool] = field(default=False, metadata={"help": "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation."})
    n_best_size: Optional[int] = field(default=20, metadata={"help": "The total number of n-best predictions to generate in the nbest_predictions.json output file."})
    max_query_length: Optional[int] = field(default=64, metadata={"help": "The maximum number of tokens for the question. Questions longer than this will "
    "be truncated to this length."})
    max_answer_length: Optional[int] = field(default=30, metadata={"help": "The maximum length of an answer that can be generated. This is needed because the start "
    "and end predictions are not conditioned on one another."})
    doc_stride: Optional[int] = field(default=128, metadata={"help":"When splitting up a long document into chunks, how much stride to take between chunks."})
    null_score_diff_threshold: Optional[float] = field(default=0.0, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."})
    
    predict_task_adapter: Optional[str] = field(default=None)
    predict_lang_adapter: Optional[str] = field(default=None)
    test_adapter: Optional[bool] = field(default=False)

    adapter_weight: Optional[str] = field(default=None)
    lang_to_vec: Optional[str] = field(default=None)

    calc_weight_step: Optional[int] = field(default=0)
    predict_save_prefix: Optional[str] = field(default=None)
    en_weight: Optional[float] = field(default=None)
    temperature: Optional[float] = field(default=1.0)

    get_attr: Optional[bool] = field(default=False)
    topk: Optional[int] = field(default=1)

    task: Optional[str] = field(default='udpos')

def load_model(args):
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  if args.model_type == "xlm":
    config.use_lang_emb = True
  tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  model = model_class.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )

  lang2id = config.lang2id if args.model_type == "xlm" else None
  logger.info("lang2id = {}".format(lang2id))
  return model, tokenizer, lang2id, config_class, model_class, tokenizer

def setup_adapter(args, adapter_args, model, train_adapter=True, load_adapter=None, load_lang_adapter=None):
  task_name = args.task_name or "squad"
  # check if adapter already exists, otherwise add it
  if task_name not in model.config.adapters.adapter_list(AdapterType.text_task):
      logging.info("Trying to decide if add adapter")
      # resolve the adapter config
      adapter_config = AdapterConfig.load(
          adapter_args.adapter_config,
          non_linearity=adapter_args.adapter_non_linearity,
          reduction_factor=adapter_args.adapter_reduction_factor,
      )
      # load a pre-trained from Hub if specified
      if adapter_args.load_adapter or load_adapter:
          logging.info("loading task adapter")
          model.load_adapter(
              adapter_args.load_adapter if load_adapter is None else load_adapter,
              AdapterType.text_task,
              config=adapter_config,
              load_as=task_name,
          )
      # otherwise, add a fresh adapter
      else:
          logging.info("Adding task adapter")
          model.add_adapter(task_name, AdapterType.text_task, config=adapter_config)
  # optionally load a pre-trained language adapter
  if adapter_args.load_lang_adapter or load_lang_adapter:
      if load_lang_adapter is None:
          # load a set of language adapters
          logging.info("loading lang adpater {}".format(adapter_args.load_lang_adapter))
          # resolve the language adapter config
          lang_adapter_config = AdapterConfig.load(
              adapter_args.lang_adapter_config,
              non_linearity=adapter_args.lang_adapter_non_linearity,
              reduction_factor=adapter_args.lang_adapter_reduction_factor,
          )
          # load the language adapter from Hub
          languages = adapter_args.language.split(",")
          adapter_names = adapter_args.load_lang_adapter.split(",")
          assert len(languages) == len(adapter_names)
          lang_adapter_names = []
          for language, adapter_name in zip(languages, adapter_names):
              print(language, adapter_name)
              lang_adapter_name = model.load_adapter(
                  adapter_name,
                  AdapterType.text_lang,
                  config=lang_adapter_config,
                  load_as=language,
              )
              lang_adapter_names.append(lang_adapter_name)
      else:
          logging.info("loading lang adpater {}".format(load_lang_adapter))
          # resolve the language adapter config
          lang_adapter_config = AdapterConfig.load(
              adapter_args.lang_adapter_config,
              non_linearity=adapter_args.lang_adapter_non_linearity,
              reduction_factor=adapter_args.lang_adapter_reduction_factor,
          )
          # load the language adapter from Hub
          lang_adapter_name = model.load_adapter(
              load_lang_adapter,
              AdapterType.text_lang,
              config=lang_adapter_config,
              load_as="lang",
          )
          lang_adapter_names = [lang_adapter_name]
  else:
      lang_adapter_name = None
  # Freeze all model weights except of those of this adapter
  model.train_adapter([task_name])

  # Set the adapters to be used in every forward pass
  if lang_adapter_name:
      model.set_active_adapters([lang_adapter_names, [task_name]])
  else:
      model.set_active_adapters([task_name])

  return model, lang_adapter_names, task_name

def predict_and_save(args, adapter_args, model, tokenizer, lang2id, lang_adapter_names, task_name, split):
  output_test_results_file = os.path.join(args.output_dir, f'{split}_results.txt')
  with open(output_test_results_file, 'a') as result_writer:
    for lang in args.predict_langs.split(','):
      
      if not os.path.exists(os.path.join(args.data_dir, f'xquad.{lang}.json')):
        logger.info(f"Language {lang}, Split {split} does not exist")
        continue

      #Activate the required language adapter
      adapter_weight = None
      if not args.adapter_weight and not args.lang_to_vec:
        if (adapter_args.train_adapter or args.test_adapter) and not args.adapter_weight:
          if lang in lang_adapter_names:
            logger.info(f'Language adapter for {lang} found')
            logger.info("Set active language adapter to {}".format(lang))
            model.set_active_adapters([[lang], [task_name]])
          else:
            logger.info(f'Language adapter for {lang} not found, using {lang_adapter_names[0]} instead')
            logger.info("Set active language adapter to {}".format(lang_adapter_names[0]))
            model.set_active_adapters([[lang_adapter_names[0]], [task_name]])
      else:
        if args.adapter_weight == 'equal':
          adapter_weight = [1/len(lang_adapter_names) for _ in lang_adapter_names]
        elif args.adapter_weight == 'equal_en':
          assert 'en' in lang_adapter_names, 'English language adapter not included'
          adapter_weight = [(1-args.en_weight)/(len(lang_adapter_names)-1) for _ in lang_adapter_names]
          en_index = lang_adapter_names.index('en')
          adapter_weight[en_index] = args.en_weight
        elif args.lang_to_vec:
          if args.en_weight is not None:
            logger.info(lang_adapter_names)
            assert 'en' in lang_adapter_names, 'English language adapter not included'
          adapter_weight = calc_l2v_weights(args, lang, lang_adapter_names)
        elif args.adapter_weight == 'load':
          filename = f'weights/{args.task}/{lang}/weights_s{args.seed}'
          logger.info(f'Loading adapter weights from {filename}')
          with open(filename) as f:
            adapter_weight = json.loads(next(f))
        elif args.adapter_weight != "0" and args.adapter_weight is not None:
          adapter_weight = [float(w) for w in args.adapter_weight.split(",")]
      logger.info('Args Adapter Weight = {}'.format(args.adapter_weight))
      logger.info('Adapter Languages = {}'.format(lang_adapter_names))
      if adapter_weight is not None:
        logger.info("Adapter Weights = {}".format(adapter_weight))
        logger.info('Sum of Adapter Weights = {}'.format(sum(adapter_weight)))
        logger.info("Length of Adapter Weights = {}".format(len(adapter_weight))) 
      model.set_active_adapters([ lang_adapter_names, [task_name]])

      #Evaluate
      results = evaluate(args, model, tokenizer, language=lang, lang2id=lang2id, adapter_weight=adapter_weight, mode=split)

      result_json = {}
      result_json['language'] = lang
      result_json['seed'] = args.seed
      result_json['language_adapters'] = lang_adapter_names
      if args.adapter_weight:
        result_json['adapter_weights'] = adapter_weight

      for key in sorted(results.keys()):
        result_json[key] = results[key]

      result_writer.write(json.dumps(result_json) + '\n')


def main():

  parser = HfArgumentParser((ModelArguments, MultiLingAdapterArguments))
  args, adapter_args = parser.parse_args_into_dataclasses()

  if (
    os.path.exists(args.output_dir)
    and os.listdir(args.output_dir)
    and args.do_train
    and not args.overwrite_output_dir
  ):
    raise ValueError(
      "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
        args.output_dir
      )
    )

  # Setup distant debugging if needed
  if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd

    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
  else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device

  # Setup logging
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
  )
  logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    args.local_rank,
    device,
    args.n_gpu,
    bool(args.local_rank != -1),
    args.fp16,
  )

  # Set seed
  set_seed(args)

  # Load pretrained model and tokenizer
  if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

  args.do_save_full_model= (not adapter_args.train_adapter)
  args.do_save_adapters=adapter_args.train_adapter
  if args.do_save_adapters:
      logging.info('save adapters')
      logging.info(adapter_args.train_adapter)
  if args.do_save_full_model:
      logging.info('save model')
      
  args.model_type = args.model_type.lower()
  model, tokenizer, lang2id, config_class, model_class, tokenizer_class = load_model(args)


  if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

  model.to(args.device)

  logger.info("Training/evaluation parameters %s", args)

  # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
  # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
  # remove the need for this code, but it is still valid.
  if args.fp16:
    try:
      import apex

      apex.amp.register_half_function(torch, "einsum")
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

  # Training
  if args.do_train:
    model, tokenizer, lang2id, config_class, model_class, tokenizer_class = load_model(args)
    if adapter_args.train_adapter:
      model,lang_adapter_names, task_name = setup_adapter(args, adapter_args, model)
      logger.info("lang adapter names: {}".format(" ".join(lang_adapter_names)))
    else:
      lang_adatper_names = []
      task_name = None
    for name, param in model.named_parameters():
      logger.info(name)
      logger.info(param.requires_grad)
    model.to(args.device)
    
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, language=args.train_lang, lang2id=lang2id)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer, lang2id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

  # Save the trained model and the tokenizer
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    if args.do_save_adapters:
      logging.info("Save adapter")
      model_to_save.save_all_adapters(args.output_dir)
    if args.do_save_adapter_fusions:
      logging.info("Save adapter fusions")
      model_to_save.save_all_adapter_fusions(args.output_dir)
    if args.do_save_full_model:
      logging.info("Save full model")
      model_to_save.save_pretrained(args.output_dir)
    
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(args.output_dir, force_download=True)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)

  # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
  results = {}
  if args.do_eval and args.local_rank in [-1, 0]:
    if args.do_train:
      logger.info("Loading checkpoints saved during training for evaluation")
      checkpoints = [args.output_dir]
      if args.eval_all_checkpoints:
        checkpoints = list(
          os.path.dirname(c)
          for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
    else:
      logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
      checkpoints = [args.model_name_or_path]

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
      # Reload the model
      global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
      model = model_class.from_pretrained(checkpoint, force_download=True)
      model.to(args.device)

      # Evaluate
      result = evaluate(args, model, tokenizer, prefix=global_step, language=args.eval_lang, lang2id=lang2id)

      result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
      results.update(result)

  logger.info("Results: {}".format(results))

  if args.do_predict:
    model, tokenizer, lang2id, config_class, model_class, tokenizer_class = load_model(args)

    logger.info('Evaluating the model on the test set of all languages specified')

    if adapter_args.train_adapter or args.test_adapter:
      load_adapter = args.predict_task_adapter

      logger.info(f'Adapter will be loaded from this path: {load_adapter}')

      load_lang_adapter = args.predict_lang_adapter
      model.model_name = args.model_name_or_path
      model, lang_adapter_names, task_name = setup_adapter(args, adapter_args, model, load_adapter=load_adapter, load_lang_adapter=load_lang_adapter)
    model.to(args.device)

    predict_and_save(args, adapter_args, model, tokenizer, lang2id, lang_adapter_names, task_name, 'test')


  return results


if __name__ == "__main__":
  main()
