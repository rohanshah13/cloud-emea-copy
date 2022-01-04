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
""" Finetuning multi-lingual models on XNLI/PAWSX (Bert, XLM, XLMRoberta)."""


import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
import json
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
  WEIGHTS_NAME,
  AdamW,
  BertConfig,
  BertForSequenceClassification,
  BertTokenizer,
  XLMConfig,
  XLMForSequenceClassification,
  XLMTokenizer,
  XLMRobertaConfig,
  XLMRobertaTokenizer,
  XLMRobertaForSequenceClassification,
  get_linear_schedule_with_warmup,
)

from transformers import (
  AdamW,
  get_linear_schedule_with_warmup,
  WEIGHTS_NAME,
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
  HfArgumentParser,
  MultiLingAdapterArguments,
  AdapterConfig,
  AdapterType,
)
from processors.utils import convert_examples_to_features
from processors.xnli import XnliProcessor
from processors.pawsx import PawsxProcessor

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

# ALL_MODELS = sum(
  # (tuple(conf.pretrained_config_archive_map.keys()) 
    # for conf in (BertConfig, XLMConfig, XLMRobertaConfig)),
  # ()
# )

MODEL_CLASSES = {
  "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
  "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
  "xlmr": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

PROCESSORS = {
  'xnli': XnliProcessor,
  'pawsx': PawsxProcessor,
}


def compute_metrics(preds, labels):
  scores = {
    "acc": (preds == labels).mean(), 
    "num": len(
      preds), 
    "correct": (preds == labels).sum()
  }
  return scores


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, lang2id=None):
  """Train the model."""
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

  global_step = 0
  epochs_trained = 0
  steps_trained_in_current_epoch = 0
  # Check if continuing training from a checkpoint
  if os.path.exists(args.model_name_or_path):
    # set global_step to gobal_step of last saved checkpoint from model path
    global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    logger.info("  Continuing training from epoch %d", epochs_trained)
    logger.info("  Continuing training from global step %d", global_step)
    logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

  best_score = 0
  best_checkpoint = None
  tr_loss, logging_loss = 0.0, 0.0
  model.zero_grad()
  train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
  )
  set_seed(args)  # Added here for reproductibility
  for i,_ in enumerate(train_iterator):
    logger.info(f'Epoch Number = {i}')
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
      logger.info(f'Step Number = {step}')
      # Skip past any already trained steps if resuming training
      if steps_trained_in_current_epoch > 0:
        steps_trained_in_current_epoch -= 1
        continue

      model.train()
      batch = tuple(t.to(args.device) for t in batch)
      inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
      if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
          batch[2] if args.model_type in ["bert"] else None
        )  # XLM don't use segment_ids
      if args.model_type == "xlm":
        inputs["langs"] = batch[4]
      outputs = model(**inputs)
      loss = outputs[0]

      if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
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

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          # Log metrics
          tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
          tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
          logging_loss = tr_loss

          # Only evaluate on single GPU otherwise metrics may not average well
          if (args.local_rank == -1 and args.evaluate_during_training):  
            results = evaluate(args, model, tokenizer, split=args.train_split, language=args.train_language, lang2id=lang2id)
            for key, value in results.items():
              tb_writer.add_scalar("eval_{}".format(key), value, global_step)

        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
          if args.eval_test_set:
            output_predict_file = os.path.join(args.output_dir, 'eval_test_results')
            total = total_correct = 0.0
            with open(output_predict_file, 'a') as writer:
              writer.write('\n======= Predict using the model from checkpoint-{}:\n'.format(global_step))
              for language in args.predict_languages.split(','):
                result = evaluate(args, model, tokenizer, split=args.test_split, language=language, lang2id=lang2id, prefix='checkpoint-'+str(global_step))
                writer.write('{}={}\n'.format(language, result['acc']))
                total += result['num']
                total_correct += result['correct']
              writer.write('total={}\n'.format(total_correct / total))

          if args.save_only_best_checkpoint:          
            result = evaluate(args, model, tokenizer, split='dev', language=args.train_language, lang2id=lang2id, prefix=str(global_step))
            logger.info(" Dev accuracy {} = {}".format(args.train_language, result['acc']))
            if result['acc'] > best_score:
              logger.info(" result['acc']={} > best_score={}".format(result['acc'], best_score))
              output_dir = os.path.join(args.output_dir, "checkpoint-best")
              best_checkpoint = output_dir
              best_score = result['acc']
              # Save model checkpoint
              if not os.path.exists(output_dir):
                os.makedirs(output_dir)
              model_to_save = (
                model.module if hasattr(model, "module") else model
              )  # Take care of distributed/parallel training
              if args.do_save_adapters:
                model_to_save.save_all_adapters(output_dir)
              if args.do_save_adapter_fusions:
                model_to_save.save_all_adapter_fusions(output_dir)
              if args.do_save_full_model:
                model_to_save.save_pretrained(output_dir)
              tokenizer.save_pretrained(output_dir)

              torch.save(args, os.path.join(output_dir, "training_args.bin"))
              logger.info("Saving model checkpoint to %s", output_dir)

              torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
              torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
              logger.info("Saving optimizer and scheduler states to %s", output_dir)
          else:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
              os.makedirs(output_dir)
            model_to_save = (
              model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            if args.do_save_adapters:
              model_to_save.save_all_adapters(output_dir)
            if args.do_save_adapter_fusions:
              model_to_save.save_all_adapter_fusions(output_dir)
            if args.do_save_full_model:
              model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

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

  return global_step, tr_loss / global_step, best_score, best_checkpoint

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


  normed_adapter_weights = [torch.nn.functional.softmax(w) for w in adapter_weights]
  #print(normed_adapter_weights)
  # logger.info(f'Final Adapter Weights = {normed_adapter_weights}')
  return normed_adapter_weights

def evaluate(args, model, tokenizer, split='train', language='en', lang2id=None, prefix="", output_file=None, label_list=None, output_only_prediction=True, adapter_weight=None, lang_adapter_names=None, task_name=None, calc_weight_step=0):
  """Evalute the model."""
  eval_task_names = (args.task_name,)
  eval_outputs_dirs = (args.output_dir,)

  results = {}
  for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
    eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, split=split, language=language, lang2id=lang2id, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
      model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} {} *****".format(prefix, language))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    sentences = None
    counter = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
      model.eval()
      logger.info(f'Batch Number = {counter}')
      batch = tuple(t.to(args.device) for t in batch)
      counter += 1

      if calc_weight_step > 0:
        adapter_weight = calc_weight_multi(args, model, batch, lang_adapter_names,)
      with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "adapter_weights": adapter_weight}
        if args.model_type != "distilbert":
          inputs["token_type_ids"] = (
            batch[2] if args.model_type in ["bert"] else None
          )  # XLM and DistilBERT don't use segment_ids
        if args.model_type == "xlm":
          inputs["langs"] = batch[4]
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]

        eval_loss += tmp_eval_loss.mean().item()
      nb_eval_steps += 1
      if preds is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        sentences = inputs["input_ids"].detach().cpu().numpy()
      else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        sentences = np.append(sentences, inputs["input_ids"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
      preds = np.argmax(preds, axis=1)
    else:
      raise ValueError("No other `output_mode` for XNLI.")
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    if output_file:
      logger.info("***** Save prediction ******")
      with open(output_file, 'w') as fout:
        pad_token_id = tokenizer.pad_token_id
        sentences = sentences.astype(int).tolist()
        sentences = [[w for w in s if w != pad_token_id]for s in sentences]
        sentences = [tokenizer.convert_ids_to_tokens(s) for s in sentences]
        #fout.write('Prediction\tLabel\tSentences\n')
        for p, l, s in zip(list(preds), list(out_label_ids), sentences):
          s = ' '.join(s)
          if label_list:
            p = label_list[p]
            l = label_list[l]
          if output_only_prediction:
            fout.write(str(p) + '\n')
          else:
            fout.write('{}\t{}\t{}\n'.format(p, l, s))
    logger.info("***** Eval results {} {} *****".format(prefix, language))
    for key in sorted(result.keys()):
      logger.info("  %s = %s", key, str(result[key]))

  return results


def load_and_cache_examples(args, task, tokenizer, split='train', language='en', lang2id=None, evaluate=False):
  # Make sure only the first process in distributed training process the 
  # dataset, and the others will use the cache
  if args.local_rank not in [-1, 0] and not evaluate:
    torch.distributed.barrier()

  processor = PROCESSORS[task]()
  output_mode = "classification"
  # Load data features from cache or dataset file
  lc = '_lc' if args.do_lower_case else ''
  cached_features_file = os.path.join(
    args.data_dir,
    "cached_{}_{}_{}_{}_{}{}".format(
      split,
      list(filter(None, args.model_name_or_path.split("/"))).pop(),
      str(args.max_seq_length),
      str(task),
      str(language),
      lc,
    ),
  )
  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
  else:
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if split == 'train':
      examples = processor.get_train_examples(args.data_dir, language)
    elif split == 'translate-train':
      examples = processor.get_translate_train_examples(args.data_dir, language)
    elif split == 'translate-test':
      examples = processor.get_translate_test_examples(args.data_dir, language)
    elif split == 'dev':
      examples = processor.get_dev_examples(args.data_dir, language)
    elif split == 'pseudo_test':
      examples = processor.get_pseudo_test_examples(args.data_dir, language)
    else:
      examples = processor.get_test_examples(args.data_dir, language)

    features = convert_examples_to_features(
      examples,
      tokenizer,
      label_list=label_list,
      max_length=args.max_seq_length,
      output_mode=output_mode,
      pad_on_left=False,
      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
      pad_token_segment_id=0,
      lang2id=lang2id,
    )
    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file %s", cached_features_file)
      torch.save(features, cached_features_file)

  # Make sure only the first process in distributed training process the 
  # dataset, and the others will use the cache
  if args.local_rank == 0 and not evaluate:
    torch.distributed.barrier()  

  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
  all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
  if output_mode == "classification":
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
  else:
    raise ValueError("No other `output_mode` for {}.".format(args.task_name))

  if args.model_type == 'xlm':
    all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_langs)
  else:  
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
  return dataset

def setup_adapter(args, adapter_args, model, train_adapter=True, load_adapter=None, load_lang_adapter=None):
  task_name = args.task_name or "xnli"
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

def load_model(args, num_labels):
  logger.info('Loading pretrained model and tokenizer')
  args.model_type = args.model_type.lower()
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=args.task_name,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  logger.info("config = {}".format(config))

  tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )

  if args.init_checkpoint:
    logger.info("loading from folder {}".format(args.init_checkpoint))
    #changed cache_dir
    model = model_class.from_pretrained(
      args.init_checkpoint,
      config=config,
      cache_dir=args.cache_dir,
      )
  else:
    logger.info("loading from existing model {}".format(args.model_name_or_path))
    model = model_class.from_pretrained(
      args.model_name_or_path,
      from_tf=bool(".ckpt" in args.model_name_or_path),
      config=config,
      cache_dir=args.cache_dir if args.cache_dir else None,
    )
  lang2id = config.lang2id if args.model_type == "xlm" else None
  logger.info("lang2id = {}".format(lang2id))

  return model, tokenizer, lang2id, config_class, model_class, tokenizer_class

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
    labels: str = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    max_seq_length: Optional[int] = field(
        default=128, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    #Added these
    train_split: Optional[str] = field(default='train')
    test_split: Optional[str] = field(default='test')
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
    save_steps: Optional[int] = field(default=-1)
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
    predict_languages: Optional[str] = field(default="en")
    train_language: Optional[str] = field(default="en")
    log_file: Optional[str] = field(default=None)
    eval_patience: Optional[int] = field(default=-1)
    bpe_dropout: Optional[float] = field(default=0)
    do_save_adapter_fusions: Optional[bool] = field(default=False)
    task_name: Optional[str] = field(default="xnli")

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

def predict_and_save(args, adapter_args, model, tokenizer, lang2id, lang_adapter_names, task_name, split):
  output_test_results_file = os.path.join(args.output_dir, f"{split}_results.txt")
  with open(output_test_results_file, "a") as result_writer:
    for lang in args.predict_languages.split(','):
      #Check if language data exists
      # if not os.path.exists(os.path.join(args.data_dir, lang, '{}.{}'.format(split, args.model_name_or_path))):
        # logger.info("Language {}, split {} does not exist".format(lang, split))
        # continue

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
      result = evaluate(args, model, tokenizer, split=split, language=lang, lang2id=lang2id, adapter_weight=adapter_weight, lang_adapter_names=lang_adapter_names, task_name=task_name, calc_weight_step=args.calc_weight_step)
      
      if args.get_attr:
        continue
      result_json = {}
      # Save results
      if args.predict_save_prefix is not None and args.predict_save_prefix:
        result_json['language'] = f'{args.predict_save_prefix}_{lang}'
      else:
        result_json['language'] = f'{lang}'
      
      result_json['seed'] = str(args.seed)
      result_json['language_adapters'] = lang_adapter_names
      if args.adapter_weight:
        result_json['adapter_weights'] = str(args.adapter_weight)
      
      for key in sorted(result.keys()):
        result_json[key] = str(result[key])
      
      result_writer.write(json.dumps(result_json) + '\n')


def main():
  parser = argparse.ArgumentParser()

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
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
  else:  # Initializes the distributed backend which sychronizes nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device

  # Setup logging
  logging.basicConfig(handlers=[logging.FileHandler(os.path.join(args.output_dir, args.log_file)), logging.StreamHandler()],
                      format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt='%m/%d/%Y %H:%M:%S',
                      level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
  logging.info("Input args: %r" % args)
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

  # Prepare dataset
  if args.task_name not in PROCESSORS:
    raise ValueError("Task not found: %s" % (args.task_name))
  processor = PROCESSORS[args.task_name]()
  args.output_mode = "classification"
  label_list = processor.get_labels()
  num_labels = len(label_list)

  args.do_save_full_model= (not adapter_args.train_adapter)
  args.do_save_adapters=adapter_args.train_adapter
  if args.do_save_adapters:
      logging.info('save adapters')
      logging.info(adapter_args.train_adapter)
  if args.do_save_full_model:
      logging.info('save model')

  # Load pretrained model and tokenizer
  # Make sure only the first process in distributed training loads model & vocab
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  

  # Make sure only the first process in distributed training loads model & vocab
  if args.local_rank == 0:
    torch.distributed.barrier()

  logger.info("Training/evaluation parameters %s", args)  

  # Training
  if args.do_train:
    model, tokenizer, lang2id, config_class, model_class, tokenizer_class = load_model(args, num_labels)
    
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
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, split=args.train_split, language=args.train_language, lang2id=lang2id, evaluate=False)
    global_step, tr_loss, best_score, best_checkpoint = train(args, train_dataset, model, tokenizer, lang2id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    logger.info(" best checkpoint = {}, best score = {}".format(best_checkpoint, best_score))

  # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
      model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    if args.do_save_adapters:
      logging.info("Save adapter")
      model_to_save.save_all_adapters(args.output_dir)
    if args.do_save_adapter_fusions:
      logging.info("Save adapter fusion")
      model_to_save.save_all_adapter_fusions(args.output_dir)
    if args.do_save_full_model:
      logging.info("Save full model")
      model_to_save.save_pretrained(args.output_dir)
    
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(args.output_dir)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    model.to(args.device)

  # Evaluation
  results = {}
  if args.init_checkpoint:
    best_checkpoint = args.init_checkpoint
  elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
    best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
  else:
    best_checkpoint = args.output_dir
  best_score = 0
  if args.do_eval and args.local_rank in [-1, 0]:
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
      checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
      )
      logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
      global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
      prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

      model = model_class.from_pretrained(checkpoint)
      model.to(args.device)
      result = evaluate(args, model, tokenizer, split='dev', language=args.train_language, lang2id=lang2id, prefix=prefix)
      if result['acc'] > best_score:
        best_checkpoint = checkpoint
        best_score = result['acc']
      result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
      results.update(result)
    
    output_eval_file = os.path.join(args.output_dir, 'eval_results')
    with open(output_eval_file, 'w') as writer:
      for key, value in results.items():
        writer.write('{} = {}\n'.format(key, value))
      writer.write("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))
      logger.info("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))

  # Prediction
  if args.do_predict and args.local_rank in [-1, 0]:
    model, tokenizer, lang2id, config_class, model_class, tokenizer_class = load_model(args, num_labels)

    if adapter_args.train_adapter or args.test_adapter:
        load_adapter = (best_checkpoint + "/" + args.task_name) if args.predict_task_adapter is None else args.predict_task_adapter
        logger.info(f'Task Adapter will be loaded from this path {load_adapter}')
        load_lang_adapter = args.predict_lang_adapter
        model.model_name = args.model_name_or_path
        model, lang_adapter_names, task_name = setup_adapter(args, adapter_args, model, load_adapter=load_adapter, load_lang_adapter=load_lang_adapter)
    model.to(args.device)

    predict_and_save(args, adapter_args, model, tokenizer, lang2id, lang_adapter_names, task_name, 'test')
    # output_predict_file = os.path.join(args.output_dir, args.test_split + '_results.txt')
    # total = total_correct = 0.0
    # with open(output_predict_file, 'a') as writer:
      # writer.write('======= Predict using the model from {} for {}:\n'.format(best_checkpoint, args.test_split))
      # for language in args.predict_languages.split(','):
        # output_file = os.path.join(args.output_dir, 'test-{}.tsv'.format(language))
        # result = evaluate(args, model, tokenizer, split=args.test_split, language=language, lang2id=lang2id, prefix='best_checkpoint', output_file=output_file, label_list=label_list)
        # writer.write('{}={}\n'.format(language, result['acc']))
        # logger.info('{}={}'.format(language, result['acc']))
        # total += result['num']
        # total_correct += result['correct']
      # writer.write('total={}\n'.format(total_correct / total))

  if args.do_predict_dev:
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path if args.model_name_or_path else best_checkpoint, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.init_checkpoint)
    model.to(args.device)
    output_predict_file = os.path.join(args.output_dir, 'dev_results')
    total = total_correct = 0.0
    with open(output_predict_file, 'w') as writer:
      writer.write('======= Predict using the model from {}:\n'.format(args.init_checkpoint))
      for language in args.predict_languages.split(','):
        output_file = os.path.join(args.output_dir, 'dev-{}.tsv'.format(language))
        result = evaluate(args, model, tokenizer, split='dev', language=language, lang2id=lang2id, prefix='best_checkpoint', output_file=output_file, label_list=label_list)
        writer.write('{}={}\n'.format(language, result['acc']))
        total += result['num']
        total_correct += result['correct']
      writer.write('total={}\n'.format(total_correct / total))



if __name__ == "__main__":
  main()
