import json
from collections import defaultdict
import argparse
import numpy as np
import os

LANG2VEC_LANGS = "am,bn,cs,de,el,es,et,eu,fi,fr,hi,hu,hy,id,ja,jv,ko,la,lv,mi,my,pt,ru,tr,vi,zh"
LANG2VEC_LANGS_EN = "en,am,bn,cs,de,el,es,et,eu,fi,fr,hi,hu,hy,id,ja,jv,ko,la,lv,mi,my,pt,ru,tr,vi,zh"
ALL_LANGS= "am,ar,bxr,cdo,cs,de,el,en,es,et,eu,fa,fi,fr,gn,hi,ht,hu,hy,id,ilo,is,ja,jv,ka,ko,kv,la,lv,mhr,mi,my,myv,pt,qu,ru,se,sw,tk,tr,vi,wo,xmf,zh,zh_yue"
ALL_QA_LANGS = "ar,cdo,de,el,en,es,et,gn,hi,ht,id,ilo,is,it,ja,jv,mhr,mi,my,qu,ru,sw,ta,th,tk,tr,vi,xmf,zh"
URIEL_LANGS = "am,bn,cs,de,el,en,es,et,eu,fi,fr,hi,ht,hu,hy,id,ilo,is,ja,jv,ka,ko,la,lv,mhr,mi,my,myv,pt,ru,se,tk,tr,vi,wo"
EN_LANGS_POS = "en,ka,vi,pt,ar,hu,am,cs,eu,id"
EN_LANGS_NER = "en,pt,id,tr,cs,vi,eu,zh_yue,fa,es"
INFILE = 'outputs/{}/my-{}-MaxLen{}_{}_{}/{}_results.txt'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='en')
    parser.add_argument('--languages', default='en')
    parser.add_argument('--task', default='ner')
    parser.add_argument('--split', default='dev')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--all_lang', dest='all_lang', action='store_true')
    parser.add_argument('--lang2vec_lang', dest='lang2vec_lang', action='store_true')
    parser.add_argument('--lang2vec_lang_en', dest='lang2vec_lang_en', action='store_true')
    parser.add_argument('--uriel_lang', dest='uriel_lang', action='store_true')
    parser.add_argument('--en_lang', dest='en_lang', action='store_true')
    parser.add_argument('--en_weight', default=0.1)
    parser.add_argument('--topk', default=1)
    parser.add_argument('--seeds', default='1,2,3')
    parser.add_argument('--eval_lang', default=None)
    parser.add_argument('--model', default='bert-base-multilingual-cased')
    parser.set_defaults(all_lang=False, lang2vec_lang=False)
    args = parser.parse_args()
    if args.all_lang:
        if args.task == 'qna':
            args.languages = ALL_QA_LANGS
        else:
            args.languages = ALL_LANGS
    if args.lang2vec_lang:
        args.languages = LANG2VEC_LANGS
    if args.uriel_lang:
        args.languages = URIEL_LANGS
    if args.lang2vec_lang_en:
        args.languages = LANG2VEC_LANGS_EN
    if args.en_lang:
        if args.task == 'ner':
            args.languages = EN_LANGS_NER
        else:
            args.languages = EN_LANGS_POS

    if args.task == 'ner':
        dataset = 'panx'
    elif args.task == 'udpos':
        dataset = 'udpos'
    else:
        dataset = 'squad'
    
    if args.task == 'qna':
        maxlen = 384
    else:
        maxlen = 128

    if args.method == 'en':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, 'en', args.split)
    elif args.method == 'related':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, args.languages, args.split)
    elif args.method == 'topk':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'topk_{args.topk}', args.split)
    elif args.method == 'ensemble':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'{args.languages}_ensemble', args.split)
    elif args.method == 'ensemble_attribution':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'{args.languages}_ensemble_attribution', args.split)
    elif args.method == 'ensemble_en':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'{args.languages}_ensemble_awequal_en_en{args.en_weight}', args.split)
    elif args.method == 'emeas1':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'{args.languages}_emea_s1', args.split)
    elif args.method == 'emeas10':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'{args.languages}_emea_s10', args.split)
    elif args.method == 'syntax_ensemble':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'{args.languages}_syntax_ensemble_en{args.en_weight}', args.split)
    elif args.method == 'learned_ensemble':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'{args.languages}_learned_ensemble_temp{args.temperature}', args.split)
    elif args.method == 'en_topk':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_en_top{args.topk}', args.split)
    elif args.method == 'en_topk_target':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_en_top{args.topk}_target', args.split)
    elif args.method == 'en_topk_hi_task':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_en_top{args.topk}_hi', args.split)
    elif args.method == 'mr_topk':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_mr_top{args.topk}', args.split)
    elif args.method == 'hi_topk':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_hi_top{args.topk}', args.split)
    elif args.method == 'hi_topk_emeas1':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'emea_s1_hi_top{args.topk}', args.split)
    elif args.method == 'hi_topk_emeas10':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'emea_s10_hi_top{args.topk}', args.split)
    elif args.method == 'hi_topk_hi_task':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_hi_top{args.topk}_hi', args.split)
    elif args.method == 'ru_topk':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_ru_top{args.topk}', args.split)
    elif args.method == 'ru_topk_emeas1':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'emea_s1_ru_top{args.topk}', args.split)
    elif args.method == 'is_topk':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_is_top{args.topk}', args.split)
    elif args.method == 'is_topk_emeas1':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'emea_s1_is_top{args.topk}', args.split)
    elif args.method == 'sw_topk':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_sw_top{args.topk}', args.split)
    elif args.method == 'wo_topk':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_wo_top{args.topk}', args.split)
    elif args.method == 'random':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_random', args.split)
    elif args.method == 'custom':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_custom', args.split)
    elif args.method == 'geo':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'geo_ensemble', args.split)
    elif args.method == 'en_topk_madx':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'ensemble_en_top{args.topk}_madx', args.split)
    elif args.method == 'ensemble_hi_task':
        infile = INFILE.format(dataset, args.model, maxlen, args.task, f'{args.languages}_ensemble_hi_task', args.split)
    else:
        print('INVALID METHOD')
        exit()
    print(infile)
    # infile = 'outputs/panx/my-bert-base-multilingual-cased-MaxLen128_ner_ensemble_en_topk/test_results.txt'
    results = defaultdict(dict)
    results_precision = defaultdict(dict)
    results_recall = defaultdict(dict)

    average_results = defaultdict(float)
    average_precision = defaultdict(float)
    average_recall = defaultdict(float)
    
    std_results = defaultdict(float)
    std_precision = defaultdict(float)
    std_recall = defaultdict(float)
    # print(infile)
    with open(infile, 'r') as f:
        for line in f:
            line = json.loads(line)
            language = line['language']
            seed = line['seed']
            if str(seed) not in args.seeds.split(','): continue
            f1 = line['f1']
            results[language][seed] = f1
            if 'precision' in line.keys():
                precision = line['precision']
                recall = line['recall']
                results_precision[language][seed] = precision
                results_recall[language][seed] = recall
    # print(results)

    for language in results.keys():
        if args.task != 'qna':
            results[language] = [100*results[language][seed] for seed in results[language].keys()]
        else:
            results[language] = [results[language][seed] for seed in results[language].keys()]
            results_precision[language] = [results_precision[language][seed] for seed in results_precision[language].keys()]
            results_recall[language] = [results_recall[language][seed] for seed in results_recall[language].keys()]

        num_seeds = len(results[language])
        # average_results[language] = average_results[language]/num_seeds
        # average_results[language] = np.mean([results[language][seed] for seed in results[language].keys()])
        # std_results[language] = np.std([results[language][seed] for seed in results[language].keys()])
        average_results[language] = np.mean(results[language])
    
        # average_precision[language] = np.mean(results_precision[language])
        # average_recall[language] = np.mean(results_recall[language])
        std_results[language] = np.std(results[language])
        # std_precision[language] = np.std(results_precision[language])
        # std_recall[language] = np.std(results_recall[language])
        if args.eval_lang is None:
            print(f'Average f1 score over {num_seeds} seeds for language {language} = {format(average_results[language],".2f")} +- {format(std_results[language],".2f")}')
            print(f'Average precision score over {num_seeds} seeds for language {language} = {format(average_precision[language],".2f")} +- {format(std_precision[language],".2f")}')
            print(f'Average recall score over {num_seeds} seeds for language {language} = {format(average_recall[language],".2f")} +- {format(std_recall[language],".2f")}')
            print('----------------------------------------------------------------')
        else:
            print_json = {}
            print_json['adapter'] = args.languages
            if language == args.eval_lang:
                print_json['num_seeds'] = num_seeds
                print_json['f1'] = average_results[language]
                print(json.dumps(print_json))
if __name__ == '__main__':
    main()

    