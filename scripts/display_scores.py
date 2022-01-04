import numpy as np
import torch
import argparse
import json
import os
import json

LANGS='am,ar,bxr,cdo,cs,de,el,en,es,et,eu,fa,fi,fr,gn,hi,ht,hu,hy,id,ilo,is,ja,jv,ka,ko,kv,la,lv,mhr,mi,my,myv,pt,qu,ru,se,sw,tk,tr,vi,wo,xmf,zh,zh_yue'.split(',')
INFILE = 'outputs/{}/cloud-weights-100-bert-base-multilingual-cased-MaxLen128_{}_{}/test_{}_s{}_importances_100.pt'
TARGETS = ['mr', 'bn', 'ta', 'fo', 'no', 'da', 'be', 'uk', 'bg']
TARGETS = ['af', 'bm', 'yo']
SEEDS = [1, 2, 3]

if not os.path.exists('weights/panx'):
    os.makedirs('weights/panx')
if not os.path.exists('weights/udpos'):
    os.makedirs('weights/udpos')

def my_format(val):
    return str("{:.6f}".format(val))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='udpos')
    args = parser.parse_args()

    if args.task == 'ner':
        dataset = 'panx'
    elif args.task == 'udpos':
        dataset = 'udpos'
    else:
        print('Invalid Task')
        exit()

    for target in TARGETS:
        if not os.path.exists(f'weights/{dataset}/{target}/'):
            os.mkdir(f'weights/{dataset}/{target}/')
        for seed in SEEDS:
            weights = []
            with open(f'weights/{dataset}/{target}/scores_s{seed}', 'w') as f:
                for lang in LANGS:
                    infile = INFILE.format(dataset, args.task, lang, target, seed)
                    if not os.path.exists(infile):
                        print(infile)
                        print('File Not Found')
                        continue    
                    importances = torch.load(infile)
                    importances = torch.mean(importances, dim=-1)
                    weights.append(torch.mean(importances).item())
                    importances = importances.tolist()
                    importances = [my_format(val) for val in importances]
                    f.write(f'=============Language = {lang}=========================\n')
                    f.write(json.dumps(importances)+'\n')
            weights = [w/sum(weights) for w in weights]
            with open(f'weights/{dataset}/{target}/weights_s{seed}', 'w') as f:
                f.write(json.dumps(weights))
            top3_indices = np.flip(np.argsort(weights))
            with open(f'weights/{dataset}/{target}/top3_s{seed}', 'w') as f:
                for index in top3_indices:
                    f.write(LANGS[index] + '\n')

if __name__ == '__main__':
    main()