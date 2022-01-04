import torch
import argparse
import json
import os
import torch
import numpy as np

LANGS='am,ar,bxr,cdo,cs,de,el,en,es,et,eu,fa,fi,fr,gn,hi,ht,hu,hy,id,ilo,is,ja,jv,ka,ko,kv,la,lv,mhr,mi,my,myv,pt,qu,ru,se,sw,tk,tr,vi,wo,xmf,zh,zh_yue'.split(',')
INFILE = 'outputs/panx/cloud-weights-100-bert-base-multilingual-cased-MaxLen128_ner_{}/test_be_s1_all_importances_100.pt'

def my_format(val):
    return str("{:.6f}".format(val))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', default='del.txt')
    args = parser.parse_args()
    
    softmax_scores = []
    str_importances = []
    all_importances = []
    with open(args.outfile, 'w') as f:
        for lang in LANGS:
            if not os.path.exists(INFILE.format(lang)):
                continue    
            importances = torch.load(INFILE.format(lang))
            importances = torch.mean(importances, dim=-1)
            # importances = torch.max(importances, dim=0)
            print(lang)
            for i in range(importances.size()[0]):
                print(importances[i])
                if i == 15:
                    print(torch.mean(importances, dim=0))
                    print('===================')
                    break
            if lang == 'ar':
                exit()
            continue
            importances = importances.tolist()
            all_importances.append(importances)
            softmax_scores.append(importances)
            # importances = [my_format(val) for val in importances]
            str_importances.append(importances)
            # f.write(f'=============Language = {lang}=========================\n')
            # f.write(json.dumps(importances)+'\n')
        all_importances  = np.array(all_importances)
        all_importances = np.mean(all_importances, axis=1)
        all_importances = all_importances/np.sum(all_importances)
        all_importances = [my_format(val) for val in all_importances]
        print(all_importances)
        print(",".join(all_importances))
        # print(sum(all_importances))
if __name__ == '__main__':
    main()