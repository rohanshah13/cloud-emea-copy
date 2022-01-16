import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval_lang', default='mr')
parser.add_argument('--task', default='udpos')
parser.add_argument('--model', default='bert-base-multilingual-cased')
args = parser.parse_args()

LANGUAGES = "am,ar,bh,bn,bxr,cdo,cs,de,el,en,es,et,eu,fa,fi,fr,gn,hi,ht,hu,hy,id,ilo,is,ja,jv,ka,ko,kv,la,lv,mhr,mi,my,myv,pt,qu,ru,se,sw,tk,tr,vi,wo,xmf,zh,zh_yue".split(',')
# LANGUAGES = "ar,cdo,de,el,en,es,et,gn,hi,ht,id,ilo,is,it,ja,jv,mhr,mi,my,qu,ru,sw,ta,th,tk,tr,vi,xmf,zh".split(',')
COMMAND = 'python3 scripts/my_average_f1.py --task {} --model {} --split test --method related --languages {} --seeds {} --eval_lang {} >> scripts/{}/en/{}.json'
SORT_COMMAND = f'python3 scripts/sort.py --task {args.task} --language {args.eval_lang}'

def main():
    for lang in LANGUAGES:
        subprocess.run(COMMAND.format(args.task, args.model, lang, '1,2,3', args.eval_lang, args.task, args.eval_lang), shell=True)

    for seed in [1,2,3]:
        for lang in LANGUAGES:
            subprocess.run(COMMAND.format(args.task, args.model, lang, f'{seed}', args.eval_lang, args.task, f'{args.eval_lang}_s{seed}'), shell=True)

    subprocess.run(SORT_COMMAND, shell=True)

if __name__ == '__main__':
    main() 