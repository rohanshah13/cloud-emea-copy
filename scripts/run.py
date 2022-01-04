import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval_lang', default='mr')
parser.add_argument('--task', default='udpos')
args = parser.parse_args()

LANGUAGES = "am,ar,bh,bn,bxr,cdo,cs,de,el,en,es,et,eu,fa,fi,fr,gn,hi,ht,hu,hy,id,ilo,is,ja,jv,ka,ko,kv,la,lv,mhr,mi,my,myv,pt,qu,ru,se,sw,tk,tr,vi,wo,xmf,zh,zh_yue".split(',')
COMMAND = 'python3 scripts/my_average_f1.py --task {} --split test --method related --languages {} --seeds {} --eval_lang {} >> scripts/{}/hi/{}.json'
SORT_COMMAND = f'python3 scripts/sort.py --task {args.task} --language {args.eval_lang}'

def main():
    for lang in LANGUAGES:
        subprocess.run(COMMAND.format(args.task, lang, '1,2,3', args.eval_lang, args.task, args.eval_lang), shell=True)

    for seed in [1,2,3]:
        for lang in LANGUAGES:
            subprocess.run(COMMAND.format(args.task, lang, f'{seed}', args.eval_lang, args.task, f'{args.eval_lang}_s{seed}'), shell=True)

    subprocess.run(SORT_COMMAND, shell=True)

if __name__ == '__main__':
    main() 