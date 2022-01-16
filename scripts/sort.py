import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--language', default='en')
parser.add_argument('--task', default='pos')
args = parser.parse_args()

OUTFILES = [f'scripts/{args.task}/en/{args.language}.json', f'scripts/{args.task}/en/{args.language}_s1.json', f'scripts/{args.task}/en/{args.language}_s2.json', f'scripts/{args.task}/en/{args.language}_s3.json']
# outfile = 'scripts/mr_ner_s1.json'
for outfile in OUTFILES:
    json_lines = []
    with open(outfile) as f:
        for line in f:
            line = line.strip()
            json_lines.append(json.loads(line))

    json_lines.sort(key = lambda x: x['f1'], reverse=True)

    with open(outfile, 'w') as f:
        for line in json_lines:
            f.write(json.dumps(line) + '\n')