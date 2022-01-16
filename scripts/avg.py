import json
import operator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='ner')
args = parser.parse_args()

scores = {}

def load_file(filename):
    my_list = []
    with open(filename) as f:
        for line in f:
            my_list.append(json.loads(line))
    return my_list

NER_LANGS = ['en', 'hi', 'ru', 'is']
POS_LANGS = ['ar', 'de', 'en', 'es', 'eu', 'fi', 'hi', 'hu', 'is', 'mr', 'pt', 'ru', 'zh']

POS_FILENAMES = [f'scripts/udpos/en/{lang}.json' for lang in NER_LANGS]
NER_FILENAMES = [f'scripts/udpos/en/{lang}.json' for lang in POS_LANGS]

if args.task == 'ner':
    FILENAMES = NER_FILENAMES
else:
    FILENAMES = POS_FILENAMES

d = []

for filename in FILENAMES:
    d.append(load_file(filename))


for i in range(len(FILENAMES)):
    # normalize each set of scores
    # make a list of f1 scores
    flist = []
    for x in d[i]:
        flist.append(x["f1"])
    maxval = max(flist)
    minval = min(flist)
    # (f- min)/(max-min)
    for x in d[i]:
        x["f1"] = (x["f1"]-minval)/(maxval-minval)

    

for i in range(len(FILENAMES)):
    for x in d[i]:
        scores[x["adapter"]] = scores.get(x["adapter"],0) + x["f1"]

print(sorted(scores.items(),key=operator.itemgetter(1), reverse=True)[:10])

