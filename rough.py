import argparse
import json

PATH = 'scripts/udpos/{}.json'

parser = argparse.ArgumentParser()
parser.add_argument('--lang', default='hi')
args = parser.parse_args()

all_adapters = []

with open(PATH.format('en')) as f:
    for line in f:
        line = json.loads(line.strip())
        all_adapters.append(line['adapter'])

lang_adapters = []
incomplete_adapters = []
with open(PATH.format(args.lang)) as f:
    for line in f:
        line = json.loads(line.strip())
        lang_adapters.append(line['adapter'])
        if line['num_seeds'] < 3:
            incomplete_adapters.append(line['adapter'])

rem_adapters = [f'"{adapter}"' for adapter in all_adapters if adapter not in lang_adapters] + [f'"{adapter}"' for adapter in incomplete_adapters]
rem_adapter_names = [f'"{adapter}/wiki@ukp"' for adapter in all_adapters if adapter not in lang_adapters] + [f'"{adapter}/wiki@ukp"' for adapter in incomplete_adapters]

print(f"ADAPTERS_LANGS=( {' '.join(rem_adapters)} )")
print(f"LANG_ADAPTER_NAMES=( {' '.join(rem_adapter_names)} )")