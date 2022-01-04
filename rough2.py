import subprocess

LANGS = ['ar', 'de', 'en', 'es', 'eu', 'fi', 'hi', 'is', 'pt', 'mr', 'zh', 'hu', 'ru']
COMMAND = 'python scripts/run.py --eval_lang {}'

for lang in ['zh']:
    subprocess.run(COMMAND.format(lang), shell=True)