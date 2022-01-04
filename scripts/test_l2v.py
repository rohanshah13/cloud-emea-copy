import lang2vec.lang2vec as l2v
from iso639 import languages

MY_LANGUAGES = ['be', 'bg', 'bn', 'da', 'de', 'en', 'fo', 'hi', 'is', 'mr', 'no', 'ru', 'ta', 'uk']
available_languages = list(l2v.LEARNED_LANGUAGES)
# 
language_names = []
language_codes = []
for language_code in available_languages:
    try:
        language = languages.get(part3=language_code)
    except:
        print(language_code)
    # print(language.name)
exit()
language_codes = []
for code in available_languages:
    try:
        language = languages.get(part3=code)
        language_names.append(language.name)
        language_codes.append(language.alpha2)
    except:
        continue
language_names = sorted(language_names)

# for language in MY_LANGUAGES:
    # print(f'Language {language}: Found = {language in language_codes}')

# print(l2v.get_features("hi", "fam"))

# features = l2v.get_features('be bg')