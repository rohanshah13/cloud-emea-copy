LANGS = ["hi,my,fa,tk,vi,zh,ka,zh_yue,hy,ru", "hi,my,fa,vi,zh,tk,zh_yue,ru,bxr,cdo", "hi,my,fa,vi,tk,jv,zh_yue,id,zh,am", "is,en,se,et,fi,fr,kv,cs,de,eu", "et,lv,fi,en,se,cs,de,fr,is,hu", "cs,en,de,fr,lv,et,hu,fi,la,eu", "lv,et,cs,hu,de,el,fi,myv,mhr,tr", "hu,el,cs,de,lv,tr,la,et,xmf,myv", "el,hu,tr,la,cs,de,xmf,lv,hy,ka"]

my_str = ''
counter = 0
for langs in LANGS:
    print(counter)
    counter += 1
    assert len(langs.split(',')) == 10 
    adapter_names = [f'{lang}/wiki@ukp' for lang in langs.split(',')]
    adapter_names = ','.join(adapter_names)
    if len(my_str) == 0:
        my_str = f'"{adapter_names}"'
    else:
        my_str = my_str + f' "{adapter_names}"'

print(my_str)