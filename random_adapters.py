import numpy as np

ADAPTER_LANGS="am,ar,bxr,cdo,cs,de,el,en,es,et,eu,fa,fi,fr,gn,hi,ht,hu,hy,id,ilo,is,ja,jv,ka,ko,kv,la,lv,mhr,mi,my,myv,pt,qu,ru,se,sw,tk,tr,vi,wo,xmf,zh,zh_yue".split(',')

np.random.seed(1)

adapters = np.random.choice(ADAPTER_LANGS, 10, replace=False)

print(','.join(adapters))