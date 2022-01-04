def get_data(langs,start):
    # langs = langs[start:start+9]
    my_str1 = [f'"{lg}"' for lg in langs]
    my_str1 = ' '.join(my_str1)
    my_str2 = [f'"{lg}/wiki@ukp"' for lg in langs]
    my_str2 = ' '.join(my_str2)
    print(my_str1)
    print(my_str2)

langs = []
with open('tmp.txt') as f:
    for line in f:
        langs.append(line.strip())

for i in range(0,43,45):
    get_data(langs,i)