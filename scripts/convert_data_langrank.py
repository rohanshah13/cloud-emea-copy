import argparse

INFILE = 'data/udpos/udpos_processed_maxlen128/{}/{}.bert-base-multilingual-cased'
# OUTFILE = '../langrank/sample-data/udpos-{}.{}'
OUTFILE = '../mlm-scoring/data/udpos-{}.{}'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='mr')
    parser.add_argument('--split', default='test')
    args = parser.parse_args()

    infile = INFILE.format(args.lang, args.split)
    outfile = OUTFILE.format(args.split, args.lang)

    output_lines = []
    curr_line = ''
    with open(infile) as f:
        for line in f:
            line = line.split()
            if len(line) != 0:
                curr_line += f'{line[0]} '
            else:
                output_lines.append(curr_line.strip())
                curr_line = ''
        output_lines.append(curr_line.strip())
        

    with open(outfile, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
            # print(line)


if __name__ == '__main__':
    main()