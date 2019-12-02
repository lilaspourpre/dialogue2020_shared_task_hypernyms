from collections import defaultdict
import codecs
import os


def read_dataset(data_path, sep='\t'):
    vocab = defaultdict(set)
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0]
            hypernyms = set(line_split[1].split(", "))
            vocab[word].update(hypernyms)
    return vocab


def get_one_csv_path(parent, kind):
    names = [name for name in os.listdir(parent)
             if name.endswith('.tsv') or name.endswith('.csv')]
    if len(names) == 0:
        raise RuntimeError('No .csv or .tsv files in {}'.format(kind))
    if len(names) > 1:
        raise RuntimeError('Multiple files in {}: {}'
                           .format(kind, ' '.join(names)))
    name, = names
    return os.path.join(parent, name)


def save_to_file(words_with_hypernyms, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        for word, hypernyms in words_with_hypernyms.items():
            f.write(word + "\t" + ", ".join(hypernyms) + "\n")
