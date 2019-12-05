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


def get_csv_paths(parent, kind):
    names = [name for name in os.listdir(parent) if (name.endswith('.tsv') or name.endswith('.csv'))]
    if len(names) == 0:
        raise RuntimeError('No .csv or .tsv files in {}'.format(kind))
    return [os.path.join(parent, name) for name in names]


def get_data(paths):
    data = {}
    assert sum([os.path.isfile(truth_file) for truth_file in paths])
    for path in paths:
        if 'public' in path:
            data['public'] = read_dataset(path)
        if 'private' in path:
            data['private'] = read_dataset(path)
    return data


def save_to_file(words_with_hypernyms, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        for word, hypernyms in words_with_hypernyms.items():
            f.write(word + "\t" + ", ".join(hypernyms) + "\n")
