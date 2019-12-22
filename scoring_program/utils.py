from collections import defaultdict
import codecs
import os


def read_dataset(data_path, sep='\t'):
    vocab = defaultdict(list)
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0]
            hypernyms = line_split[1]
            vocab[word].append(hypernyms)
    return vocab


def get_submitted(parent):
    names = [name for name in os.listdir(parent) if (name.endswith('.tsv') or name.endswith('.csv'))]
    if len(names) == 0:
        raise RuntimeError('No .csv or .tsv files in submitted')
    if len(names) > 1:
        raise RuntimeError('Multiple files in submitted: {}'.format(' '.join(names)))
    return read_dataset(os.path.join(parent, names[0]))


def get_reference(parent):
    names = [os.path.join(parent, name) for name in os.listdir(parent) if (name.endswith('.tsv') or name.endswith('.csv'))]
    if len(names) == 0:
        raise RuntimeError('No .csv or .tsv files in reference')
    if len(names) != 2:
        raise RuntimeError('There should be exact two files in reference: {}'.format(' '.join(names)))
    return read_dataset(names[0]), read_dataset(names[1])
