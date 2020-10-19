import os
import argparse
import json
import time
from collections import defaultdict

from nltk.corpus import WordNetCorpusReader
from tqdm import tqdm

from fasttext_vectorize_en import compute_synsets_from_wordnets
from ruwordnet.ruwordnet_reader import RuWordnet

found_lemmas = defaultdict(int)


# -------------------------------------------------------------
# get ruwordnet
# -------------------------------------------------------------

def retrieve_ruwordnet_positions(input_filename: str, output_path: str, synset_senses: dict, sense2synset: dict):
    with open(input_filename, 'rt', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as w:
        for sentence in f:
            tokenized, lemmatized = sentence[:-1].split('\t')
            occurences = []
            for index, (token, lemma) in enumerate(zip(tokenized.split(), lemmatized.split())):
                synsets, end = get_end(sentence, lemma, index + 1, synset_senses, sense2synset)
                if synsets:
                    for synset in synsets:
                        occurences.append((synset, (index, end)))
                        found_lemmas[synset] += 1

                synsets, end = get_end(sentence, token, index + 1, synset_senses, sense2synset)
                if synsets:
                    for synset in synsets:
                        occurences.append((synset, (index, end)))
                        found_lemmas[synset] += 1
            if occurences:
                w.write(json.dumps([tokenized, occurences]) + "\n")


def get_end(sentence, first_lemma, index, senses_chain, sense2synset):
    last_index = index
    if first_lemma in senses_chain:
        sense_phrase = [first_lemma]
        for cur_index, lemma in enumerate(sentence[index:]):
            if lemma in senses_chain[sense_phrase[-1]]:
                sense_phrase.append(lemma)
                last_index = index + int(cur_index)
            else:
                break
        if len(sense2synset[" ".join(sense_phrase).upper()]) > 1:
            return sense2synset[" ".join(sense_phrase).upper()], last_index
    return False, last_index


# -------------------------------------------------------------
# get test data
# -------------------------------------------------------------


def retrieve_word_positions(input_filename, output_path, testset) -> None:
    with open(input_filename, 'rt', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as w:
        for sentence in f:
            tokenized, lemmatized = sentence[:-1].split('\t')
            lemmas = []
            for index, (token, lemma) in enumerate(zip(tokenized.split(), lemmatized.split())):
                if lemma in testset:
                    lemmas.append((lemma.upper(), (index, index + 1)))
                    found_lemmas[lemma] += 1
                if token in testset:
                    lemmas.append((token.upper(), (index, index + 1)))
                    found_lemmas[token] += 1
            if lemmas:
                w.write(json.dumps([tokenized, lemmas]) + "\n")


# -------------------------------------------------------------
#  ruwordnet transformations
# -------------------------------------------------------------

def create_senses_data(all_senses, pos, lower=True):
    synset_senses = defaultdict(set)
    sense2synset = defaultdict(list)

    for _, synset, text in all_senses:
        if synset.endswith(pos):
            sense2synset[text].append(synset)
            text = text.lower() if lower else text
            for token, next_token in create_synset_senses(text):
                synset_senses[token].add(next_token)
    return synset_senses, sense2synset


def create_synset_senses(text):
    tokens = text.split()
    return [(token, next_token) for token, next_token in zip(tokens, tokens[1:] + [True])]


def read_test_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set(f.read().lower().split("\n")[:-1])


def parse_args():
    # create the top-level parser
    parser = argparse.ArgumentParser(description="get context with positions")
    parser.add_argument('--corpus_path', type=str, dest="corpus_path", help="lemmatized ud news corpus path")
    parser.add_argument('--output_path', type=str, dest="output_path", help='output_path')
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "ruwordnet" command
    ruwordnet_parser = subparsers.add_parser('ruwordnet', help='ruwordnet help')
    ruwordnet_parser.add_argument('--ruwordnet_path1', type=str, help='ruwordnet database path')
    ruwordnet_parser.add_argument('--ruwordnet_path2', type=str, help='ruwordnet database path')
    ruwordnet_parser.add_argument('--pos', choices='NV', help="choose pos-tag to subset ruwordnet")

    # create the parser for the "wordnet" command
    wordnet_parser = subparsers.add_parser('wordnet', help='ruwordnet help')
    wordnet_parser.add_argument('--wordnet_old', type=str, help='wordnet old path')
    wordnet_parser.add_argument('--wordnet_new', type=str, help='wordnet new path')
    wordnet_parser.add_argument('--pos', choices='nv', help="choose pos-tag to subset ruwordnet")

    # create the parser for the "data" command
    parser_b = subparsers.add_parser('data', help='data help')
    parser_b.add_argument('--data_path', type=str, dest="data_path", help='path to test data')

    return parser.parse_args()


def main():
    args = parse_args()

    description1 = "---- File {0} took {1} seconds ----\n"
    description2 = "All: {2}, Found: {3}, Left: {4}"
    description = description1 + description2

    if "ruwordnet_path1" in args:
        file_paths = tqdm([os.path.join(x, i) for x, _, z in os.walk(args.corpus_path) for i in z])

        # ------------ RuWordnet initialization ------------
        ruwordnet1 = RuWordnet(db_path=args.ruwordnet_path1, ruwordnet_path="")
        ruwordnet2 = RuWordnet(db_path=args.ruwordnet_path2, ruwordnet_path="")
        senses = ruwordnet1.get_all_senses() + ruwordnet2.get_all_senses()
        synset_senses, sense2synset = create_senses_data(senses, args.pos)
        synsets = set(ruwordnet1.get_all_ids(args.pos))
        print(sense2synset)
        # ------------ Find contexts ------------
        # for filename in file_paths:
        #     start_time = time.time()
        #     retrieve_ruwordnet_positions(filename, args.output_path, synset_senses, sense2synset)
        #     file_paths.set_description(description.format(filename, (time.time() - start_time),
        #                                                   len(synsets), len(found_lemmas),
        #                                                   len(synsets.difference(set(found_lemmas)))))
        #
        # print(description2.format(len(synsets), len(found_lemmas), len(synsets.difference(set(found_lemmas)))))
        # print(found_lemmas)
        # print(synsets.difference(set(found_lemmas)))

    if "wordnet_old" in args:
        wordnet_old = WordNetCorpusReader(args.wordnet_old, None)
        wordnet_new = WordNetCorpusReader(args.wordnet_new, None)
        synsets = compute_synsets_from_wordnets(wordnet_old, wordnet_new, 'n')

        for synset in synsets:
            print(set([i.name() for i in wordnet_old.synset(synset).lemmas()] +
                  [i.name() for i in wordnet_new.synset(synset).lemmas()]))
        # for filename in file_paths:
        #     start_time = time.time()
        #     retrieve_ruwordnet_positions(filename, args.output_path, synset_senses, sense2synset)
        #     file_paths.set_description(description.format(filename, (time.time() - start_time),
        #                                                   len(synsets), len(found_lemmas),
        #                                                   len(synsets.difference(set(found_lemmas)))))
        #
        # print(description2.format(len(synsets), len(found_lemmas), len(synsets.difference(set(found_lemmas)))))
        # print(found_lemmas)
        # print(synsets.difference(set(found_lemmas)))

    elif "data_path" in args:
        file_paths = tqdm([os.path.join(x, i) for x, _, z in os.walk(args.corpus_path) for i in z])

        data = read_test_data(args.data_path)
        for filename in file_paths:
            start_time = time.time()
            retrieve_word_positions(filename, args.output_path, data)
            file_paths.set_description(description.format(filename, (time.time() - start_time),
                                                          len(data), len(found_lemmas),
                                                          len(data.difference(set(found_lemmas)))))

        print(description2.format(len(data), len(found_lemmas), len(data.difference(set(found_lemmas)))))
        print(found_lemmas)
        print(data.difference(set(found_lemmas)))


if __name__ == '__main__':
    main()
