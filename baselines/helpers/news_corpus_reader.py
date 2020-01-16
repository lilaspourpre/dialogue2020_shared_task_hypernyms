import os
import argparse
import json
import time
from collections import defaultdict

from ruwordnet.ruwordnet_reader import RuWordnet


def retrieve_sentences_with_positions(input_filename: str, output_dir: str, synset_senses: dict, testset: set,
                                      sense_dict: dict) -> int:
    overall_tokens = 0
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_filename))[0])
    tokens = []
    lemmas = []
    word_positions = defaultdict(list)
    ruwordnet_positions = defaultdict(list)
    sense_start = 0
    sense_phrase = []

    with open(input_filename, 'rt', encoding='utf-8') as f, \
            open(output_path, 'w', encoding='utf-8') as w:
        for line in f:
            if line == "\n":
                overall_tokens += len(tokens)
                # add last sense phrase if exist
                if len(sense_phrase) > 0:
                    ruwordnet_positions[" ".join(sense_phrase).upper()].append((sense_phrase, len(tokens)))

                # save to file
                if len(tokens) > 0 and (ruwordnet_positions or word_positions):
                    ruwordnet_positions = dict([(sense_dict[name], indices)
                                                for name, indices in ruwordnet_positions.items() if name in sense_dict])
                    w.write(json.dumps([tokens, {"ruwordnet": ruwordnet_positions,
                                                 "words": word_positions}]) + "\n")

                # clean buffer
                tokens = []
                lemmas = []
                word_positions = defaultdict(list)
                ruwordnet_positions = defaultdict(list)
                sense_start = 0
                sense_phrase = []

            elif not line.startswith("#"):
                split_line = line[:-1].split("\t")

                index = int(split_line[0])
                token = split_line[1]
                lemma = split_line[2]

                tokens.append(token)
                lemmas.append(lemma)

                # add position of words from testset
                if lemma in testset:
                    word_positions[lemma].append((index, index + 1))

                # compute position of synset
                if len(sense_phrase) == 0:
                    if lemma in synset_senses:
                        sense_start = index
                        sense_phrase = [lemma]

                elif lemma in synset_senses[sense_phrase[-1]]:
                    sense_phrase.append(lemma)

                else:
                    ruwordnet_positions[" ".join(sense_phrase).upper()].append((sense_start, index))

                    if lemma in synset_senses:
                        sense_start = index
                        sense_phrase = [lemma]
                    else:
                        sense_start = 0
                        sense_phrase = []
    return overall_tokens


def create_synset_senses(text):
    tokens = text.split()
    return [(token, next_token) for token, next_token in zip(tokens, tokens[1:] + [True])]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest="input_dir")
    parser.add_argument('-o', '--output', dest="output_dir")
    parser.add_argument('-t', '--test', dest="test_path")
    parser.add_argument('-r', '--ruwordnet', dest="ruwordnet_path")
    parser.add_argument('-s', '--start', dest="start", type=int)
    parser.add_argument('-e', '--end', dest="end", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    nb = 0
    ruwordnet = RuWordnet(db_path=args.ruwordnet_path, ruwordnet_path="")
    sense_dict = dict([(text, synset) for id, synset, text in ruwordnet.get_all_senses()])
    print("----- database loaded -----")

    synset_senses = defaultdict(set)
    for _, synset, text in ruwordnet.get_all_senses():
        for token, next_token in create_synset_senses(text.lower()):
            synset_senses[token].add(next_token)
    print("----- synsets loaded -----")

    testset = set()
    for path in os.listdir(args.test_path):
        with open(os.path.join(args.test_path, path), 'r', encoding='utf-8') as f:
            testset.update(f.read().lower().split("\n")[:-1])
    print("----- testset loaded -----")

    # retrieve positions
    file_paths = [os.path.join(x, i) for x, _, z in os.walk(args.input_dir) for i in z]  # [args.start: args.end]
    for filename in file_paths:
        start_time = time.time()
        nb += retrieve_sentences_with_positions(filename, args.output_dir, synset_senses, testset, sense_dict)
        print(f"---- File {filename} took {(time.time() - start_time)} seconds ----")

    print(f"overall tokens: {nb}")


if __name__ == '__main__':
    main()
