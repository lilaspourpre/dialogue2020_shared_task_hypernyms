import argparse
from collections import defaultdict
from nltk.corpus import WordNetCorpusReader

from vectorizers.fasttext_vectorizer import FasttextVectorizer
import os


def extract_senses(synsets) -> dict:
    result = defaultdict(list)
    for synset in synsets:
        for lemma in synset.lemmas():
            result[synset.name()].append(lemma.name().replace("_", " "))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="get context with positions")
    parser.add_argument('--data_path', type=str, dest="data_path", help="data path with terms to vectorize", required=False)
    parser.add_argument('--data_dir', type=str, dest="data_dir", help="data dir with files to vectorize", required=False)
    parser.add_argument('--fasttext', type=str, dest="fasttext_path", help='fasttext path')
    parser.add_argument('--output_path', type=str, dest="output_path", help='output path')
    parser.add_argument('--language', type=str, dest="language", help='output path', default="EN")
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "ruwordnet" command
    wordnet = subparsers.add_parser('wordnet', help='ruwordnet help')
    wordnet.add_argument('--wordnet', type=str, help='wordnet database path')
    wordnet.add_argument('--pos', choices='nv', help="choose pos-tag to subset ruwordnet")

    return parser.parse_args()


def main():
    args = parse_args()
    ft_vec = FasttextVectorizer(args.fasttext_path)

    if args.data_path:
        # read data
        with open(args.data_path, 'r', encoding='utf-8') as f:
            dataset = [line.split("\t")[1].replace(" ", "_") for line in f.read().split("\n") if line]

        # vectorize wordnet
        if "wordnet" in args:
            wn = WordNetCorpusReader(args.wordnet, None)
            for word in dataset:
                print(word, wn.synsets(word, pos=args.pos))
        else:
            ft_vec.vectorize_multiword_data(dataset, args.output_path, to_upper=False)

    elif args.data_dir:
        for system_dir in os.listdir(args.data_dir):
            for dirpath, _, filenames in os.walk(os.path.join(args.data_dir, system_dir, args.language)):
                for filename in filenames:
                    if filename.endswith(".terms"):
                        input_path = os.path.join(dirpath, filename)
                        os.makedirs(os.path.join(args.output_path, system_dir), exist_ok=True)
                        output_path = os.path.join(args.output_path, system_dir,
                                               filename.replace(".terms", ".txt").replace(system_dir+"_", ""))
                        with open(input_path, 'r', encoding='utf-8') as f:
                            dataset = [line.split("\t")[1].replace(" ", "_") for line in f.read().split("\n") if line]
                        ft_vec.vectorize_multiword_data(dataset, output_path, to_upper=False)
                        print(f"Processed: {filename}")
    else:
        raise Exception("Please, specify either --data_dir or --data_path")


if __name__ == '__main__':
    main()
