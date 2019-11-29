import argparse
import time
from utils import *
from ruwordnet.ruwordnet_reader import RuWordnet

ruwordnet = RuWordnet("dataset/ruwordnet", db_path='../ruwordnet/ruwordnet.db')


def load_args():
    parser = argparse.ArgumentParser(description="evaluation script for Dialogue 2020 Hypernym Detection shared task")
    parser.add_argument("-t", "--true", dest="true",
                        required=True, help="path to file with true hypernyms")
    parser.add_argument("-p", "--predicted", dest="predicted",
                        required=True, help="path to file with predicted hypernyms")
    return parser.parse_args()


def compute_jaccard(true: set, predicted: set, additional_sum=0):
    intersection = len(true.intersection(predicted)) + additional_sum
    return intersection / len(true.union(predicted))


def get_additional_weights(true: set, predicted: set) -> int:
    relatives = set([relative for synset in predicted for relative in get_relatives(synset)])
    weights = [0.5 for true_hypernym in true.difference(predicted) if true_hypernym in relatives]
    return sum(weights)


def get_relatives(synset: str) -> list:
    return ruwordnet.get_hypernyms_by_id(synset) + ruwordnet.get_hyponyms_by_id(synset)


def compute_f1(true: set, predicted: set, additional_sum: float):
    tp = len(true.intersection(predicted)) + additional_sum
    fp = len(predicted.difference(true)) - additional_sum
    fn = len(true.difference(predicted)) - additional_sum

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def evaluate(true: dict, predicted: dict) -> tuple:
    all_correct = 0
    jaccard_sum = 0
    soft_jaccard = 0
    f1_sum = 0

    all_relatives = 0
    mean_relatives = 0

    for neologism in true:
        # getting sets of hypernyms for true and predicted
        true_hypernyms = true.get(neologism, set())
        predicted_hypernyms = predicted.get(neologism, set())

        relatives = 0
        for t_h in true_hypernyms:
            for p_h in predicted_hypernyms:
                if p_h != '' and ruwordnet.are_relatives(t_h, p_h):
                    relatives += 1
                    break
        all_relatives += relatives
        mean_relatives += relatives / len(true_hypernyms)

        # count all_correct
        all_correct += int(true_hypernyms == predicted_hypernyms)

        # get metrics
        additional_sum = get_additional_weights(true_hypernyms, predicted_hypernyms)
        jaccard_sum += compute_jaccard(true_hypernyms, predicted_hypernyms)
        soft_jaccard += compute_jaccard(true_hypernyms, predicted_hypernyms, additional_sum)
        f1_sum += compute_f1(true_hypernyms, predicted_hypernyms, additional_sum)
    return all_correct / len(true), jaccard_sum / len(true), soft_jaccard / len(true), f1_sum / len(true), \
           all_relatives/len([v for values in true.values() for v in values]), mean_relatives/len(true)


def main():
    args = load_args()
    true = read_dataset(args.true)
    predicted = read_dataset(args.predicted)
    start = time.time()
    all_correct, jaccard, soft_jaccard, f1, relatives_all, mean_graph = evaluate(true, predicted)
    print(f"Fully correct: {all_correct:.3}\n"
          f"Jaccard strict: {jaccard:.3}\n"
          f"Soft Jaccard: {soft_jaccard:.3}\n"
          f"F1 score: {f1:.3}\n"
          f'All relatives {relatives_all*100:.3}%\n'
          f'Mean relatives {mean_graph*100:.3}%\n')
    end = time.time()
    print(f'Time:{end - start}')


if __name__ == '__main__':
    main()
