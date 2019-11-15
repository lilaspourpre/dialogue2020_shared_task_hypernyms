import argparse
import math
from utils import *


def load_args():
    parser = argparse.ArgumentParser(description="evaluation script for Dialogue 2020 Hypernym Detection shared task")
    parser.add_argument("-t", "--true", dest="true",
                        required=True, help="path to file with true hypernyms")
    parser.add_argument("-p", "--predicted", dest="predicted",
                        required=True, help="path to file with predicted hypernyms")
    parser.add_argument("-e", "--existing", action='store_true',
                        required=False, help="evaluate neologisms that appear from the predicted dataset only")
    return parser.parse_args()


def evaluate(true: dict, predicted: dict) -> tuple:
    precisions_macro_sum = 0
    precisions_micro_sum = 0
    all_correct = 0

    for neologism in true:
        # getting sets of hypernyms for true and predicted
        true_hypernyms = true.get(neologism, {})
        predicted_hypernyms = predicted.get(neologism, {})

        # count micro and macro sums
        intersection_length = len(true_hypernyms.intersection(predicted_hypernyms))
        precisions_macro_sum += intersection_length/len(true_hypernyms)
        precisions_micro_sum += intersection_length
        all_correct += math.floor(intersection_length/len(true_hypernyms))

    print(f"\tCorrect hyperonyms: {precisions_micro_sum} (out of {sum(map(len, true.values()))})")
    print(f"\tAll correct hyperonyms per neologism: {all_correct} (out of {len(true)} neologisms)")
    return precisions_macro_sum/len(true), precisions_micro_sum/sum(map(len, true.values()))


def main():
    args = load_args()
    true = read_dataset(args.true)
    predicted = read_dataset(args.predicted)

    if args.existing:
        print("Evaluating only on predicted neologisms:")
        true = {i: true[i] for i in set(true).intersection(set(predicted))}

    macro_precision, micro_precision = evaluate(true, predicted)
    print(f"\tprecision macro: {macro_precision:.3}\n\tprecision micro: {micro_precision:.3}")


if __name__ == '__main__':
    main()
