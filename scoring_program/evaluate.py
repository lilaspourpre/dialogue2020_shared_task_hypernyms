#!/usr/bin/env python
import argparse
import os
import sys

from utils import get_submitted, get_reference


# This is python 2 code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    output_dir = sys.argv[2]

    submit_dir = os.path.join(args.input_dir, 'res')
    truth_dir = os.path.join(args.input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        raise RuntimeError('{} does not exist'.format(submit_dir))

    if not os.path.isdir(truth_dir):
        raise RuntimeError('{} does not exist'.format(truth_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')

    with open(output_filename, 'wt') as output_file:
        all_true, direct_parents = get_reference(truth_dir)
        submitted = get_submitted(submit_dir)
        if set(all_true) != set(submitted):
            print("Not all words are presented in your file")
        mean_ap, mean_rr = get_score(all_true, direct_parents, submitted)
        output_file.write("map: {0}\nmrr: {1}\n".format(mean_ap, mean_rr))


def get_score(all_true, direct_true, predicted, k=10):
    ap_sum = 0
    rr_sum = 0

    for neologism in all_true:
        # getting sets of hypernyms for true and predicted
        all_hypernyms = set(all_true.get(neologism, []))
        direct_hypernyms = set(direct_true.get(neologism, []))
        predicted_hypernyms = predicted.get(neologism, [])

        # get metrics
        ap_sum += max(compute_ap(all_hypernyms, predicted_hypernyms, k),
                      compute_ap(direct_hypernyms, predicted_hypernyms, k))
        rr_sum += max(compute_rr(all_hypernyms, predicted_hypernyms, k),
                      compute_rr(direct_hypernyms, predicted_hypernyms, k))

    return ap_sum / len(all_true), rr_sum / len(all_true)


def compute_ap(actual, predicted, k=10):
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def compute_rr(true, predicted, k=10):
    for i, synset in enumerate(predicted[:k]):
        if synset in true:
            return 1.0 / (i + 1.0)
    return 0.0


if __name__ == '__main__':
    main()
