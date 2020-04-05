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
        true = get_reference(truth_dir)
        submitted = get_submitted(submit_dir)
        if len(set(true).intersection(set(submitted))) == 0:
            raise Exception("Reference and Submitted files have no samples in common")
        elif set(true) != set(submitted):
            print("Not all words are presented in your file")
        mean_ap, mean_rr = get_score(true, submitted, 10)
        output_file.write("map: {0}\nmrr: {1}\n".format(mean_ap, mean_rr))


def get_score(true, predicted, k=10):
    ap_sum = 0
    rr_sum = 0

    for neologism in true:
        # getting sets of hypernyms for true and predicted
        true_hypernyms = true.get(neologism, [])
        predicted_hypernyms = predicted.get(neologism, [])

        # get metrics
        ap_sum += compute_ap(true_hypernyms, predicted_hypernyms, k)
        rr_sum += compute_rr(set([j for i in true_hypernyms for j in i]), predicted_hypernyms, k)
    return ap_sum / len(true), rr_sum / len(true)


def compute_ap(actual, predicted, k=10):
    if not actual:
        return 0.0

    predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    already_predicted = set()
    skipped = 0
    for i, p in enumerate(predicted):
        if p in already_predicted:
            skipped += 1
            continue
        for parents in actual:
            if p in parents:
                num_hits += 1.0
                score += num_hits / (i + 1.0 - skipped)
                already_predicted.update(parents)
                break

    return score / min(len(actual), k)


def compute_rr(true, predicted, k=10):
    for i, synset in enumerate(predicted[:k]):
        if synset in true:
            return 1.0 / (i + 1.0)
    return 0.0


if __name__ == '__main__':
    main()
