#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import os
import sys

from utils import get_csv_path, read_dataset


# This is python 2 code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--ruwordnet_path', required=False, default=None)
    parser.add_argument('--db_path', required=False, default=None)
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
        truth = read_dataset(get_csv_path(truth_dir, 'reference'))
        submitted = read_dataset(get_csv_path(submit_dir, 'submission'))
        if set(truth) != set(submitted):
            print("Not all words are presented in your file")
        mean_ap, mean_rr = get_score(truth, submitted)
        output_file.write("map: {0}\nmrr: {1}\n".format(mean_ap, mean_rr))


def get_score(true, predicted, k=10):
    ap_sum = 0
    rr_sum = 0

    for neologism in true:
        # getting sets of hypernyms for true and predicted
        true_hypernyms = set(true.get(neologism, []))
        predicted_hypernyms = predicted.get(neologism, [])

        # get metrics
        ap_sum += compute_ap(true_hypernyms, predicted_hypernyms, k)
        rr_sum += compute_rr(true_hypernyms, predicted_hypernyms, k)

    return ap_sum / len(true), rr_sum / len(true)


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
            return 1/(i+1)
    return 0
