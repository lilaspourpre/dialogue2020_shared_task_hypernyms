#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import os
import sys

from utils import get_data, get_csv_paths, read_dataset
from ruwordnet.ruwordnet_reader import RuWordnet


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

        if args.ruwordnet_path is None and args.db_path is None:
            ruwordnet = RuWordnet(ruwordnet_path=None, db_path=os.path.join(truth_dir, 'ruwordnet.db'))
        else:
            ruwordnet = RuWordnet(args.ruwordnet_path, args.db_path)

        truth_files = get_csv_paths(truth_dir, 'reference')
        submitted_files = get_csv_paths(submit_dir, 'submission')
        if len(truth_files) == 1:
            write_scores_to_file(read_dataset(truth_files[0]), read_dataset(submitted_files[0]),
                                 ruwordnet, 'public', output_file)
        elif len(truth_files) == 2:
            truth_data = get_data(truth_files)
            submitted_data = get_data(submitted_files)

            if 'public' in truth_data and 'public' in submitted_data:
                write_scores_to_file(truth_data['public'], submitted_data['public'],
                                     ruwordnet, 'public', output_file)

            if 'private' in truth_data and 'private' in submitted_data:
                write_scores_to_file(truth_data['private'], submitted_data['private'],
                                     ruwordnet, 'private', output_file)


def write_scores_to_file(truth, submitted, ruwordnet, dataset_type, output_file):
    f1, jaccard, jaccard_weighted = get_score(truth, submitted, ruwordnet)
    output_file.write("f1_{0}: {1}\njaccard_{0}: {2}\njaccard_weighted_{0}: {3}\n".format(dataset_type,
                                                                                          f1, jaccard,
                                                                                          jaccard_weighted))


def get_score(true, predicted, ruwordnet):
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
        additional_sum = get_additional_weights(true_hypernyms, predicted_hypernyms, ruwordnet)
        jaccard_sum += compute_jaccard(true_hypernyms, predicted_hypernyms)
        soft_jaccard += compute_jaccard(true_hypernyms, predicted_hypernyms, additional_sum)
        f1_sum += compute_f1(true_hypernyms, predicted_hypernyms, additional_sum)

    return f1_sum / len(true), jaccard_sum / len(true), soft_jaccard / len(true)


def compute_jaccard(true, predicted, additional_sum=0):
    intersection = len(true.intersection(predicted)) + additional_sum
    return intersection / len(true.union(predicted))


def get_additional_weights(true, predicted, ruwordnet):
    relatives = set([relative for synset in predicted for relative in get_relatives(synset, ruwordnet)])
    weights = [0.5 for true_hypernym in true.difference(predicted) if true_hypernym in relatives]
    return sum(weights)


def get_relatives(synset, ruwordnet):
    return ruwordnet.get_hypernyms_by_id(synset) + ruwordnet.get_hyponyms_by_id(synset)


def compute_f1(true, predicted, additional_sum):
    tp = len(true.intersection(predicted)) + additional_sum
    fp = len(predicted.difference(true)) - additional_sum
    fn = len(true.difference(predicted)) - additional_sum

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


if __name__ == '__main__':
    main()
