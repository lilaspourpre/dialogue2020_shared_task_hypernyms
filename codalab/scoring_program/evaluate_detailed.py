#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import codecs
import os
import sys
import scipy
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
import scipy.stats
# This is Python 2 code


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
        truth_file = get_one_csv_path(truth_dir, 'reference')
        assert os.path.isfile(truth_file)
        public_path = os.path.join(truth_dir, 'public.txt')
        with codecs.open(public_path, 'rt', encoding='utf8') as f:
            public_words = set(line.strip() for line in f)

        submission_file = get_one_csv_path(submit_dir, 'submission')

        assert os.path.isfile(submission_file)

        df = read_csv(truth_file)
        submission = read_csv(submission_file)

        words = list(df['word'].unique())
        private_words = {w for w in words if w not in public_words}
        submission = submission.loc[submission.index.isin(df.index)]
        submission = submission.loc[submission['word'].isin(words)]
        df['predict_sense_id'] = submission['predict_sense_id']
        df['predict_sense_id'].fillna('', inplace=True)
        score = {}
        for split_words, split in zip([public_words, private_words],
                                      ['public', 'private']):
            df_split = df.loc[df['word'].isin(split_words)]
            score[split] = ari_per_word_weighted(df_split)

        output_file.write('public:{}'.format(score['public']))
        output_file.write('\n')

        output_file.write('private:{}'.format(score['private']))
        output_file.write('\n')
        score_df(df, output_dir)


def get_one_csv_path(parent, kind):
    names = [name for name in os.listdir(parent)
             if name.endswith('.tsv') or name.endswith('.csv')]
    if len(names) == 0:
        raise RuntimeError('No .csv or .tsv files in {}'.format(kind))
    if len(names) > 1:
        raise RuntimeError('Multiple files in {}: {}'
                           .format(kind, ' '.join(names)))
    name, = names
    return os.path.join(parent, name)


def ari_per_word_weighted(df):
    words = {
        word: (adjusted_rand_score(df_word['gold_sense_id'].values,
                                   df_word['predict_sense_id'].values),
               len(df_word))
        for word in df.word.unique()
        for df_word in (df.loc[df['word'] == word],)}

    cumsum = sum(ari * count for ari, count in words.values())
    if cumsum == 0:
        return 0
    total = sum(count for _, count in words.values())
    return cumsum / total

def describe(a):
    quants = 'quartiles ' + ' '.join(('%.2f' % x for x in scipy.stats.mstats.mquantiles(a)))
    stat = scipy.stats.describe(a)
    other = 'mean %.2f, var %.2f (min, max) (%.2f %.2f), nobs %.2f' % (stat.mean, stat.variance, stat.minmax[0], stat.minmax[1], stat.nobs)
    return quants + ' ' + other   

def score_df(df, output_dir):
    a = df.groupby('word').agg(
        {'context': 'count', 'gold_sense_id': 'nunique', 'predict_sense_id': 'nunique'}).reset_index()
    with open(os.path.join(output_dir,'metrics.stat'),'w') as log:
        print(describe(a.predict_sense_id), '#PREDICTED senses per word', file=log)
        df.groupby('word').agg({'predict_sense_id':lambda x: ' '.join('%s(%d)' % e for e in x.value_counts().sort_index().iteritems())}).to_csv(os.path.join(output_dir, 'predict_sense_ids.stat') ,encoding='utf-8')
        if df.gold_sense_id.isnull().any():
            print('All or som gold sense ids are not specified, cannot evaluate metrics depending on true clustering knowledge!', file=log)
        else:
            print(describe(a.gold_sense_id), '#GOLD senses per word', file=log)
            print(describe(a.predict_sense_id - a.gold_sense_id), '#PREDICTED-#GOLD senses per word', file=log)

        whcv = []
        for word in df.word.unique():
            df1 = df[df.word == word]
            h, c, v = metrics.homogeneity_completeness_v_measure(df1.gold_sense_id, df1.predict_sense_id)
            whcv.append((word, h, c, v))
        hcv = pd.DataFrame.from_records(whcv, columns=['word', 'homogeneity', 'completeness', 'v-measure'])
        a = a.merge(hcv, on='word')
        print(describe(a.homogeneity), 'homogeneity', file=log)
        print(describe(a.completeness), 'completeness', file=log)
        print(describe(a['v-measure']), 'v-measure', file=log)
        a.to_csv(os.path.join(output_dir, 'perword_hcv.stat'), index=False, encoding='utf-8')

def read_csv(filename):
    df = pd.read_csv(
        filename, sep='\t',
        encoding='utf8',
        dtype={
            'gold_sense_id': str,
            'predict_sense_id': str,
        })
    return df.set_index('context_id', drop=True)


if __name__ == '__main__':
    main()
