import sys
import json

from utils import *
from bert_baseline import BertBaseline
from fasttext_baseline import FasttextBaseline


def load_config():
    if len(sys.argv) < 2:
        raise Exception("Please specify path to config file")
    return load_json(sys.argv[1])


def load_json(jsonpath):
    with open(jsonpath, 'r', encoding='utf-8')as j:
        params = json.load(j)
    return params


def main():
    params = load_config()
    baselines = [BertBaseline(params), FasttextBaseline()]
    test_data = read_dataset(params['test_path'])
    for baseline in baselines:
        results = baseline.predict_hypernyms(list(test_data))
        save_to_file(results, params['output_path'])


if __name__ == '__main__':
    main()
