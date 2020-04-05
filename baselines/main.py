import sys
import json
import codecs

from nltk.corpus import WordNetCorpusReader

from ruwordnet.ruwordnet_reader import RuWordnet
from predict_models import BaselineModel, HCHModel, RankedModel


def load_config():
    if len(sys.argv) < 2:
        raise Exception("Please specify path to config file")
    with open(sys.argv[1], 'r', encoding='utf-8')as j:
        params = json.load(j)
    return params


def generate_hypernym_fn(params, model):
    # for English WordNet
    if params['language'] == 'en':
        wn = WordNetCorpusReader(params["ruwordnet_path"], None)
        return lambda x: [hypernym.name() for hypernym in wn.synset(x).hypernyms()
                          if hypernym.name() in model.w2v_synsets.vocab]
    # for RuWordNet
    elif params['language'] == 'ru':
        ruwordnet = RuWordnet(db_path=params["db_path"], ruwordnet_path=params["ruwordnet_path"])
        return lambda x: ruwordnet.get_hypernyms_by_id(x)
    else:
        raise Exception("language is not supported")


def save_to_file(words_with_hypernyms, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        for word, hypernyms in words_with_hypernyms.items():
            for hypernym in hypernyms:
                f.write(f"{word}\t{hypernym}\n")


def main():
    models = {"baseline": BaselineModel, "hch": HCHModel, "ranked": RankedModel}
    params = load_config()
    with open(params['test_path'], 'r', encoding='utf-8') as f:
        test_data = f.read().split("\n")[:-1]
    model = models[params["model"]](params)
    print("Model loaded")

    results = model.predict_hypernyms(list(test_data), generate_hypernym_fn(params, model))
    save_to_file(results, params['output_path'])


if __name__ == '__main__':
    main()
