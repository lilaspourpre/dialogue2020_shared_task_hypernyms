import sys
import json
import codecs

from nltk.corpus import WordNetCorpusReader

from evaluate_nn_predictions import ClassifierNode2VecRankedModel
from ruwordnet.ruwordnet_reader import RuWordnet
from predict_models import BaselineModel, HCHModel, RankedModel, HyponymModel, RankedWikiModel, SemevalModel, LRModel
from predict_models import Node2vecEmbeddingsModel, Node2vecBaselineModel, Node2VecRankedModel, PoincareEmbeddingsModel
from predict_models import CombinedModel, Node2VecModel, AllModel
from semeval2016_task13.semeval_taxonomy import SemEvalTaxonomy


def load_config():
    if len(sys.argv) < 2:
        raise Exception("Please specify path to config file")
    with open(sys.argv[1], 'r', encoding='utf-8')as j:
        params = json.load(j)
    return params


def generate_taxonomy_fns(params, model):
    # for English WordNet
    if params['language'] == 'en':
        wn = WordNetCorpusReader(params["ruwordnet_path"], None)
        return lambda x: [hypernym.name() for hypernym in wn.synset(x).hypernyms()
                          if hypernym.name() in model.w2v_synsets.vocab], \
               lambda x: [hyponym.name() for hyponym in wn.synset(x).hyponyms() if hyponym.name()
                          in model.w2v_synsets.vocab], \
               lambda x: x.split(".")[0].replace("_", " ")
    # for RuWordNet
    elif params['language'] == 'ru':
        ruwordnet = RuWordnet(db_path=params["db_path"], ruwordnet_path=params["ruwordnet_path"])
        return lambda x: ruwordnet.get_hypernyms_by_id(x), lambda x: ruwordnet.get_hyponyms_by_id(x), \
               lambda x: ruwordnet.get_name_by_id(x)
    # for semeval
    elif params['task'] == 'semeval':
        taxonomy = SemEvalTaxonomy(taxonomy_path=params['taxonomy_path'], use_underscore=True)
        return lambda x: taxonomy.get_hypernym(x), lambda x: taxonomy.get_hyponym(x), lambda x: x
    else:
        raise Exception("task / language is not supported")


def save_to_file(words_with_hypernyms, output_path, params):
    # ruwordnet = RuWordnet(db_path=params["db_path"], ruwordnet_path=params["ruwordnet_path"])
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        for word, hypernyms in words_with_hypernyms.items():
            # word = word.replace("_", " ")  # uncomment for semeval
            for hypernym in hypernyms:
                # hypernym = hypernym.replace("_", " ") # uncomment for semeval
                f.write(f"{word}\t{hypernym}\n")  # \t{ruwordnet.get_name_by_id(hypernym)}\n")


def main():
    models = {"baseline": BaselineModel, "hch": HCHModel, "ranked": RankedModel, "hyponym": HyponymModel,
              "wiki": RankedWikiModel, 'semeval': SemevalModel, "lr": LRModel, "node2vec": Node2vecEmbeddingsModel,
              "node2vec_base": Node2vecBaselineModel, "node2vec_ranked": Node2VecRankedModel,
              "neural": ClassifierNode2VecRankedModel, "poincare": PoincareEmbeddingsModel,
              "combined": CombinedModel, "node2vec_proj": Node2VecModel, "all": AllModel}
    params = load_config()
    with open(params['test_path'], 'r', encoding='utf-8') as f:
        if params['task'] == 'semeval':
            test_data = [i.split("\t")[-1].replace(" ", "_") for i in f.read().split("\n") if i]
        else:
            test_data = f.read().split("\n")[:-1]
    model = models[params["model"]](params)
    print("Model loaded")

    topn = 10 if "topn" not in params else params["topn"]
    results = model.predict_hypernyms(list(test_data), *generate_taxonomy_fns(params, model), topn)
    save_to_file(results, params['output_path'], params)


if __name__ == '__main__':
    main()
