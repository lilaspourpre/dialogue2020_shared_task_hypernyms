import sys
import json
import codecs
from abc import abstractmethod

from ruwordnet.ruwordnet_reader import RuWordnet
from gensim.models import KeyedVectors


class Model:
    @abstractmethod
    def predict_hypernyms(self, neologisms, topn=10):
        pass

    @abstractmethod
    def __compute_hypernyms(self, neologisms, topn=10):
        pass


class BaselineModel(Model):
    def __init__(self, params):
        self.ruwordnet = RuWordnet(db_path=params["db_path"], ruwordnet_path=params["ruwordnet_path"])
        self.w2v_ruwordnet = KeyedVectors.load_word2vec_format(params['ruwordnet_vectors_path'], binary=False)
        self.w2v_data = KeyedVectors.load_word2vec_format(params['data_vectors_path'], binary=False)

    def predict_hypernyms(self, neologisms, topn=10) -> dict:
        return {neologism: self.__compute_hypernyms(neologism, topn) for neologism in neologisms}

    def __compute_hypernyms(self, neologism, topn=10) -> list:
        return [i[0] for i in self.w2v_ruwordnet.similar_by_vector(self.w2v_data[neologism], topn)]


class FastTextModel(Model):
    def __init__(self, params):
        self.ruwordnet = RuWordnet(db_path=params["db_path"], ruwordnet_path=params["ruwordnet_path"])
        self.w2v_ruwordnet = KeyedVectors.load_word2vec_format(params['ruwordnet_vectors_path'], binary=False)
        self.w2v_data = KeyedVectors.load_word2vec_format(params['data_vectors_path'], binary=False)

    def predict_hypernyms(self, neologisms, topn=10) -> dict:
        return {neologism: self.__compute_hypernyms(neologism, topn) for neologism in neologisms}

    def __compute_hypernyms(self, neologism, topn=10) -> list:
        hypernyms = []
        associates = [i[0] for i in self.w2v_ruwordnet.similar_by_vector(self.w2v_data[neologism], topn)]
        for associate in associates:
            print(associate, self.ruwordnet.get_name_by_id(associate))
            hypernyms.extend(self.ruwordnet.get_hypernyms_by_id(associate))
        return hypernyms[:10]


def load_config():
    if len(sys.argv) < 2:
        raise Exception("Please specify path to config file")
    with open(sys.argv[1], 'r', encoding='utf-8')as j:
        params = json.load(j)
    return params


def save_to_file(words_with_hypernyms, output_path, ruwordnet):
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        for word, hypernyms in words_with_hypernyms.items():
            for hypernym in hypernyms:
                f.write(f"{word}\t{hypernym}\t{ruwordnet.get_name_by_id(hypernym)}\n")


def main():
    params = load_config()
    with open(params['test_path'], 'r', encoding='utf-8') as f:
        test_data = f.read().split("\n")[:-1]
    baseline = FastTextModel(params)
    results = baseline.predict_hypernyms(list(test_data))
    save_to_file(results, params['output_path'], baseline.ruwordnet)


if __name__ == '__main__':
    main()
