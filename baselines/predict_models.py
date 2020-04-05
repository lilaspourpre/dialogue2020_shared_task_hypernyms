from abc import abstractmethod, ABC
from collections import Counter
from gensim.models import KeyedVectors
from scipy import spatial
from operator import itemgetter

from ruwordnet.ruwordnet_reader import RuWordnet


class Model(ABC):
    def __init__(self, params):
        self.w2v_synsets = KeyedVectors.load_word2vec_format(params['synsets_vectors_path'], binary=False)
        self.w2v_data = KeyedVectors.load_word2vec_format(params['data_vectors_path'], binary=False)

    def predict_hypernyms(self, neologisms, get_hypernym_fn, topn=10):
        return {neologism: self.compute_candidates(neologism, get_hypernym_fn, topn) for neologism in neologisms}

    @abstractmethod
    def compute_candidates(self, neologisms, get_hypernym_fn, topn=10):
        pass


# ---------------------------------------------------------------------------------------------
# Baseline Model
# ---------------------------------------------------------------------------------------------

class BaselineModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def compute_candidates(self, neologism, get_hypernym_fn, topn=10) -> list:
        return list(map(itemgetter(0), self.generate_associates(neologism, topn)))

    def generate_associates(self, neologism, topn=10) -> list:
        return self.w2v_synsets.similar_by_vector(self.w2v_data[neologism], topn)


# ---------------------------------------------------------------------------------------------
# Hypernym of Co-Hypernyms Model
# ---------------------------------------------------------------------------------------------

class HCHModel(BaselineModel):
    def __init__(self, params):
        super().__init__(params)

    def compute_candidates(self, neologism, get_hypernym_fn, topn=10):
        return self.compute_hchs(neologism, get_hypernym_fn, topn=10)[:topn]

    def compute_hchs(self, neologism, compute_hypernyms, topn=10) -> list:
        associates = map(itemgetter(0), self.generate_associates(neologism, topn))
        hchs = [hypernym for associate in associates for hypernym in compute_hypernyms(associate)]
        return hchs


# ---------------------------------------------------------------------------------------------
# Ranked Model
# ---------------------------------------------------------------------------------------------

class RankedModel(HCHModel):
    def __init__(self, params):
        super().__init__(params)
        self.ruwordnet = RuWordnet(db_path=params["db_path"], ruwordnet_path=params["ruwordnet_path"])

    def compute_candidates(self, neologism, get_hypernym_fn, topn=10):
        hypernyms = self.compute_hchs(neologism, get_hypernym_fn, topn)
        second_order_hypernyms = [s_o for hypernym in hypernyms for s_o in get_hypernym_fn(hypernym)]

        all_hypernyms = Counter(hypernyms + second_order_hypernyms)
        sorted_hypernyms = reversed(sorted(all_hypernyms.items(), key=lambda x: self.__get_score(neologism, *x)))

        return [i[0] for i in sorted_hypernyms][:topn]

    def __get_score(self, neologism, candidate, count):
        return count * self.__get_similarity(neologism, candidate)

    def __get_similarity(self, neologism, candidate):
        v1 = self.w2v_data[neologism]
        v2 = self.w2v_synsets[candidate]
        return 1 - spatial.distance.cosine(v1, v2)
