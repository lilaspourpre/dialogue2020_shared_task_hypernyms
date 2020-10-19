from collections import Counter, defaultdict

from gensim.models import KeyedVectors
from scipy import spatial

from predict_models import RankedModel
from ruwordnet.ruwordnet_reader import RuWordnet
from vectorizers.projection_vectorizer import ProjectionVectorizer
from operator import itemgetter


class ClassifierNode2VecRankedModel(RankedModel):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.node2vec = KeyedVectors.load_word2vec_format(params["node2vec_path"], binary=False)
        self.projection = ProjectionVectorizer(self.w2v_data, params["projection_path"])
        self.predicted = self.generate_predictions(params["predictions"])

    def generate_predictions(self, path):
        data = defaultdict(list)
        ruwordnet = RuWordnet(self.params["db_path"], self.params["ruwordnet_path"])

        with open(path, 'r', encoding='utf-8') as f:  # "./labelled_hch.tsv"
            for line in f:
                label, _, neologism, candidate_word = line.strip().split("\t")
                label = float(label)
                candidate = ruwordnet.get_id_by_name(candidate_word)
                if label == 1.0:
                    data[neologism].append(candidate)
        return data

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        hypernyms = self.compute_hchs(neologism, get_hypernym_fn, topn)
        second_order_hypernyms = [s_o for hypernym in hypernyms for s_o in get_hypernym_fn(hypernym)]

        node2vec, node2vec_vector = self.generate_node2vec(neologism, get_hypernym_fn, topn)
        all_hypernyms = Counter(hypernyms + second_order_hypernyms)

        sorted_hypernyms = reversed(sorted(all_hypernyms.items(), key=lambda x: self.get_node2vec_score(neologism,
                                                                                                        node2vec_vector,
                                                                                                        *x)))

        return [i[0] for i in sorted_hypernyms][:topn]

    def generate_node2vec(self, neologism, compute_hypernyms, topn=10) -> list:
        neighbours, node2vec_vector = self.projection.predict_projection_word(neologism, self.node2vec)
        associates = map(itemgetter(0), neighbours)
        hchs = [hypernym for associate in associates for hypernym in compute_hypernyms(associate)]
        return hchs, node2vec_vector

    def get_node2vec_score(self, neologism, node2vec_vector, candidate, count):
        nn_score = 0.5 if candidate in self.predicted[neologism] else 1
        return count * (self.get_similarity(neologism, candidate)) + \
               self.get_node2vec_similarity(node2vec_vector, candidate)

    def get_node2vec_similarity(self, v1, candidate):
        v2 = self.node2vec[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)

data = defaultdict(list)
ruwordnet = RuWordnet("../dataset/ruwordnet.db", None)

with open("./labelled_hch.tsv", 'r', encoding='utf-8') as f:
    for line in f:
        label, similarity, neologism, candidate_word = line.strip().split("\t")
        label = float(label)
        similarity = float(similarity)
        candidate = ruwordnet.get_id_by_name(candidate_word)
        if label == 1.0:
            data[neologism].append((candidate, similarity))

with open("predictions_classification_private_nouns.tsv", 'w', encoding='utf-8') as w:
    for i in data:
        candidates = reversed(sorted(set(data[i]), key=lambda x: x[1]))
        for candidate in candidates:
            w.write(f"{i}\t{candidate[0]}\t{ruwordnet.get_name_by_id(candidate[0])}\n")
