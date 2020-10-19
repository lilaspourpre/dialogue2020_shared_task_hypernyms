import json
import math
import re
from abc import abstractmethod, ABC
from collections import Counter, defaultdict
from operator import itemgetter

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.poincare import PoincareKeyedVectors
from scipy import spatial

from vectorizers.projection_vectorizer import ProjectionVectorizer


class Model(ABC):
    def __init__(self, params):
        self.w2v_synsets = KeyedVectors.load_word2vec_format(params['synsets_vectors_path'], binary=False)
        self.w2v_data = KeyedVectors.load_word2vec_format(params['data_vectors_path'], binary=False)

    def predict_hypernyms(self, neologisms, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        return {neologism: self.compute_candidates(neologism, get_hypernym_fn,
                                                   get_hyponym_fn, get_taxonomy_name_fn,
                                                   topn) for neologism in neologisms}

    def get_score(self, neologism, candidate, count):
        return count * self.get_similarity(neologism, candidate)

    def get_similarity(self, neologism, candidate):
        v1 = self.w2v_data[neologism]
        v2 = self.w2v_synsets[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)

    @abstractmethod
    def compute_candidates(self, neologisms, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        pass


# ---------------------------------------------------------------------------------------------
# Baseline Model
# ---------------------------------------------------------------------------------------------

class BaselineModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10) -> list:
        return list(map(itemgetter(0), self.generate_associates(neologism, topn)))

    def generate_associates(self, neologism, topn=10) -> list:
        return self.w2v_synsets.similar_by_vector(self.w2v_data[neologism], topn)


# ---------------------------------------------------------------------------------------------
# Hypernym of Co-Hypernyms Model
# ---------------------------------------------------------------------------------------------

class HCHModel(BaselineModel):
    def __init__(self, params):
        super().__init__(params)

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
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

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        hypernyms = self.compute_hchs(neologism, get_hypernym_fn, topn)
        second_order_hypernyms = [s_o for hypernym in hypernyms for s_o in get_hypernym_fn(hypernym)]

        all_hypernyms = Counter(hypernyms + second_order_hypernyms)
        sorted_hypernyms = reversed(sorted(all_hypernyms.items(), key=lambda x: self.get_score(neologism, *x)))

        return [i[0] for i in sorted_hypernyms][:topn]


# ---------------------------------------------------------------------------------------------
# Hyponym Model
# ---------------------------------------------------------------------------------------------


def distance2vote(d, a=3.0, b=5.0, y=1.0):
    sim = np.maximum(0, 1 - d ** 2 / 2)
    return np.exp(-d ** a) * y * sim ** b


def compute_distance(s):
    return np.sqrt(2 * (1 - s))


class HyponymModel(HCHModel):
    def __init__(self, params):
        super().__init__(params)

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        associates = self.generate_associates(neologism, 100)
        votes = Counter()
        for associate, similarity in associates:
            distance = compute_distance(similarity)
            for hypernym in get_hypernym_fn(associate):
                votes[hypernym] += distance2vote(distance)
                for second_order in get_hypernym_fn(hypernym):
                    votes[second_order] += distance2vote(distance, y=0.5)
        return list(map(itemgetter(0), votes.most_common(topn)))


# ---------------------------------------------------------------------------------------------
# Semeval Model
# ---------------------------------------------------------------------------------------------


class SemevalModel(HCHModel):
    def __init__(self, params):
        super().__init__(params)
        self.wiktionary = self.__get_wiktionary(params['wiki_path'])
        self.wiki_model = KeyedVectors.load_word2vec_format(params['wiki_vectors_path'], binary=False)
        self.delete_bracets = re.compile(r"\(.+?\)")
        if params['language'] == 'ru':
            self.pattern = re.compile("[^А-я \-]")
        else:
            self.pattern = re.compile("[^A-z \-]")

    def __get_wiktionary(self, path):
        wiktionary = {}
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                wiktionary[data['word']] = {"hypernyms": data['hypernyms'], "synonyms": data['synonyms'],
                                            "meanings": data['meanings']}
        return wiktionary

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        hypernyms = self.compute_hchs(neologism, get_hypernym_fn, topn)
        all_hypernyms = Counter(hypernyms)
        associates = self.generate_associates(neologism, 50)
        votes = Counter()
        for associate, similarity in associates:
            for hypernym in get_hypernym_fn(associate):
                votes[hypernym] += similarity
        sorted_hypernyms = reversed(sorted((all_hypernyms + votes).items(),
                                           key=lambda x: self.get_wiki_score(neologism, get_taxonomy_name_fn, *x)
                                           ))
        return [i[0] for i in sorted_hypernyms][:topn]

    def get_wiki_score(self, neologism, get_taxonomy_fn, candidate, count):
        wiki_count = 0.3
        definition_count = 0.8
        synonym_count = 1
        wiki_similarity = 1
        if neologism.lower() in self.wiktionary:
            wiktionary_data = self.wiktionary[neologism.lower()]
            candidate_words = self.delete_bracets.sub("", get_taxonomy_fn(candidate)).split(',')

            if any([candidate_word.lower() in wiktionary_data['hypernyms'] for candidate_word in candidate_words]):
                wiki_count = 2

            if any([any([candidate_word.lower() in i for candidate_word in candidate_words])
                    for i in wiktionary_data['meanings']]):
                definition_count = 2

            if any([candidate_word.lower() in wiktionary_data['synonyms'] for candidate_word in candidate_words]):
                synonym_count = 2

            wiki_similarities = []
            # for wiki_hypernym in wiktionary_data['hypernyms']:
            #     wiki_hypernym = wiki_hypernym.replace("|", " ").replace('--', '')
            #     wiki_hypernym = self.pattern.sub("", wiki_hypernym)
            #     if not all([i == " " for i in wiki_hypernym]):
            #         wiki_similarities.append(self.compute_similarity(wiki_hypernym.replace(" ", "_"), candidate))
            if wiki_similarities:
                wiki_similarity = sum(wiki_similarities) / len(wiki_similarities)

        return synonym_count * 2 + definition_count * 0.8 + wiki_count * 0.5 + 0.6 * count * self.get_similarity(
            neologism,
            candidate) + 2 * wiki_similarity
        # return synonym_count * 0.5 + definition_count * 0.4 + wiki_count * 0.3 + 0.6 * count * self.get_similarity(neologism, candidate) + 2*wiki_similarity

    def compute_similarity(self, neologism, candidate):
        v1 = self.wiki_model[neologism]
        v2 = self.w2v_synsets[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)


# ---------------------------------------------------------------------------------------------
# Wiki Model
# ---------------------------------------------------------------------------------------------


class RankedWikiModel(HCHModel):
    def __init__(self, params):
        super().__init__(params)
        self.wiktionary = self.__get_wiktionary(params['wiki_path'])
        self.wiki_model = KeyedVectors.load_word2vec_format(params['wiki_vectors_path'], binary=False)
        self.node2vec = KeyedVectors.load_word2vec_format(params["node2vec_path"], binary=False)
        self.n = params['n']
        self.projection = ProjectionVectorizer(self.w2v_data, params["projection_path"])

        self.delete_bracets = re.compile(r"\(.+?\)")
        if params['language'] == 'ru':
            self.pattern = re.compile("[^А-я \-]")
        else:
            self.pattern = re.compile("[^A-z \-]")

    def __get_wiktionary(self, path):
        wiktionary = {}
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                wiktionary[data['word']] = {"hypernyms": data['hypernyms'], "synonyms": data['synonyms'],
                                            "meanings": data['meanings']}
        return wiktionary

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        hypernyms = self.compute_hchs(neologism, get_hypernym_fn, topn)
        second_order_hypernyms = [s_o for hypernym in hypernyms for s_o in get_hypernym_fn(hypernym)]
        all_hypernyms = Counter(hypernyms + second_order_hypernyms)
        associates = self.generate_associates(neologism, 50)

        node2vec, mean_node2vec = self.generate_node2vec(neologism, get_hypernym_fn, topn)

        votes = Counter()
        for associate, similarity in associates:
            for hypernym in get_hypernym_fn(associate):
                votes[hypernym] += similarity
        sorted_hypernyms = reversed(sorted((all_hypernyms + votes).items(),
                                           key=lambda x: self.get_wiki_score(neologism, get_taxonomy_name_fn, mean_node2vec,
                                                                             *x)
                                           ))
        return [i[0] for i in sorted_hypernyms][:topn]

    def get_wiki_score(self, neologism, get_taxonomy_fn, node2vec_vector, candidate, count):
        wiki_count = 0.3
        definition_count = 0.8
        synonym_count = 1
        wiki_similarity = 1
        node2vec_count = 1

        if neologism.lower() in self.wiktionary:
            wiktionary_data = self.wiktionary[neologism.lower()]
            candidate_words = self.delete_bracets.sub("", get_taxonomy_fn(candidate)).split(',')

            if any([candidate_word.lower() in wiktionary_data['hypernyms'] for candidate_word in candidate_words]):
                wiki_count = 2

            if any([any([candidate_word.lower() in i for candidate_word in candidate_words])
                    for i in wiktionary_data['meanings']]):
                definition_count = 2

            if any([candidate_word.lower() in wiktionary_data['synonyms'] for candidate_word in candidate_words]):
                synonym_count = 2

            wiki_similarities = []
            for wiki_hypernym in wiktionary_data['hypernyms']:
                wiki_hypernym = wiki_hypernym.replace("|", " ").replace('--', '')
                wiki_hypernym = self.pattern.sub("", wiki_hypernym)
                if not all([i == " " for i in wiki_hypernym]):
                    wiki_similarities.append(self.compute_similarity(wiki_hypernym.replace(" ", "_"), candidate))
            if wiki_similarities:
                wiki_similarity = sum(wiki_similarities) / len(wiki_similarities)

        node2vec_similarity = self.get_node2vec_similarity(node2vec_vector, candidate)

        # return synonym_count * 0.5 + definition_count * 0.8 + wiki_count * 0.5 + \
        #        0.6 * count * self.get_similarity(neologism, candidate) + 2 * wiki_similarity + 2 * node2vec_similarity
        return synonym_count * 2 + definition_count * 0.4 + wiki_count * 0.3 + 0.6 * count * self.get_similarity(
        neologism, candidate) + 2 * wiki_similarity + 2 * node2vec_similarity

    def compute_similarity(self, neologism, candidate):
        v1 = self.wiki_model[neologism]
        v2 = self.w2v_synsets[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)

    def get_node2vec_similarity(self, v1, candidate):
        v2 = self.node2vec[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)

    def get_node2vec(self, neologism, topn=10) -> list:
        neighbours, _ = self.projection.predict_projection_word(neologism, self.node2vec, topn=topn)
        return neighbours

    def generate_node2vec(self, neologism, compute_hypernyms, topn=10) -> list:
        associates = map(itemgetter(0), self.get_node2vec(neologism, topn))
        hchs = [hypernym for associate in associates for hypernym in compute_hypernyms(associate) if associate in self.w2v_synsets]
        _, node2vec_vector = self.projection.predict_projection_word(neologism, self.node2vec)
        return hchs, node2vec_vector


# ---------------------------------------------------------------------------------------------
# LR Model
# ---------------------------------------------------------------------------------------------


class LRModel(HCHModel):
    def __init__(self, params):
        super().__init__(params)
        self.wiktionary = self.__get_wiktionary(params['wiki_path'])
        self.wiki_model = KeyedVectors.load_word2vec_format(params['wiki_vectors_path'], binary=False)
        self.node2vec = KeyedVectors.load_word2vec_format(params["node2vec_path"], binary=False)
        # self.projection = ProjectionVectorizer(self.w2v_data, params["projection_path"])

        self.delete_bracets = re.compile(r"\(.+?\)")
        if params['language'] == 'ru':
            self.pattern = re.compile("[^А-я \-]")
        else:
            self.pattern = re.compile("[^A-z \-]")

    def __get_wiktionary(self, path):
        wiktionary = {}
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                wiktionary[data['word']] = {"hypernyms": data['hypernyms'], "synonyms": data['synonyms'],
                                            "meanings": data['meanings']}
        return wiktionary

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        hypernyms = self.compute_hchs(neologism, get_hypernym_fn, topn)
        second_order_hypernyms = [s_o for hypernym in hypernyms for s_o in get_hypernym_fn(hypernym)]
        all_hypernyms = Counter(hypernyms + second_order_hypernyms)
        associates = self.generate_associates(neologism, 100)

        votes = Counter()
        for associate, similarity in associates:
            distance = compute_distance(similarity)
            for hypernym in get_hypernym_fn(associate):
                votes[hypernym] += distance2vote(distance)
                for second_order in get_hypernym_fn(hypernym):
                    votes[second_order] += distance2vote(distance, y=0.5)

        sorted_hypernyms = Counter()
        for candidate in all_hypernyms + votes:
            count = all_hypernyms.get(candidate, 1)
            similarity, wiki_similarity, in_synonyms, \
            in_hypernyms, in_definition, not_in_hypernyms, \
            not_in_synonyms, not_in_definition, not_wiki_similarity = self.compute_weights(neologism, candidate,
                                                                                           get_taxonomy_name_fn)

            # score = 1.8554857 * similarity * count + wiki_similarity * 0.54429005 + in_synonyms * -5.42561308 +\
            #         in_hypernyms * 3.68804553 + in_definition * 11.0702077

            hyponym_count = votes.get(candidate, 0.0)
            if hyponym_count == 0.0:
                not_hyponym_count = 1.0
            else:
                not_hyponym_count = 0.0

            # _, node2vec_vector = self.projection.predict_projection_word(neologism, self.node2vec)
            # node2vec_similarity = self.get_node2vec_similarity(node2vec_vector, candidate)

            score = count * similarity * 3.34506023 + wiki_similarity * 6.82626317 + not_wiki_similarity * 1.87866841 + \
                    in_synonyms * 2.01482447 + not_in_synonyms * -0.36626615 + in_hypernyms * 1.42307518 + \
                    not_in_hypernyms * 0.22548314 + in_definition * 2.06956417 + -0.42100585 * not_in_definition + \
                    hyponym_count * 13.73838828 + not_hyponym_count * 0.0

            # score = count * similarity * 2.82635361 + wiki_similarity * 5.84246245 + not_wiki_similarity * 1.65507643 + \
            #                     in_synonyms * 1.90194106 + not_in_synonyms * -0.26925806 + in_hypernyms * 1.61310472 + \
            #                     not_in_hypernyms * 0.01957828 + in_definition * 2.12732149 + -0.49463849 * not_in_definition + \
            #                     hyponym_count * 11.79374631 + not_hyponym_count * 0.0 + -1.65538476 * node2vec_similarity
            sorted_hypernyms[candidate] = score
        return [i[0] for i in sorted_hypernyms.most_common(topn)]


    def get_node2vec_similarity(self, v1, candidate):
        v2 = self.node2vec[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)

    def get_node2vec(self, neologism, topn=10) -> list:
        neighbours, _ = self.projection.predict_projection_word(neologism, self.node2vec, topn=topn)
        return neighbours

    def generate_node2vec(self, neologism, compute_hypernyms, topn=10) -> list:
        associates = map(itemgetter(0), self.get_node2vec(neologism, topn))
        hchs = [hypernym for associate in associates for hypernym in compute_hypernyms(associate)]
        return hchs

    def compute_weights(self, neologism, candidate, get_taxonomy_name_fn):
        similarity = self.get_similarity(neologism, candidate)
        wiki_similarity = 0.0
        not_wiki_similarity = 0.0
        in_synonyms = 0.0
        in_hypernyms = 0.0
        in_definition = 0.0
        not_in_synonyms = 0.0
        not_in_hypernyms = 0.0
        not_in_definition = 0.0

        candidate_words = self.delete_bracets.sub("", get_taxonomy_name_fn(candidate)).split(',')
        if neologism.lower() in self.wiktionary:
            wiktionary_data = self.wiktionary[neologism.lower()]

            if any([candidate_word.lower() in wiktionary_data['hypernyms'] for candidate_word in candidate_words]):
                in_hypernyms = 1.0
            else:
                not_in_hypernyms = 1.0

            if any([candidate_word.lower() in wiktionary_data['synonyms'] for candidate_word in candidate_words]):
                in_synonyms = 1.0
            else:
                not_in_synonyms = 1.0

            if any([any([candidate_word.lower() in i for candidate_word in candidate_words])
                    for i in wiktionary_data['meanings']]):
                in_definition = 1.0
            else:
                not_in_definition = 1.0

            wiki_similarities = []
            # for wiki_hypernym in wiktionary_data['hypernyms']:
            #     wiki_hypernym = wiki_hypernym.replace("|", " ").replace('--', '')
            #     wiki_hypernym = self.pattern.sub("", wiki_hypernym)
            #     if not all([i == " " for i in wiki_hypernym]):
            #         wiki_similarities.append(self.compute_similarity(wiki_hypernym.replace(" ", "_"), candidate))
            if wiki_similarities:
                wiki_similarity = sum(wiki_similarities) / len(wiki_similarities)
            else:
                not_wiki_similarity = 1.0
        else:
            not_wiki_similarity = 1.0

        return similarity, wiki_similarity, in_synonyms, in_hypernyms, in_definition, not_in_hypernyms, not_in_synonyms, not_in_definition, not_wiki_similarity

    def compute_similarity(self, neologism, candidate):
        v1 = self.wiki_model[neologism]
        v2 = self.w2v_synsets[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)

    def get_node2vec(self, neologism, topn=10) -> list:
        neighbours, _ = self.projection.predict_projection_word(neologism, self.node2vec, topn=topn)
        return neighbours

    def generate_node2vec(self, neologism, compute_hypernyms, topn=10) -> list:
        associates = map(itemgetter(0), self.get_node2vec(neologism, topn))
        hchs = [hypernym for associate in associates for hypernym in compute_hypernyms(associate)]
        return hchs


# ---------------------------------------------------------------------------------------------
# Node2vec Model
# ---------------------------------------------------------------------------------------------


class Node2vecBaselineModel(BaselineModel):
    def __init__(self, params):
        super().__init__(params)
        self.node2vec = KeyedVectors.load_word2vec_format(params["node2vec_path"], binary=False)
        self.projection = ProjectionVectorizer(self.w2v_data, params["projection_path"])

    def generate_associates(self, neologism, topn=10) -> list:
        neighbours, _ = self.projection.predict_projection_word(neologism, self.node2vec)
        return neighbours


class Node2VecRankedModel(RankedModel):
    def __init__(self, params):
        super().__init__(params)
        self.node2vec = KeyedVectors.load_word2vec_format(params["node2vec_path"], binary=False)
        self.projection = ProjectionVectorizer(self.w2v_data, params["projection_path"])

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        node2vec, node2vec_vector = self.generate_node2vec(neologism, get_hypernym_fn, topn)
        second_order_hypernyms = [s_o for hypernym in node2vec for s_o in get_hypernym_fn(hypernym)]
        all_hypernyms = Counter(node2vec + second_order_hypernyms)

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
        return count * (self.get_similarity(neologism, candidate)) #+ self.get_node2vec_similarity(node2vec_vector,
                                                                    #                              candidate))

    def get_node2vec_similarity(self, v1, candidate):
        v2 = self.node2vec[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)


class Node2VecModel(RankedModel):
    def __init__(self, params):
        super().__init__(params)
        self.node2vec_wordnet = KeyedVectors.load_word2vec_format(params["node2vec_path"], binary=False)
        self.node2vec = KeyedVectors.load_word2vec_format(params["projection_path"], binary=False)

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        hypernyms = self.compute_hchs(neologism, get_hypernym_fn, topn)
        second_order_hypernyms = [s_o for hypernym in hypernyms for s_o in get_hypernym_fn(hypernym)]

        node2vec = self.compute_node2vec_candidates(neologism, get_hypernym_fn, topn)
        second_order = [s_o for hypernym in node2vec for s_o in get_hypernym_fn(hypernym)]

        ft_hypernyms = Counter(hypernyms + second_order_hypernyms)
        n2v_hypernyms = Counter(node2vec + second_order)


        final_candidates = defaultdict(float)

        for candidate, count in ft_hypernyms.items():
            final_candidates[candidate] += count * self.get_similarity(neologism, candidate)

        for candidate, count in n2v_hypernyms.items():
            final_candidates[candidate] += count * self.get_node2vec_similarity(neologism, candidate)

        return [i[0] for i in reversed(sorted(final_candidates.items(), key=lambda x: x[1]))][:topn]

    def compute_node2vec_candidates(self, neologism, compute_hypernyms, topn=10) -> list:
        neighbours = self.node2vec_wordnet.similar_by_vector(self.node2vec[neologism], topn)
        associates = map(itemgetter(0), neighbours)
        hchs = [hypernym for associate in associates for hypernym in compute_hypernyms(associate)]
        return hchs

    def get_node2vec_score(self, neologism, candidate, count):
        return count * (self.get_similarity(neologism, candidate) + self.get_node2vec_similarity(neologism, candidate))

    def get_node2vec_similarity(self, neologism, candidate):
        v1 = self.node2vec[neologism]
        v2 = self.node2vec_wordnet[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)


class PoincareEmbeddingsModel(BaselineModel):
    def __init__(self, params):
        super().__init__(params)
        self.poincare_model = PoincareKeyedVectors.load_word2vec_format(params["poincare_path"], binary=False)
        self.n = params["n"]

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn,
                           get_taxonomy_name_fn, topn=10) -> list:
        similars = [i[0] for i in self.w2v_synsets.similar_by_vector(self.w2v_data[neologism])]
        mean_poincare = self.aggregate(similars)
        candidates = [i[0] for i in self.poincare_model.most_similar(mean_poincare)]
        hchs = self.compute_hchs(candidates, get_hypernym_fn)
        return [i[0] for i in Counter(candidates+hchs).most_common(10)]


    def get_similarity(self, neologism, candidate):
        distance = self.poincare_model.distances(neologism, [candidate])[0]
        similarity = 1 / (1 + distance)
        return similarity

    def compute_hchs(self, associates, compute_hypernyms, topn=10) -> list:
        hchs = [hypernym for associate in associates for hypernym in compute_hypernyms(associate)]
        return hchs

    def aggregate(self, synsets):
        synsets = synsets[:self.n]
        gammas = [(1 / math.sqrt(1 - np.linalg.norm(self.poincare_model[i]) ** 2),
                   self.poincare_model[i]) for i in synsets if i in self.poincare_model.vocab]
        sum_v = sum([i[0] for i in gammas])
        return sum([(i[0] / sum_v) * i[1] for i in gammas])


class Node2vecEmbeddingsModel(BaselineModel):
    def __init__(self, params):
        super().__init__(params)
        self.node2vec = KeyedVectors.load_word2vec_format(params["node2vec_path"], binary=False)
        self.n = params["n"]

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn,
                           get_taxonomy_name_fn, topn=10) -> list:
        similars = [i[0] for i in self.w2v_synsets.similar_by_vector(self.w2v_data[neologism])]
        mean_node2vec = np.mean([self.node2vec[i] for i in similars[:self.n] if i in self.node2vec.vocab], 0)
        candidates = [i[0] for i in self.node2vec.similar_by_vector(mean_node2vec)]
        candidates = [i for i in candidates if i in self.w2v_synsets.vocab]
        hchs = self.compute_hchs(candidates, get_hypernym_fn)
        return [i[0] for i in Counter(candidates+hchs).most_common(10)]

    def compute_hchs(self, associates, compute_hypernyms, topn=10) -> list:
        hchs = [hypernym for associate in associates for hypernym in compute_hypernyms(associate)]
        return hchs


class CombinedModel(BaselineModel):
    def __init__(self, params):
        super().__init__(params)
        self.node2vec = Node2vecEmbeddingsModel(params)
        self.poincare = PoincareEmbeddingsModel(params)
        self.fasttext = RankedModel(params)

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn,
                           get_taxonomy_name_fn, topn=10) -> list:
        fasttext_candidates = self.fasttext.compute_candidates(neologism, get_hypernym_fn, get_hyponym_fn,
                           get_taxonomy_name_fn, 20)
        poincare_candidates = self.poincare.compute_candidates(neologism, get_hypernym_fn, get_hyponym_fn,
                           get_taxonomy_name_fn, 20)
        node2vec_candidates = self.node2vec.compute_candidates(neologism, get_hypernym_fn, get_hyponym_fn,
                                                               get_taxonomy_name_fn, 20)
        all_candidates = Counter(fasttext_candidates + poincare_candidates + node2vec_candidates)
        return [i[0] for i in all_candidates.most_common(len(all_candidates))]


# ---------------------------------------------------------------------------------------------
# Wiki Model
# ---------------------------------------------------------------------------------------------


class AllModel(HCHModel):
    def __init__(self, params):
        super().__init__(params)
        self.wiktionary = self.__get_wiktionary(params['wiki_path'])
        self.wiki_model = KeyedVectors.load_word2vec_format(params['wiki_vectors_path'], binary=False)
        self.node2vec = KeyedVectors.load_word2vec_format(params["node2vec_path"], binary=False)
        self.n = params['n']
        self.projection = ProjectionVectorizer(self.w2v_data, params["projection_path"])
        self.poincare_model = PoincareKeyedVectors.load_word2vec_format(params["poincare_path"], binary=False)
        self.n = params["n"]

        self.delete_bracets = re.compile(r"\(.+?\)")
        if params['language'] == 'ru':
            self.pattern = re.compile("[^А-я \-]")
        else:
            self.pattern = re.compile("[^A-z \-]")

    def __get_wiktionary(self, path):
        wiktionary = {}
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                wiktionary[data['word']] = {"hypernyms": data['hypernyms'], "synonyms": data['synonyms'],
                                            "meanings": data['meanings']}
        return wiktionary

    def compute_candidates(self, neologism, get_hypernym_fn, get_hyponym_fn, get_taxonomy_name_fn, topn=10):
        hypernyms = self.compute_hchs(neologism, get_hypernym_fn, topn)
        second_order_hypernyms = [s_o for hypernym in hypernyms for s_o in get_hypernym_fn(hypernym)]
        all_hypernyms = Counter(hypernyms + second_order_hypernyms)
        associates = self.generate_associates(neologism, 50)

        node2vec, mean_node2vec = self.generate_node2vec(neologism, get_hypernym_fn, topn)
        similars = [i[0] for i in self.w2v_synsets.similar_by_vector(self.w2v_data[neologism])]
        poincare_vector = self.aggregate(similars)

        votes = Counter()
        for associate, similarity in associates:
            for hypernym in get_hypernym_fn(associate):
                votes[hypernym] += similarity
        sorted_hypernyms = reversed(sorted((all_hypernyms + votes).items(),
                                           key=lambda x: self.get_wiki_score(neologism, get_taxonomy_name_fn,
                                                                             mean_node2vec, poincare_vector,
                                                                             *x)
                                           ))
        return [i[0] for i in sorted_hypernyms][:topn]

    def get_wiki_score(self, neologism, get_taxonomy_fn, node2vec_vector, poincare_vector, candidate, count):
        wiki_count = 0.3
        definition_count = 0.8
        synonym_count = 1
        wiki_similarity = 1

        if neologism.lower() in self.wiktionary:
            wiktionary_data = self.wiktionary[neologism.lower()]
            candidate_words = self.delete_bracets.sub("", get_taxonomy_fn(candidate)).split(',')

            if any([candidate_word.lower() in wiktionary_data['hypernyms'] for candidate_word in candidate_words]):
                wiki_count = 2

            if any([any([candidate_word.lower() in i for candidate_word in candidate_words])
                    for i in wiktionary_data['meanings']]):
                definition_count = 2

            if any([candidate_word.lower() in wiktionary_data['synonyms'] for candidate_word in candidate_words]):
                synonym_count = 2

            wiki_similarities = []
            for wiki_hypernym in wiktionary_data['hypernyms']:
                wiki_hypernym = wiki_hypernym.replace("|", " ").replace('--', '')
                wiki_hypernym = self.pattern.sub("", wiki_hypernym)
                if not all([i == " " for i in wiki_hypernym]):
                    wiki_similarities.append(self.compute_similarity(wiki_hypernym.replace(" ", "_"), candidate))
            if wiki_similarities:
                wiki_similarity = sum(wiki_similarities) / len(wiki_similarities)

        node2vec_similarity = self.get_node2vec_similarity(node2vec_vector, candidate)
        poincare_similarity = self.get_poincare_similarity(poincare_vector, candidate)

        # return synonym_count * 0.5 + definition_count * 0.8 + wiki_count * 0.5 + \
        #        0.6 * count * self.get_similarity(neologism, candidate) + 2 * wiki_similarity + 2 * node2vec_similarity
        return synonym_count * 2 + definition_count * 0.4 + wiki_count * 0.3 + 0.6 * count * self.get_similarity(
        neologism, candidate) + 2 * wiki_similarity + 2 * node2vec_similarity + 2 * poincare_similarity

    def compute_similarity(self, neologism, candidate):
        v1 = self.wiki_model[neologism]
        v2 = self.w2v_synsets[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)

    def get_node2vec_similarity(self, v1, candidate):
        v2 = self.node2vec[candidate]
        v1 = v1 / (sum(v1 ** 2) ** 0.5)
        v2 = v2 / (sum(v2 ** 2) ** 0.5)
        return 1 - spatial.distance.cosine(v1, v2)

    def get_node2vec(self, neologism, topn=10) -> list:
        neighbours, _ = self.projection.predict_projection_word(neologism, self.node2vec, topn=topn)
        return neighbours

    def generate_node2vec(self, neologism, compute_hypernyms, topn=10) -> list:
        associates = map(itemgetter(0), self.get_node2vec(neologism, topn))
        hchs = [hypernym for associate in associates for hypernym in compute_hypernyms(associate) if associate in self.w2v_synsets]
        _, node2vec_vector = self.projection.predict_projection_word(neologism, self.node2vec)
        return hchs, node2vec_vector

    def get_poincare_similarity(self, neologism, candidate):
        distance = self.poincare_model.distances(neologism, [candidate])[0]
        similarity = 1 / (1 + distance)
        return similarity

    def aggregate(self, synsets):
        synsets = synsets[:self.n]
        gammas = [(1 / math.sqrt(1 - np.linalg.norm(self.poincare_model[i]) ** 2),
                   self.poincare_model[i]) for i in synsets if i in self.poincare_model.vocab]
        sum_v = sum([i[0] for i in gammas])
        return sum([(i[0] / sum_v) * i[1] for i in gammas])
