import numpy as np
from gensim.models.fasttext import load_facebook_model
from string import punctuation
from ruwordnet.ruwordnet_reader import RuWordnet
from vectorizer import Vectorizer


class FasttextVectorizer(Vectorizer):
    def __init__(self, model_path):
        super().__init__()
        self.model = load_facebook_model(model_path)

    # -------------------------------------------------------------
    # vectorize ruwordnet
    # -------------------------------------------------------------

    def vectorize_ruwordnet(self, synsets, output_path):
        rwn_vectors = self.__get_ruwordnet_vectors(synsets)
        self.__save_as_w2v([i[0] for i in synsets], rwn_vectors, output_path)

    def __get_ruwordnet_vectors(self, synsets):
        vectors = np.zeros((len(synsets), self.model.vector_size))
        for i, (_id, text) in enumerate(synsets):
            vectors[i, :] = self.__get_avg_vector(text)
        return vectors

    def __get_avg_vector(self, text):
        words = [i.strip(punctuation) for i in text.split()]
        return np.sum(self.__get_data_vectors(words), axis=0)/len(words)

    # -------------------------------------------------------------
    # vectorize data
    # -------------------------------------------------------------

    def vectorize_data(self, data, output_path):
        data_vectors = self.__get_data_vectors(data)
        self.__save_as_w2v(data, data_vectors, output_path)

    def __get_data_vectors(self, data):
        vectors = np.zeros((len(data), self.model.vector_size))
        for i, word in enumerate(data):  # TODO: how to do it more effective or one-line
            vectors[i, :] = self.model[word]
        return vectors


if __name__ == '__main__':
    ft_vec = FasttextVectorizer("models/cc.ru.300.bin")
    ruwordnet = RuWordnet(db_path="../dataset/ruwordnet.db", ruwordnet_path=None)

    v_synsets = ruwordnet.get_all_synsets(endswith="V")
    ft_vec.vectorize_ruwordnet(v_synsets, "models/vectors/ruwordnet_verbs_fasttext_single.txt")

    n_synsets = ruwordnet.get_all_synsets(endswith="N")
    ft_vec.vectorize_ruwordnet(n_synsets, "models/vectors/ruwordnet_nouns_fasttext_single.txt")

