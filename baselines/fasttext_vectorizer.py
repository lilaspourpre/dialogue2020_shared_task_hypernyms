import numpy as np
from gensim.models.fasttext import load_facebook_model
from string import punctuation


class FasttextVectorizer:
    def __init__(self, model_path):
        self.model = load_facebook_model(model_path)

    # -------------------------------------------------------------
    # vectorize ruwordnet
    # -------------------------------------------------------------

    def vectorize_ruwordnet(self, synsets, output_path):
        rwn_vectors = self.__get_ruwordnet_vectors(synsets)
        self.save_as_w2v([i[0] for i in synsets], rwn_vectors, output_path)

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
        self.save_as_w2v(data, data_vectors, output_path)

    def __get_data_vectors(self, data):
        vectors = np.zeros((len(data), self.model.vector_size))
        for i, word in enumerate(data):  # TODO: how to do it more effective or one-line
            vectors[i, :] = self.model[word]
        return vectors

    def save_as_w2v(self, words: list, vectors: np.array, output_path):
        assert len(words) == len(vectors)
        with open(output_path, 'w', encoding='utf-8') as w:
            w.write(f"{vectors.shape[0]} {vectors.shape[1]}\n")
            for word, vector in zip(words, vectors):
                vector_line = " ".join(map(str, vector))
                w.write(f"{word} {vector_line}\n")


if __name__ == '__main__':
    ft_vec = FasttextVectorizer("models/cc.ru.300.bin")
    with open("../dataset/public/verbs_public_no_labels.tsv", 'r', encoding='utf-8') as f:
        dataset = f.read().split("\n")[:-1]
    ft_vec.vectorize_data(dataset, "models/vectors/verbs_public_fasttext.txt")
