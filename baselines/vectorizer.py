from abc import abstractmethod
import numpy as np


class Vectorizer:
    @abstractmethod
    def vectorize_ruwordnet(self, db_path, output_path):
        pass

    @abstractmethod
    def vectorize_data(self, data_path, output_path):
        pass

    @abstractmethod
    def __get_ruwordnet_vectors(self, ruwordnet):
        pass

    @abstractmethod
    def __get_data_vectors(self, data):
        pass

    def __save_as_w2v(self, words: list, vectors: np.array, output_path):
        assert len(words) == len(vectors)
        with open(output_path, 'w', encoding='utf-8') as w:
            w.write(f"{vectors.shape[0]} {vectors.shape[1]}\n")
            for word, vector in zip(words, vectors):
                vector_line = " ".join(map(str, vector))
                w.write(f"{word} {vector_line}\n")
