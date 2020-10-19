import numpy as np
from numpy.linalg import norm


class BaseVectorizer():
    def __init__(self, embeddings_model):
        self.model = embeddings_model


    def get_norm_vec(self, vec):
        vec = vec / norm(vec)
        return vec

    def get_mean_vec(self, vecs, words):
        vec = np.sum(vecs, axis=0)
        vec = np.divide(vec, len(words))
        return vec

    def get_norm_mean_vec(self, vecs, words):
        vec = self.get_mean_vec(vecs, words)
        vec = self.get_norm_vec(vec)
        return vec


class ProjectionVectorizer(BaseVectorizer):
    """
    векторизация текста матрицей трансформации
    """
    def __init__(self, embeddings_path, projection_path):
        super(ProjectionVectorizer, self).__init__(embeddings_path)
        self.projection = np.loadtxt(projection_path, delimiter=',')

    def project_vec(self, src_vec):
        """
        :param src_vec: input vector to project
        :return:
        """
        test = np.mat(src_vec)
        test = np.c_[1.0, test]  # Adding bias term
        predicted_vec = np.dot(self.projection, test.T)
        predicted_vec = np.squeeze(np.asarray(predicted_vec))
        return predicted_vec

    def predict_projection_word(self, src_word, tar_emdedding_model, topn=10):
        """
        given the input word predict nearest_neighbours and the vector
        """
        src_vec = self.model[src_word]
        predicted_vec = self.project_vec(src_vec)
        nearest_neighbors = tar_emdedding_model.most_similar(positive=[predicted_vec], topn=topn)
        return nearest_neighbors, predicted_vec
