import numpy as np
from model import Model


class BertBaseline(Model):
    def __init__(self, params):
        self.news_corpus = read_news_corpus(params['news_corpus'])
        self.bert_model = BertEncoder(params['bert_path'])

    def predict_hypernyms(self, neologisms) -> dict:
        words_with_hypernyms = {}
        for neologism in neologisms:
            neologism_context_vector = self._get_context_vector(neologism)

        return words_with_hypernyms

    def _get_context_vector(self, word):
        context_sentences, indices = self._get_contexts_with_indices(word)
        embedded_sentences = self.bert_model.encode(context_sentences)
        neologism_vectors = [sentence[index] for sentence, index in zip(embedded_sentences, indices)]
        return np.mean(neologism_vectors, axis=0)

    def _get_contexts_with_indices(self, word) -> dict:
        pass
