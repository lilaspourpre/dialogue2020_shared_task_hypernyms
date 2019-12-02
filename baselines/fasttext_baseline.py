from gensim.models.fasttext import load_facebook_model
from model import Model
from ruwordnet.ruwordnet_reader import RuWordnet
from utils import save_to_file, read_dataset


class FasttextBaseline(Model):
    def __init__(self, model_path, wordnet_path, db_path):
        self.model = load_facebook_model(model_path)
        self.wordnet = RuWordnet(wordnet_path, db_path)
        self.synset_names = self.wordnet.get_synset_names()

    def predict_neighbour_hypernyms(self, neologisms):
        return {neologism: self.compute_neighbour_hypernyms(neologism) for neologism in neologisms}

    def predict_parent_hypernyms(self, neologisms):
        return {neologism: self.compute_parent_hypernyms(neologism) for neologism in neologisms}

    def compute_parent_hypernyms(self, neologism):
        similars = self.model.similar_by_word(neologism,  100)
        most_similar = next((x[0] for x in similars if x[0].upper() in self.synset_names), '')
        if most_similar is None:
            print(neologism)
        return self.wordnet.get_hypernyms_by_name(most_similar)

    def compute_neighbour_hypernyms(self, neologism):
        similars = self.model.similar_by_word(neologism,  100)
        most_similar = [self.wordnet.get_id_by_name(x[0]) for x in similars
                        if x[0].upper() in self.synset_names and x[1]][:3]
        if not most_similar:
            print(neologism)
        return most_similar


if __name__ == '__main__':
    ftt = FasttextBaseline("models/cc.ru.300.bin", "../dataset/data", "../data/data.db")
    # test_data = ['блогер', 'аккаунт', 'транш', 'пауэрлифтинг', 'репост']
    test_data = read_dataset('../dataset/for_nouns_final.txt')
    result = ftt.predict_neighbour_hypernyms(test_data)
    save_to_file(result, "../dataset/predicted_other_nouns.tsv")
