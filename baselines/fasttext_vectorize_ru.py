import os
from collections import defaultdict

from ruwordnet.ruwordnet_reader import RuWordnet
from vectorizers.fasttext_vectorizer import FasttextVectorizer


def process_data(vectorizer, input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = f.read().lower().split("\n")[:-1]
    vectorizer.vectorize_words(dataset, output_file)


if __name__ == '__main__':
    ft = FasttextVectorizer("models/cc.ru.300.bin")
    ruwordnet = RuWordnet(db_path="../dataset/ruwordnet.db", ruwordnet_path=None)
    vector_path = "models/vectors/fasttext/ru/"

    # ----------------------
    # vectorize synsets
    # ----------------------
    # noun_synsets = defaultdict(list)
    # verb_synsets = defaultdict(list)
    # all_synsets = defaultdict(list)
    #
    # for sense_id, synset_id, text in ruwordnet.get_all_senses():
    #     if synset_id.endswith("N"):
    #         noun_synsets[synset_id].append(text.lower())
    #     elif synset_id.endswith("V"):
    #         verb_synsets[synset_id].append(text.lower())
    #     all_synsets[synset_id].append(text.lower())
    # ft.vectorize_groups(noun_synsets, os.path.join(vector_path, "ruwordnet.txt"))
    # ft.vectorize_groups(noun_synsets, os.path.join(vector_path, "ruwordnet_nouns.txt"))
    # ft.vectorize_groups(verb_synsets, os.path.join(vector_path, "ruwordnet_verbs.txt"))
    #
    # # ----------------------
    # # vectorize data
    # # ----------------------
    process_data(ft, "ruwordnet_non-restricted-nouns_no_labels_final.tsv", "non-restricted_nouns_fasttext.txt")
    process_data(ft, "ruwordnet_non-restricted-verbs_no_labels_final.tsv", "non-restricted_verbs_fasttext.txt")
    # process_data(ft, "../dataset/public/nouns_public_no_labels.tsv", os.path.join(vector_path, "nouns_public.txt"))
    # process_data(ft, "../dataset/private/verbs_private_no_labels.tsv", os.path.join(vector_path, "verbs_private.txt"))
    # process_data(ft, "../dataset/private/nouns_private_no_labels.tsv", os.path.join(vector_path, "nouns_private.txt"))
