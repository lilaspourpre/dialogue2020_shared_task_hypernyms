from nltk.corpus import WordNetCorpusReader
from collections import defaultdict
from vectorizers.fasttext_vectorizer import FasttextVectorizer
import os


def compute_synsets_from_wordnets(old_wordnet, new_wordnet, pos):
    synsets_old = set(old_wordnet.all_synsets(pos))
    #synsets_new = set(new_wordnet.all_synsets(pos))
    #reference = synsets_new.intersection(synsets_old)
    return extract_senses(synsets_old)


def extract_senses(synsets) -> dict:
    result = defaultdict(list)
    for synset in synsets:
        for lemma in synset.lemmas():
            result[synset.name()].append(lemma.name().replace("_", " "))
    return result


def process_data(vectorizer, input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = f.read().split("\n")[:-1]
    vectorizer.vectorize_multiword_data(dataset, output_file, to_upper=False)


def main():
    ft = FasttextVectorizer("models/cc.en.300.bin")
    wn2 = WordNetCorpusReader('D:\\dialogue2020\\semevals\\semeval-2016-task-14\\WN1.6', None)
    wn3 = WordNetCorpusReader('D:\\dialogue2020\\semevals\\semeval-2016-task-14\\WN3.0', None)
    input_path = "D:/dialogue2020/semevals/semeval-2016-task-14/reader/"
    vector_path = "models/vectors/fasttext/en/new"

    # vectorize wordnet
    noun_synsets = compute_synsets_from_wordnets(wn2, wn3, 'n')
    verb_synsets = compute_synsets_from_wordnets(wn2, wn3, 'v')
    ft.vectorize_groups(noun_synsets, os.path.join(vector_path, "nouns_wordnet_fasttext_1.6-3.0.txt"), to_upper=False)
    ft.vectorize_groups(verb_synsets, os.path.join(vector_path, "verbs_wordnet_fasttext_1.6-3.0.txt"), to_upper=False)

    # vectorize words
    process_data(ft, os.path.join(input_path, "no_labels_nouns_en_new.1.6-3.0.tsv"),
                 os.path.join(vector_path, "nouns_fasttext_cut_1.6-3.0.txt"))
    process_data(ft, os.path.join(input_path, "no_labels_verbs_en_new.1.6-3.0.tsv"),
                 os.path.join(vector_path, "verbs_fasttext_cut_1.6-3.0.txt"))


if __name__ == '__main__':
    main()
