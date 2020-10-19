from nltk.corpus import WordNetCorpusReader
from fasttext_vectorize_en import compute_synsets_from_wordnets

wn2 = WordNetCorpusReader('D:\\dialogue2020\\semevals\\semeval-2016-task-14\\WN1.7', None)
wn3 = WordNetCorpusReader('D:\\dialogue2020\\semeval-2016-task-14\\WN3.0', None)
input_path = "D:/dialogue2020/semeval-2016-task-14/reader/"
vector_path = "models/vectors/fasttext/en/"

# vectorize wordnet
noun_synsets = compute_synsets_from_wordnets(wn2, wn3, 'n')
verb_synsets = compute_synsets_from_wordnets(wn2, wn3, 'v')

