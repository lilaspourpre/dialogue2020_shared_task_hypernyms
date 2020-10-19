from vectorizers.fasttext_vectorizer import FasttextVectorizer
import json
import re


pattern = re.compile("[^A-z \-]")
hypernyms = set()
counter = 0
ft = FasttextVectorizer("models/cc.en.300.bin")
with open("../wiki_en.jsonlines", 'r') as f:
    for line in f:
        for hypernym in json.loads(line)['hypernyms']:
            hypernym = hypernym.replace("|", " ").replace('--', '')
            hypernym = pattern.sub("", hypernym)
            if not all([i == " " for i in hypernym]):
                hypernyms.add(hypernym.replace(" ", "_"))
        counter += 1
print(counter)
ft.vectorize_multiword_data(hypernyms, "models/vectors/fasttext/en/wiki.txt", to_upper=False)
