import os
import gzip
import json


def read_news_corpus(directory: str) -> list:
    file_paths = [os.path.join(x, i) for x, _, z in os.walk(directory) for i in z]
    return [i for f in file_paths[:10] for i in read_dataframe(f)]


def read_dataframe(filename: str) -> list:
    articles = []
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f:
            name, sentences_line = line[:-1].split(",", 1)
            if sentences_line != '[]':
                articles.append(NewsArticle(name, json.loads(sentences_line.strip('"').replace('""', '"'))))
    return articles


class NewsArticle:
    def __init__(self, name: str, sentences: list):
        self.name = name
        self.sentences = sentences

    def get_sentences(self, word):
        sentences = []
        return sentences


def main():
    path = "dataset/news_dataframes"
    news_articles = read_news_corpus(path)


if __name__ == '__main__':
    main()
