from bs4 import BeautifulSoup
import os
import codecs

from ruwordnet.database import DatabaseRuWordnet


def get_soup(file):
    with codecs.open(file, encoding='utf-8')as f:
        handler = f.read()
    return BeautifulSoup(handler, features="lxml")


def parse_synset(file):
    soup = get_soup(file)
    return [(element.attrs['id'], element.attrs['ruthes_name']) for element in soup.findAll('synset')]


def parse_relations(file):
    relations = []
    soup = get_soup(file)
    for element in soup.findAll('relation'):
        relation = element.attrs
        if relation['name'] in ('hypernym', 'instance hypernym'):
            relations.append((relation['parent_id'], relation['child_id']))
    return relations


def get_wordnet_files_from_path(path):
    synsets = []
    relations = []
    for directory, _, files in os.walk(path):
        for i in files:
            if i.startswith('synsets'):
                synsets.append(os.path.join(directory, i))
            elif i.startswith('synset_relation'):
                relations.append(os.path.join(directory, i))
    return synsets, relations


class RuWordnet(DatabaseRuWordnet):
    def __init__(self, db_path, ruwordnet_path):
        super(RuWordnet, self).__init__(db_path)
        self.__initialize_db(ruwordnet_path)

    def __initialize_db(self, path):
        if self.is_empty():
            print("Inserting data to data db")
            synset_files, relation_files = get_wordnet_files_from_path(path)
            synsets = {id_: ruthes_name for file in synset_files for id_, ruthes_name in parse_synset(file)}
            relations = [relation for file in relation_files for relation in parse_relations(file)]
            self.insert_synsets(synsets.items())
            self.insert_relations(relations)
