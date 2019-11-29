import networkx as nx
from bs4 import BeautifulSoup
import os

from ruwordnet.database import DatabaseRuWordnet


def get_soup(file):
    with open(file, encoding='utf-8')as f:
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


def get_wordnet_files_from_path(path: str) -> tuple:
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
    def __init__(self, path, db_path="ruwordnet.db"):
        super().__init__(db_path)
        self.__initialize_db(*get_wordnet_files_from_path(path))
        self.G = self.create_graph()

    def __initialize_db(self, synset_files, relation_files):
        if self.is_empty():
            print("Inserting data to ruwordnet db")
            synsets = {id_: ruthes_name for file in synset_files for id_, ruthes_name in parse_synset(file)}
            relations = [relation for file in relation_files for relation in parse_relations(file)]
            self.insert_synsets(synsets.items())
            self.insert_relations(relations)

    def get_shortest_path_length(self, first_node, second_node):
        if first_node not in self.G.nodes or second_node not in self.G.nodes:
            return -1
        if nx.has_path(self.G, first_node, second_node):
            return nx.shortest_path_length(self.G, first_node, second_node)
        return -1

    def are_relatives(self, first_node, second_node):
        return 0 <= self.get_shortest_path_length(first_node, second_node) <= 2

    def create_graph(self):
        G = nx.Graph()
        for parent, child in self.get_all_relations():
            G.add_edge(parent, child)
        print('Graph size: {} nodes, {} edges'.format(len(G.nodes), len(G.edges)))
        return G


def main():
    path = "../dataset/ruwordnet"
    ruwordnet = RuWordnet(path)
    print([(i, ruwordnet.get_name_by_id(i)) for i in ruwordnet.get_hypernyms_by_name('автор')])
    print(ruwordnet.are_relatives("102729-N", "100091-N"))


if __name__ == '__main__':
    main()
