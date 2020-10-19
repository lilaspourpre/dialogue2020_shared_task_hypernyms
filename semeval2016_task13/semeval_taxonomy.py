import networkx as nx


class SemEvalTaxonomy:
    def __init__(self, taxonomy_path, use_underscore=True):
        self.use_underscore = use_underscore
        self.G = self.create_taxonomy(taxonomy_path)

    def create_taxonomy(self, taxonomy_path):
        G = nx.DiGraph()
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            for line in f:
                child, parent = line.strip().split("\t")[1:]
                if self.use_underscore:
                    child, parent = child.replace(" ", "_"), parent.replace(" ", "_")
                G.add_edge(child, parent)
        return G

    def get_hypernym(self, word):
        if self.use_underscore:
            word = word.replace(" ", "_")
        return [i for i in self.G.neighbors(word)]

    def get_hyponym(self, word):
        if self.use_underscore:
            word = word.replace(" ", "_")
        return [i for i in self.G.predecessors(word)]

    def get_nodes(self):
        return self.G.nodes


if __name__ == '__main__':
    tax = SemEvalTaxonomy("data/gs_taxo/EN/science_eurovoc_en.taxo", use_underscore=True)
    print(tax.get_hypernym("geography"))
