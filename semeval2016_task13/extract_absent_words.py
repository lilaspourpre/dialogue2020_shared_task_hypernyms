import sys
import os
from collections import Counter
import pandas as pd
import networkx as nx


def get_rootset(taxonomy_path):
    G = nx.DiGraph()
    with open(taxonomy_path, 'r', encoding="utf-8") as f:
        for line in f:
            child, parent = line.lower()[:-1].split("\t")[1:]
            G.add_edge(parent, child)
    G.remove_edges_from(G.selfloop_edges())
    root_nodes = [n for n, d in G.in_degree() if d == 0]
    return set(root_nodes)


def get_termlist_with_root(filepath: str, taxonomy_path: str):
    root_set = get_rootset(taxonomy_path)
    with open(filepath, 'r', encoding="utf-8") as f:
        terms = set([line.split("\t")[1] for line in f.read().lower().split("\n") if line])
    return {"terms": terms.difference(root_set), "root": root_set}


def get_workspace(language, gs_terms_dir, gs_taxonomies_path, systems_dir):
    # -------------- terms --------------
    gs_terms = {}

    gs_terms_dir = os.path.join(gs_terms_dir, language)
    gs_taxonomies_dir = os.path.join(gs_taxonomies_path, language)

    for term_path, taxonomy_path in zip(os.listdir(gs_terms_dir), os.listdir(gs_taxonomies_dir)):
        dataset_name = os.path.splitext(term_path)[0]
        assert dataset_name == os.path.splitext(taxonomy_path)[0]
        gs_terms[dataset_name] = get_termlist_with_root(os.path.join(gs_terms_dir, term_path),
                                                        os.path.join(gs_taxonomies_dir, taxonomy_path))

    # -------------- systems --------------

    systems = {system: os.path.join(systems_dir, system, language) for system in os.listdir(systems_dir)}

    return gs_terms, systems


def reformat_name(name):
    reformatted_name = os.path.splitext(name)[0]
    if "environment" in reformatted_name:
        reformatted_name = "environment_eurovoc_en"
    if reformatted_name.split("_")[0].isupper():
        reformatted_name = reformatted_name.replace(reformatted_name.split("_")[0] + "_", "")
    if not reformatted_name.endswith("_en"):
        reformatted_name += "_en"
    return reformatted_name


def write_to_file(terms, out_file):
    with open(out_file, 'w', encoding='utf-8', newline='\n') as w:
        for i, term in enumerate(terms):
            w.write(f"{i}\t{term}\n")


def get_filename(outpath, system, reformatted_name, name):
    return os.path.join(outpath, f"{system}_{reformatted_name}_{name}.terms")


def main():
    parsed_data_path = "parsed_data"
    if len(sys.argv) != 5:
        raise Exception("not enough arguments: <language> <gs terms path> <gs taxonomy path> <systems path>")
    gs_terms, systems = get_workspace(*sys.argv[1:])

    for system, system_dir in systems.items():
        os.makedirs(os.path.join(system_dir, parsed_data_path), exist_ok=True)
        path = os.path.join(system_dir, parsed_data_path)

        for taxo_name in os.listdir(system_dir):
            if taxo_name.endswith("taxo"):
                reformatted_name = reformat_name(taxo_name)

                all_terms = gs_terms[reformatted_name]['terms']
                correct_root = gs_terms[reformatted_name]['root']

                system_terms, system_root = get_termlist_with_root(os.path.join(system_dir, taxo_name),
                                                                   os.path.join(system_dir, taxo_name)).values()

                all_absent_synsets = system_terms.intersection(all_terms).union(correct_root)
                all_absent_terms = all_terms.difference(all_absent_synsets)

                only_orphan_synsets = system_terms.intersection(all_terms).union(system_root).union(correct_root)
                only_orphan_terms = all_terms.difference(only_orphan_synsets)

                print(system, reformatted_name, correct_root, system_root, all_absent_terms == only_orphan_terms)
                print(len(all_terms), len(all_absent_synsets), len(all_absent_terms), len(only_orphan_synsets), len(only_orphan_terms))
                print("--")

                # if len(all_absent_terms) > 0 and all_absent_terms != only_orphan_terms:
                #     write_to_file(all_absent_terms, get_filename(path, system, reformatted_name, "all_absent"))
                #     write_to_file(all_absent_synsets, get_filename(path, system, reformatted_name, "all_absent_synsets"))
                # if len(only_orphan_terms) > 0:
                #     write_to_file(only_orphan_terms, get_filename(path, system, reformatted_name, "only_orphan"))
                #     write_to_file(only_orphan_synsets, get_filename(path, system, reformatted_name, "only_orphan_synsets"))


if __name__ == '__main__':
    main()
