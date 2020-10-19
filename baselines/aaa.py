
with open("ruwordnet_non-restricted-verbs_no_labels_final.tsv", 'r', encoding="utf-8") as f1, \
        open("../dataset/private/verbs_private_no_labels.tsv", 'r', encoding="utf-8") as f2, \
        open("../dataset/public/verbs_public_no_labels.tsv", 'r', encoding="utf-8") as f3:

    non_restricted = set(f1.read().split('\n'))
    restricted = set(f2.read().split('\n') + f3.read().split('\n'))
    restricted.remove("")

    print(f"Len non-restricted: {len(non_restricted)}")
    print(f"Len restricted: {len(restricted)}")
    print(f"Intersection: {len(restricted.intersection(non_restricted))}")
    print(f"Len difference: {len(restricted.difference(non_restricted))}")
    print(f"Difference: {restricted.difference(non_restricted)}")
