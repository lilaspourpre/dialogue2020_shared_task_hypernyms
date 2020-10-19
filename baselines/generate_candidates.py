import sys
import json

from predict_models import HCHModel
from ruwordnet.ruwordnet_reader import RuWordnet


def load_config():
    if len(sys.argv) < 2:
        raise Exception("Please specify path to config file")
    with open(sys.argv[1], 'r', encoding='utf-8')as j:
        params = json.load(j)
    return params


def main():
    params = load_config()
    model = HCHModel(params)
    ruwordnet = RuWordnet(db_path=params["db_path"], ruwordnet_path=params["ruwordnet_path"])

    with open(params['test_path'], 'r', encoding='utf-8') as f:
        test_data = f.read().split("\n")[:-1]

    with open("private_nouns_top100_candidates_second_order.tsv", "w", encoding="utf-8") as w:
        for neologism in test_data:
            candidates = model.generate_associates(neologism, topn=10)
            for candidate, similarity in candidates:
                w.write(f"{neologism}\t{candidate}\t{similarity}\n")
                for second_order in ruwordnet.get_hypernyms_by_id(candidate):
                    w.write(f"{neologism}\t{second_order}\t{model.get_similarity(neologism, second_order)}\n")


if __name__ == '__main__':
    main()
