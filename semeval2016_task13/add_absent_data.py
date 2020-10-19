from scoring_program.utils import *

if __name__ == '__main__':
    # input_file = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\semeval2016_task13\\data\\systems\\TAXI\\EN\\TAXI_food.taxo"
    # additional_synsets = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\baselines\\predictions\\semeval\\predicted_TAXI_food_en_only_orphan_semeval.tsv"
    # outtput_file = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\semeval2016_task13\\data\\systems\\TAXI\\EN\\improved\\food_en_improved_semeval.taxo"
    input_file = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\semeval2016_task13\\data\\systems\\TAXI\\EN\\TAXI_science.taxo"
    additional_synsets = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\baselines\\predictions\\semeval\\predicted_TAXI_science_en_only_orphan_semeval.tsv"
    outtput_file = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\semeval2016_task13\\data\\systems\\TAXI\\EN\\improved\\science_en_improved_semeval.taxo"
    # input_file = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\semeval2016_task13\\data\\systems\\TAXI\\EN\\TAXI_environment.taxo"
    # additional_synsets = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\baselines\\predictions\\semeval\\predicted_TAXI_environment_eurovoc_en_only_orphan_semeval.tsv"
    # outtput_file = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\semeval2016_task13\\data\\systems\\TAXI\\EN\\improved\\environment_en_improved_semeval.taxo"
    # input_file = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\semeval2016_task13\\data\\systems\\TAXI\\EN\\food_wordnet_en.taxo"
    # additional_synsets = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\baselines\\predictions\\semeval\\predicted_TAXI_food_wordnet_en_only_orphan_semeval.tsv"
    # outtput_file = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\semeval2016_task13\\data\\systems\\TAXI\\EN\\improved\\food_wordnet_en_improved_semeval.taxo"
    # input_file = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\semeval2016_task13\\data\\systems\\TAXI\\EN\\science_wordnet_en.taxo"
    # additional_synsets = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\baselines\\predictions\\semeval\\predicted_TAXI_science_wordnet_en_only_orphan_semeval.tsv"
    # outtput_file = "D:\\dialogue2020\\dialogue2020_shared_task_hypernyms\\semeval2016_task13\\data\\systems\\TAXI\\EN\\improved\\science_wordnet_en_improved_semeval.taxo"

    predicted = read_dataset(additional_synsets)
    pairs = {}
    for word, variants in predicted.items():
        # for variant in variants:
        #     if variant in word:
        #         pairs[word] = variant
        # if word not in pairs:
        #     pairs[word] = variants[0]
        pairs[word] = variants[0]
    pairs = list(pairs.items())

    max_index = 0
    with open(input_file, 'r', encoding='utf-8') as r, open(outtput_file, 'w', encoding='utf-8', newline="\n") as w:
        for line in r:
            index = int(line.split("\t")[0])
            if index > max_index:
                max_index = index
            w.write(line)
        for n, pair in enumerate(pairs):
            if n != len(pairs)-1:
                w.write(str(max_index+n+1) + "\t" + "\t".join(pair) + "\n")
            else:
                w.write(str(max_index + n + 1) + "\t" + "\t".join(pair))
