def read_dataset(data_path, sep='\t'):
    vocab = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0]
            hypernyms = set(line_split[1].split(", "))
            vocab[word] = hypernyms
    return vocab


def save_to_file(words_with_hypernyms: dict, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("neologism\thypernym_ids\n")
        for word, hypernyms in words_with_hypernyms.items():
            f.write(word+"\t"+", ".join(hypernyms)+"\n")
