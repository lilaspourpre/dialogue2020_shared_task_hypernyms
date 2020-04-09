import sys
import os


def conll2txt(files_path, output_dir):
    with open(os.path.join(files_path), 'rt', encoding='utf-8') as f, \
            open(os.path.join(output_dir), 'w', encoding='utf-8', newline="\n") as w:
        sentence = []
        words = []
        for line in f:
            line = line[:-1]
            if line and not line.startswith("#"):
                line_split = line.split("\t")
                sentence.append(line_split[2])
                words.append(line_split[1])
            if (line.startswith("#") or "SpacesAfter=\\n" in line) and sentence:
                w.write(" ".join(words) + "\t" + " ".join(sentence) + "\n")
                sentence = []
                words = []
        w.write(" ".join(words) + "\t" + " ".join(sentence) + "\n")


def main():
    if len(sys.argv) < 3:
        raise Exception("Please specify path to conll texts and the output directory")
    conll2txt(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
