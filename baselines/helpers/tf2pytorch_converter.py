import sys
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Please specify BERT path")

    BERT_MODEL_PATH = sys.argv[1]  # ../models/pretrained_berts/enbert_cased_torch/
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=False)

    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
        BERT_MODEL_PATH + 'bert_model.ckpt',
        BERT_MODEL_PATH + 'bert_config.json',
        BERT_MODEL_PATH + 'pytorch_model.bin')
