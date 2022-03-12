from bert_CU.bert_embedding import bert_embedding
from bert_CU.bert_LibriTTS_embedding import bert_embedding as bert_LibriTTS_embedding

import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, help="index")
    args = parser.parse_args()
    bert_embedding('./data/biaobei100/biaobei_prepare_result_22050/text.txt', './data/biaobei100/biaobei_prepare_result_22050/embeds-{}'.format(args.index), args.index)