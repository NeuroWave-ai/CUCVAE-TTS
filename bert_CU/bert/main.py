import tensorflow as tf
from bert_CU.bert import modeling, optimization, tokenization
import numpy as np

class ImportBert():
    """  Importing and running isolated TF graph """
    def __init__(self):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        conf.gpu_options.per_process_gpu_memory_fraction=0.01
        print('############here$$$$$$$$$$$$')
        with tf.Session(graph=self.graph, config=conf) as sess:
            config_path = './bert_CU/bert/multi_cased_L-12_H-768_A-12/bert_config.json'
            checkpoint_path = './bert_CU/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt'
            dict_path = './bert_CU/bert/multi_cased_L-12_H-768_A-12/vocab.txt'
            bert_config = modeling.BertConfig.from_json_file(config_path)
            self.tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=False)

            self.input_ids_p = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids_p")
            self.input_mask_p = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_mask_p")
            self.segment_ids_p = tf.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids_p")

            model = modeling.BertModel(config=bert_config, is_training=False, input_ids=self.input_ids_p,
                                       input_mask=self.input_mask_p, token_type_ids=self.segment_ids_p,
                                        use_one_hot_embeddings=False)
            self.bert_output = model.get_pooled_output()
            bert_saver = tf.train.Saver()
            bert_saver.restore(self.sess, save_path=checkpoint_path)

    # def infer(self, text1):
    #     tokens = self.tokenizer.tokenize(text1)
    #     tokens = ['[CLS]'] + tokens
    #     input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    #     input_mask = [1] * len(input_ids)
    #     segment_ids = [0] * len(input_ids)
    #
    #     # tokens1 = self.tokenizer.tokenize(text2)
    #     # tokens1 = ['[SEP]'] + tokens1
    #     # input_ids1 = self.tokenizer.convert_tokens_to_ids(tokens1)
    #     # input_mask1 = [1] * len(input_ids1)
    #     # segment_ids1 = [1] * len(input_ids1)
    #     #
    #     # tokens += tokens1
    #     # input_ids += input_ids1
    #     # input_mask += input_mask1
    #     # segment_ids += segment_ids1
    #     input_ids = np.reshape(np.array(input_ids), (1, -1))
    #     input_mask = np.reshape(np.array(input_mask), (1, -1))
    #     segment_ids = np.reshape(np.array(segment_ids), (1, -1))
    #     return self.sess.run(self.bert_output, feed_dict={
    #     self.input_ids_p: input_ids,
    #     self.input_mask_p: input_mask,
    #     self.segment_ids_p: segment_ids
    # })
    def infer(self, text1, text2):
        tokens = self.tokenizer.tokenize(text1)
        tokens = ['[CLS]'] + tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        tokens1 = self.tokenizer.tokenize(text2)
        tokens1 = ['[SEP]'] + tokens1
        input_ids1 = self.tokenizer.convert_tokens_to_ids(tokens1)
        input_mask1 = [1] * len(input_ids1)
        segment_ids1 = [1] * len(input_ids1)

        tokens += tokens1
        input_ids += input_ids1
        input_mask += input_mask1
        segment_ids += segment_ids1
        input_ids = np.reshape(np.array(input_ids), (1, -1))
        input_mask = np.reshape(np.array(input_mask), (1, -1))
        segment_ids = np.reshape(np.array(segment_ids), (1, -1))
        return self.sess.run(self.bert_output, feed_dict={
        self.input_ids_p: input_ids,
        self.input_mask_p: input_mask,
        self.segment_ids_p: segment_ids
    })


