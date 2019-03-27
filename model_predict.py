import tensorflow as tf
import data_helper
from tqdm import tqdm
import numpy as np
import math
import os
batch = data_helper.Batch()
import pickle

class MatchModel():
    # word2id, id2word, trainingSamples, trainingLabels = data_helper.loadDataset('data/train_data.pkl')
    def __init__(self, dropout_rate=0.1, lstm_dim=100, embedding_size=300, word2id=None, max_gradient_norm=5.0,
                 is_training=True, mode=None):
        self.dropout_rate = dropout_rate
        self.lstm_dim = lstm_dim
        self.embedding_size = embedding_size
        self.word2id = word2id
        self.vocab_size = len(self.word2id)
        # print(self.vocab_size)
        self.max_gradient_norm = max_gradient_norm
        self.is_training = is_training
        self.mode = mode
        # # 执行模型构建部分的代码
        # self.build_match_model(is_training=self.is_training)

    def cosine_distance(self, y1, y2):
        # y1 [....,a, 1, d]
        # y2 [....,1, b, d]
        eps = 1e-6
        cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
        y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
        y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
        return cosine_numerator / y1_norm / y2_norm

    def cal_relevancy_matrix(self, in_question_repres, in_passage_repres):
        in_question_repres_tmp = tf.expand_dims(in_question_repres, 1)  # [batch_size, 1, question_len, dim]
        in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2)  # [batch_size, passage_len, 1, dim]
        relevancy_matrix = self.cosine_distance(in_question_repres_tmp, in_passage_repres_tmp)
        # [batch_size, passage_len, question_len]
        return relevancy_matrix

    def _create_lstm_cell(self, is_training=True, num_layers=1):
        def single_lstm_cell(lstm_dim):
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            context_lstm_cell = tf.contrib.rnn.GRUCell(lstm_dim)
            # context_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
            # 添加dropout
            if is_training:
                context_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell,
                                                                  output_keep_prob=(1 - self.dropout_rate))
            return context_lstm_cell

        # 列表中每个元素都是调用single_rnn_cell函数
        context_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([single_lstm_cell(self.lstm_dim) for _ in range(num_layers)])
        # cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell(self.rnn_size) for _ in range(self.num_layers)])
        return context_lstm_cell

    def build_match_model(self, is_training=True):
        print('building model... ...')
        self.p_inputs = tf.placeholder(tf.int32, [None, None], name='p_inputs')
        self.p_inputs_length = tf.placeholder(tf.int32, [None], name='p_inputs_length')
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

        self.q_inputs = tf.placeholder(tf.int32, [None, None], name='q_inputs')
        self.q_inputs_length = tf.placeholder(tf.int32, [None], name='q_inputs_length')
        # 根据目标序列长度，选出其中最大值，然后使用该值构建序列长度的mask标志。用一个sequence_mask的例子来说明起作用
        #  tf.sequence_mask([1, 3, 2], 5)
        #  [[True, False, False, False, False],
        #  [True, True, True, False, False],
        #  [True, True, False, False, False]]
        # self.max_q_sequence_length = tf.reduce_max(self.q_inputs_length, name='max_q_len')
        # self.mask = tf.sequence_mask(self.q_inputs_length, self.max_q_sequence_length, dtype=tf.float32,
        #                              name='masks')

        with tf.variable_scope('p_input', reuse=tf.AUTO_REUSE):
            # 创建LSTMCell，两层+dropout
            p_context_lstm_cell = self._create_lstm_cell(is_training=is_training)
            # embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size], trainable=True,
            #                             initializer=tf.truncated_normal_initializer(stddev=0.1))
            if is_training:
                with open('data/vector_n.pkl', 'rb') as handle:
                    data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
                    word_vec = data['word_vec']
                # np.reshape(word_vec, [self.vocab_size, self.embedding_size])
                # print(len(word_vec))
                embedding = tf.get_variable('embedding', trainable=False,
                                            initializer=word_vec)
            else:
                embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size], trainable=False,
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))

            p_inputs_embedded = tf.nn.embedding_lookup(embedding, self.p_inputs)
            # p_inputs_embedded:[batch_size,p_inputs_size,embedding_size]
            p_outputs, p_state = tf.nn.dynamic_rnn(p_context_lstm_cell, p_inputs_embedded,
                                                   sequence_length=self.p_inputs_length,
                                                   dtype=tf.float32)
            # p_outputs:[batch_size,p_inputs_size,lstm_dim]

        with tf.variable_scope('q_input'):
            # 定义要使用的attention机制。
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.lstm_dim, memory=p_outputs,
                                                                       memory_sequence_length=self.p_inputs_length)
            # 定义decoder阶段要是用的LSTMCell，然后为其封装attention wrapper
            q_context_lstm_cell = self._create_lstm_cell(is_training)
            q_context_lstm_cell = tf.contrib.seq2seq.AttentionWrapper(cell=q_context_lstm_cell,
                                                                      attention_mechanism=attention_mechanism,
                                                                      attention_layer_size=self.lstm_dim,
                                                                      name='Attention_Wrapper')
            q_inputs_embedded = tf.nn.embedding_lookup(embedding, self.q_inputs)
            q_outputs, q_state = tf.nn.dynamic_rnn(q_context_lstm_cell, q_inputs_embedded,
                                                   sequence_length=self.q_inputs_length,
                                                   dtype=tf.float32)

        with tf.variable_scope('match_layer'):
            relevancy_matrix = self.cal_relevancy_matrix(p_outputs, q_outputs)
            relevancy_matrix = tf.concat(relevancy_matrix, axis=2)
            # relevancy_matrix = tf.reduce_max(relevancy_matrix, axis=2) # [batch_size,p_inputs_size]
            relevancy_matrix = tf.reshape(relevancy_matrix, [self.batch_size, 900])
            if self.mode == 'test':
                w_0 = tf.get_variable("w_0", [900, 300], dtype=tf.float32, trainable=True,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
                b_0 = tf.get_variable("b_0", [300], dtype=tf.float32, trainable=True,
                                      initializer=tf.zeros_initializer())
                w_1 = tf.get_variable("w_1", [300, 2], dtype=tf.float32, trainable=True,
                                      initializer=tf.zeros_initializer())
                b_1 = tf.get_variable("b_1", [2], dtype=tf.float32, trainable=True, initializer=tf.zeros_initializer())
                logits = tf.nn.relu(tf.matmul(relevancy_matrix, w_0) + b_0)
                logits = tf.nn.dropout(logits, self.keep_prob_placeholder)
                logits = tf.matmul(logits, w_1) + b_1  # [batch_size, 2]
                self.prob_t = tf.nn.softmax(logits, 1)
                self.prob_max = tf.reduce_max(self.prob_t, 0)[1]
                self.aa_t = tf.argmax(self.prob_t, 0)[1]
                # self.bb_t = tf.argmax(self.truth, 1)
                # self.correct_pred_t = tf.equal(tf.argmax(self.prob_t, 1), tf.argmax(self.truth, 1))
                # self.accuracy_t = tf.reduce_mean(tf.cast(self.correct_pred_t, tf.float32))
        saver = tf.train.Saver(tf.global_variables())
        return saver


    def infer(self, sess, batch):
        feed_dict = {self.p_inputs: batch.p_inputs,
                     self.p_inputs_length: batch.p_inputs_length,
                     self.q_inputs: batch.q_inputs,
                     self.q_inputs_length: batch.q_inputs_length,
                     self.keep_prob_placeholder: 0.5,
                     self.batch_size: 100}
        predict, index = sess.run([self.prob_max, self.aa_t], feed_dict=feed_dict)
        # print(predict, accuracy)
        return predict, index

    def recurrent_p(self, p, predictSamples):
        predict_prob = []
        batch_index = []
        batches = data_helper.getBatches_predict(predictSamples, p, 100)
        for nextBatch in batches:
            prob_i, batch_index_i = model.infer(sess1, nextBatch)
            predict_prob.append(prob_i)
            batch_index.append(batch_index_i)
        max_prob = max(predict_prob)
        if max_prob > 0.5:
            batch_num = predict_prob.index(max(predict_prob))
            sen_index = batch_num * 100 + batch_index[batch_num] + 1
            print('最匹配的句子为第{}批中的第{}句'.format(batch_num, batch_index[batch_num]))
        else:
            sen_index = -1
            print('No matching!!')
        return sen_index


if __name__ == '__main__':
    word2id, id2word, predictSamples = data_helper.loadDataset_predict('data/predict_data.pkl')
    sen_list, id_list = data_helper.build_p_input('data/sen2id.txt')
    model = MatchModel(word2id=word2id, is_training=False, mode='test')
    model.build_match_model(is_training=False)
    saver = tf.train.Saver(max_to_keep=3)
    model_dir = 'model_sen_match_15/'
    ckpt = tf.train.get_checkpoint_state(model_dir)
    print(ckpt.model_checkpoint_path)
    eps_list = []
    idea = 'love love love'
    for i in range(len(sen_list)):
        eps_i = []
        p = sen_list[i]+idea
        with tf.Session() as sess1:
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print('Reloading model parameters..')
                saver.restore(sess1, ckpt.model_checkpoint_path)
            while len(eps_i)==0 or eps_i[-1]!=-1 and len(eps_i)<15:
                sen_index = model.recurrent_p(p, predictSamples)
                if sen_index == -1:
                    if len(eps_i) == 0:
                        eps_i.append(-1)
                    else:
                        if len(eps_i)>=5 and eps_i[-1]-eps_i[-5]==4:
                            eps_i.append(-1)
                        else:
                            sen_p = eps_i[-1]+1
                            eps_i.append(sen_p)
                            p = sen_list[sen_p]+idea
                else:
                    if sen_index in eps_i:
                        eps_i.append(-1)
                    else:
                        eps_i.append(sen_index)
                        p = sen_list[sen_index]+idea
            eps_list.append(eps_i)

            # predict_prob = []
            # batch_index = []
            # batches = data_helper.getBatches_predict(predictSamples, p, 100)
            # for nextBatch in batches:
            #     prob_i, batch_index_i = model.infer(sess1, nextBatch)
            #     predict_prob.append(prob_i)
            #     batch_index.append(batch_index_i)
            # max_prob = max(predict_prob)
            # if max_prob>0.5:
            #     batch_num = predict_prob.index(max(predict_prob))
            #     sen_index = batch_num*100+batch_index[batch_num]+1
            #     print('最匹配的句子为第{}批中的第{}句'.format(batch_num, batch_index[batch_num]))
            # else:
            #     sen_index = -1
            #     print('No matching!!')
            # eps_i.append(sen_index)
    file = open('data/eps_list.txt', 'w')
    file.write(str(eps_list))
    file.close()
