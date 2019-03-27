from __future__ import absolute_import # 相对引入
from __future__ import division # 精确除法
from __future__ import print_function # 使旧版本兼容新版本的语法规则

import os
import nltk
import numpy as np
import pickle
import random
import collections
import string
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3

class Batch:
    #batch类，里面包含了encoder输入，decoder输入，decoder标签，decoder样本长度mask
    def __init__(self):
        self.p_inputs = []
        self.p_inputs_length = []
        self.q_inputs = []
        self.q_inputs_length = []
        self.targets = []
        self.p_inputs_char = []
        self.q_inputs_char = []
        self.random_index = None
        self.p_flags = []
        self.q_flags = []

def loadDataset(filename):
    '''
    读取样本数据
    :param filename: 文件路径，是一个字典，包含word2id、id2word分别是单词与索引对应的字典和反序字典，
                    trainingSamples样本数据，每一条都是QA对
    :return: word2id, id2word, trainingSamples
    '''
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['trainingSamples']
        trainingLabels = data['trainingLabels']
    return word2id, id2word, trainingSamples, trainingLabels

def loadDataset_test(filename):
    '''
    读取样本数据
    :param filename: 文件路径，是一个字典，包含word2id、id2word分别是单词与索引对应的字典和反序字典，
                    trainingSamples样本数据，每一条都是QA对
    :return: word2id, id2word, trainingSamples
    '''
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['testingSamples']
        trainingLabels = data['testingLabels']
    return word2id, id2word, trainingSamples, trainingLabels

def loadDataset_predict_(filename, filename_char):
    '''
    读取样本数据
    :param filename: 文件路径，是一个字典，包含word2id、id2word分别是单词与索引对应的字典和反序字典，
                    trainingSamples样本数据，每一条都是QA对
    :return: word2id, id2word, trainingSamples
    '''
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
        samples = data['predictSamples_pad']
    dataset_path_char = os.path.join(filename_char)
    print('Loading dataset from {}'.format(dataset_path_char))
    with open(dataset_path_char, 'rb') as handle_:
        data = pickle.load(handle_)
        samples_char = data['predictSamples_char_pad']
    return word2id, id2word, samples, samples_char

def loadDataset_flags(filename):
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        try:
            samples = data['trainingSamples_flags']
        except:
            samples = data['testingSamples_flags']
    return samples


def loadDataset_char(filename):
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        try:
            samples = data['trainingSamples_char']
        except:
            samples = data['testingSamples_char']
    return samples

def createBatch(samples, labels, index):
    '''
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer]的形式，id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    batch = Batch()
    batch.random_index = index
    # batch.p_inputs_length = [len(sample[0]) for sample in samples]
    # batch.q_inputs_length = [len(sample[1]) for sample in samples]
    batch.p_inputs_length = [249 for sample in samples]
    batch.q_inputs_length = [249 for sample in samples]

    # max_p_length = max(batch.p_inputs_length)
    # max_q_length = max(batch.q_inputs_length)
    max_p_length = 249
    max_q_length = 249
    # i = 0
    # print(samples[75])
    for sample in samples:
        #将source进行反序并PAD值本batch的最大长度
        p = sample[0]
        # i+=1
        # print(i)
        if len(p) >= max_p_length:
            batch.p_inputs.append(p[:max_p_length])
        else:
            pad_p = [padToken] * (max_p_length - len(p))
            batch.p_inputs.append(p + pad_p)
        # 将target进行PAD，并添加END符号
        q = sample[1]
        if len(q) >= max_q_length:
            batch.q_inputs.append(q[:max_q_length])
        else:
            pad_q = [padToken] * (max_q_length - len(q))
            batch.q_inputs.append(q + pad_q)
        # batch.target_inputs.append([goToken] + target + pad[:-1])
    for label in labels:
        # print(label)
        if label == '0':
            batch.targets.append([1,0])
            # print(1)
        else:
            batch.targets.append([0,1])
    return batch

def build_p_input(filename):
    sen2id = load_dict_from_file(filename)
    sentence_list = []
    id_list = []
    for sen in sen2id.keys():
        sentence_list.append(sen)
        id_list.append(sen2id[sen])
    return sentence_list, id_list

def createBatch_char(samples):
    batch = Batch()
    # batch.p_inputs_length = [len(sample[0]) for sample in samples]
    # batch.q_inputs_length = [len(sample[1]) for sample in samples]
    batch.p_inputs_length = [249 for sample in samples]
    batch.q_inputs_length = [249 for sample in samples]
    max_seq_length = 249

    max_char_length = 116
    def char_pad(sample_i):
        if len(sample_i)< max_seq_length:
            pad = [[padToken]*max_char_length]*(max_seq_length-len(sample_i))
            sample_i = sample_i+pad
        sample_pad = []
        for word in sample_i:
            if len(word)<max_char_length:
                pad_w = [padToken]*(max_char_length-len(word))
                word = word+pad_w
            sample_pad.append(word)
        return sample_pad
    for sample in samples:
        p, q = sample[0], sample[1]
        p_pad = char_pad(p)
        q_pad = char_pad(q)
        batch.p_inputs_char.append(p_pad)
        batch.q_inputs_char.append(q_pad)
    return batch

def createBatch_flags(samples):
    batch = Batch()
    # batch.p_inputs_length = [len(sample[0]) for sample in samples]
    # batch.q_inputs_length = [len(sample[1]) for sample in samples]
    batch.p_inputs_length = [249 for sample in samples]
    batch.q_inputs_length = [249 for sample in samples]
    max_seq_length = 249
    max_char_lenghth = 116
    def flags_pad(sample_i):
        if len(sample_i)<max_seq_length:
            pad_flags = [0]*(max_seq_length-len(sample_i))
            sample_i = sample_i+pad_flags
        return sample_i
    for sample in samples:
        p, q = sample[0], sample[1]
        p_pad = flags_pad(p)
        q_pad = flags_pad(q)
        batch.p_flags.append(p_pad)
        batch.q_flags.append(q_pad)
    return batch

def getBatches(trainingSamples, trainingLabels, batch_size):
    '''
    根据读取出来的所有数据和batch_size将原始数据分成不同的小batch。对每个batch索引的样本调用createBatch函数进行处理
    :param data: loadDataset函数读取之后的trainingSamples，就是QA对的列表
    :param batch_size: batch大小
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''
    #每个epoch之前都要进行样本的shuffle
    data_len = len(trainingSamples)
    index = np.arange(0, data_len, 1, np.int32)
    random.shuffle(index)
    random_trainingSamples = []
    random_trainingLabels = []
    for i in list(index):
        random_trainingSamples.append(trainingSamples[i])
        random_trainingLabels.append(trainingLabels[i])
    trainingSamples = random_trainingSamples
    trainingLabels = random_trainingLabels
    # random.shuffle(trainingSamples)
    batches = []
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            if i+batch_size < data_len:
                yield trainingSamples[i:min(i + batch_size, data_len)], trainingLabels[i:min(i + batch_size, data_len)]
    for samples, labels in genNextSamples():
        batch = createBatch(samples, labels, index)
        batches.append(batch)
    return batches

def getBatches_char(trainingSamples, batch_size, index):
    data_len = len(trainingSamples)
    # index = np.arange(0, data_len, 1, np.int32)
    # random.shuffle(index)
    random_trainingSamples = []
    for i in list(index):
        random_trainingSamples.append(trainingSamples[i])
    trainingSamples = random_trainingSamples
    batches = []
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            if i+batch_size < data_len:
                yield trainingSamples[i:min(i + batch_size, data_len)]
    for samples in genNextSamples():
        batch = createBatch_char(samples)
        batches.append(batch)
    return batches

def getBatches_flags(trainingSamples, batch_size, index):
    data_len = len(trainingSamples)
    # index = np.arange(0, data_len, 1, np.int32)
    # random.shuffle(index)
    random_trainingSamples = []
    for i in list(index):
        random_trainingSamples.append(trainingSamples[i])
    trainingSamples = random_trainingSamples
    batches = []
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            if i+batch_size < data_len:
                yield trainingSamples[i:min(i + batch_size, data_len)]
    for samples in genNextSamples():
        batch = createBatch_flags(samples)
        batches.append(batch)
    return batches


def sentence2enco(sentence, word2id):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    if sentence == '':
        return None
    #分词
    tokens = nltk.word_tokenize(sentence)
    # if len(tokens) > 20:
    #     return None
    #将每个单词转化为id
    wordIds = []
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
    #调用createBatch构造batch
    batch = createBatch([[wordIds, []]])
    return batch

def build_dictionary(sentences):
    # Turn sentences which are list of strings into list of words
    texts = [x.lower() for x in sentences]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation.replace('\'', '')) for x in texts]
    # Remove numbers
    # texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    split_sentence = [nltk.word_tokenize(sentence) for sentence in texts]
    # Turn sentences which are list of strings into list of words
    split_word = [w.split() for s in split_sentence for w in s]
    # print(split_word)
    split_words = []
    for w in split_word:
        [word] = w
        split_words.append(word)
    # print(len(split_words))
    # Initial list of [word, word_count] for each word, starting with unknown
    # count = [('RARE', -1)]
    count = []
    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    # print('vcab_len',len(collections.Counter(split_words).most_common()))
    count.extend(collections.Counter(split_words).most_common(99996))
    # count.extend(collections.Counter(split_words).most_common(10000))
    # Now create the dictionary
    word2id = {'padToken':0, 'goToken':1, 'eosToken':2, 'unknownToken':3}
    id2word = {0:'padToken', 1:'goToken', 2:'eosToken', 3:'unknownToken'}
    # For each word that we want in the dictionary, add it, then make it the value of the prior dictionary length
    for word, word_count in count:
        word_id = len(word2id)
        word2id[word] = word_id
        id2word[word_id] = word
    # print(len(word2id))
    return word2id, id2word
##============读取字典===============##
def load_dictionary():
    '''
    读取字典
    :return: word2id, id2word
    '''
    with open('data/dictionary.pkl', 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
    return word2id, id2word

def load_dictionary_char():
    '''
    读取字典
    :return: char2id, id2char
    '''
    with open('data/dictionary_char.pkl', 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        char2id = data['char2id']
        id2char = data['id2char']
    return char2id, id2char

##===========处理数据================##
def process_data(word2id, filename):
    '''
    处理数据
    :return: trainingSamples, trainingLables
    '''
    trainingSamples = []
    trainingLabels = []
    def sen2idx(sentence):
        if sentence == '':
            return None
        # 分词
        tokens = nltk.word_tokenize(sentence)
        # 将每个单词转化为id
        wordIds = []
        for token in tokens:
            wordIds.append(word2id.get(token, unknownToken))
        return wordIds

    with open(filename, 'r') as f:
        for line in f:
            data_pairs = line.split('\t')[1:3]
            sentence_0 = data_pairs[0].lower()
            [sentence_0] = [''.join(c for c in sentence_0 if c not in string.punctuation.replace('\'', ''))]
            wordIds_0 = sen2idx(str(sentence_0))
            sentence_1 = data_pairs[1].lower()
            [sentence_1] = [''.join(c for c in sentence_1 if c not in string.punctuation.replace('\'', ''))]
            wordIds_1 = sen2idx(str(sentence_1))
            if sentence_1 and sentence_0 and wordIds_0 != [] and wordIds_1 != []:
                # print('data0:',data_pairs[0],'\nsentence0:',sentence_0,'\ndata1:',data_pairs[1],'\nsentence1:',sentence_1)
                id_pairs = [wordIds_0, wordIds_1]
                trainingSamples.append(id_pairs)
                trainingLabels.append(line.split('\t')[0])
    return trainingSamples, trainingLabels

def process_data_snli(filename, word2id):
    '''
    处理数据
    :return: trainingSamples, trainingLables
    '''
    trainingSamples = []
    trainingLabels = []

    def sen2idx(sentence):
        if sentence == '':
            return None
        # 分词
        tokens = nltk.word_tokenize(sentence)
        # 将每个单词转化为id
        wordIds = []
        for token in tokens:
            wordIds.append(word2id.get(token, unknownToken))
        return wordIds
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        sentence0 = data['sentence1']
        sentence1 = data['sentence2']
        label = data['label']
    for i in range(len(label)):
        sentence_0 = sentence0[i].lower()
        [sentence_0] = [''.join(c for c in sentence_0 if c not in string.punctuation.replace('\'', ''))]
        wordIds_0 = sen2idx(str(sentence_0))
        sentence_1 = sentence1[i].lower()
        [sentence_1] = [''.join(c for c in sentence_1 if c not in string.punctuation.replace('\'', ''))]
        wordIds_1 = sen2idx(str(sentence_1))
        if sentence_1 and sentence_0 and wordIds_0 != [] and wordIds_1 != []:
            # print('data0:',data_pairs[0],'\nsentence0:',sentence_0,'\ndata1:',data_pairs[1],'\nsentence1:',sentence_1)
            id_pairs = [wordIds_0, wordIds_1]
            trainingSamples.append(id_pairs)
            trainingLabels.append(label[i])
    return trainingSamples, trainingLabels

def build_id2vec():
    word2vec = {}
    id2vec = {}
    with open('data/dictionary.pkl', 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
    with open('data/glove_300d.txt', 'r') as f:
        for line in f:
            line = line.split()
            if line[0] in word2id.keys():
                line_nn = []
                for i in line[1:]:
                    try:
                        line_nn.append(float(i))
                    except:
                        continue;
                word2vec[line[0]] = line_nn
                id2vec[word2id[line[0]]] = line_nn
    identity_sum = [0.] * 300
    for word in word2id.keys():
        if word not in word2vec.keys():
            word2vec[word] = identity_sum
            id2vec[word2id[word]] = identity_sum
    # print(len(word2vec), len(word2id), len(word2id.keys()), len(id2vec))
    word_vec = []
    for i in range(len(id2vec.keys())):
        word_vec.append(id2vec[i])
    # print(word_vec)
    all_data = {'word2vec': word2vec, 'id2vec': id2vec, 'word_vec': word_vec}
    output_file = open('data/vector_n.pkl', 'wb')
    pickle.dump(all_data, output_file)

def load_dict_from_file(filepath):
    _dict = {}
    try:
        with open(filepath, 'r') as dict_file:
            for line in dict_file:
                (key, value) = line.strip().split(':::')
                _dict[key] = value
    except IOError as ioerr:
        print("文件 %s 不存在" % (filepath))
    return _dict

def build_txtsamples(filename):
    sen2id = load_dict_from_file(filename)
    sentence_list = []
    for sen in sen2id.keys():
        sen = sen.lower()
        [sen] = [''.join(c for c in sen if c not in string.punctuation.replace('\'', ''))]
        sentence_list.append(sen)
    sen_ids = []
    word2id, id2word = load_dictionary()
    for sentence in sentence_list:
        tokens = nltk.word_tokenize(sentence)
        wordIds = []
        for token in tokens:
            wordIds.append(word2id.get(token.lower(), unknownToken))
        sen_ids.append(wordIds)
    print(sen_ids)
    all_data = {'word2id': word2id, 'id2word': id2word,
               'predictSamples':sen_ids}
    output_file = open('data/predict_data.pkl', 'wb')
    pickle.dump(all_data, output_file)

def build_q_input(filename):
    sen2id = load_dict_from_file(filename)
    sentence_list = []
    id_list = []
    for sen in sen2id.keys():
        sentence_list.append(sen)
        id_list.append(sen2id[sen])
    return sentence_list, id_list

def createBatch_predict(samples, sens):
    '''
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer]的形式，id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    batch = Batch()
    batch.q_inputs_length = [30 for sample in samples]
    batch.p_inputs_length = [30 for sample in samples]

    max_q_length = 30
    max_p_length = 30

    for sample in samples:
        #将source进行反序并PAD值本batch的最大长度
        # pad_q = []

        #将target进行PAD，并添加END符号
        q = sample
        if len(q) > max_q_length:
            batch.q_inputs.append(q[:max_q_length])
        else:
            pad_q = [padToken] * (max_q_length - len(q))
        #     while len(pad_q)<(max_q_length-len(q)):
        #         pad_q = pad_q + q[:min((max_q_length-len(q)-len(pad_q)),len(q))]
        #     pad_q = pad_q[:(max_q_length-len(q))]
        #     # pad = q[:(max_q_length-len(q))]
            batch.q_inputs.append(q + pad_q)
        #pad_q = [padToken] * (max_q_length - len(q))
        #batch.q_inputs.append(q + pad_q)
    for sen in sens:
        p = sen
        # pad_p = []
        if len(p) > max_p_length:
            batch.p_inputs.append(p[:max_p_length])
        else:
            pad_p = [padToken] * (max_p_length - len(p))
            batch.p_inputs.append(p + pad_p)
        #     while len(pad_p)<(max_p_length-len(p)):
        #         pad_p = pad_p + p[:min((max_p_length-len(p)-len(pad_p)),len(p))]
        #     pad_p = pad_p[:(max_p_length-len(p))]
        #     # print(len(p + pad_p))
        #     # pad = q[:(max_q_length-len(q))]
        #     batch.p_inputs.append(p + pad_p)
        
        #batch.target_inputs.append([goToken] + target + pad[:-1])
    return batch

def getBatches_predict(predictSamples, p, batch_size):
    '''
    根据读取出来的所有数据和batch_size将原始数据分成不同的小batch。对每个batch索引的样本调用createBatch函数进行处理
    :param data: loadDataset函数读取之后的trainingSamples，就是QA对的列表
    :param batch_size: batch大小
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''
    #每个epoch之前都要进行样本的shuffle
    p_ids = sentence_p_process(p)
    data_len = len(predictSamples)
    index = np.arange(0, data_len, 1, np.int32)
    # random.shuffle(index)
    random_predictSamples = []
    random_p = []
    for i in list(index):
        random_predictSamples.append(predictSamples[i])
        random_p.append(p_ids)
    trainingSamples = random_predictSamples
    batches = []
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            if i+batch_size < data_len:
                yield trainingSamples[i:min(i + batch_size, data_len)], random_p[i:min(i + batch_size, data_len)]
    for samples, p_sens in genNextSamples():
        batch = createBatch_predict(samples, p_sens)
        batches.append(batch)
    return batches

def sentence_p_process(p):
    p = p.lower()
    [p] = [''.join(c for c in p if c not in string.punctuation.replace('\'', ''))]
    tokens = nltk.word_tokenize(p)
    wordIds = []
    word_charIds = []
    word2id, id2word = load_dictionary()
    char2id, id2char = load_dictionary_char()
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
        charIds = []
        for char in token:
            charIds.append(char2id.get(char, padToken))
        word_charIds.append(charIds)
    return wordIds, word_charIds

def sentence_p_q_process(p, p_char, q_pad, q_char_pad):
    ## 处理单个p和多个q的函数
    data_len, seq_len, char_len = np.array(q_char_pad).shape
    q_flags_pad = []
    p_flags_pad = []
    for q_sentence in q_pad:
        p_flags = []
        q_flags = []
        for i in q_sentence:
            q_flags.append(1 if i in p and i!=unknownToken else 0)
            if len(q_flags)<seq_len:
                q_flags = q_flags+[0]*(seq_len-len(q_flags))
            else:
                q_flags = q_flags[:seq_len]
        for j in p:
            p_flags.append(1 if j in q_sentence and j!=unknownToken else 0)
            if len(p_flags)<seq_len:
                p_flags = p_flags+[0]*(seq_len-len(p_flags))
            else:
                p_flags = p_flags[:seq_len]
        q_flags_pad.append(q_flags)
        p_flags_pad.append(p_flags)

    def char_pad(p_char):
        i_list = []
        for i in p_char:
            if len(i)>=char_len:
                i = i[:char_len]
            else:
                pad_i = [0]*(char_len-len(i))
                i = i+pad_i
            i_list.append(i)
        return i_list
    if len(p)>=seq_len:
        p_pad = p[:seq_len]
        p_char = p_char[:seq_len]
    else:
        p_pad_ = [0]*(seq_len-len(p))
        p_char_pad_ = [[0]]*(seq_len-len(p))
        p_pad = p+p_pad_
        p_char = p_char+p_char_pad_
    p_char_pad = char_pad(p_char)
    return p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad

def sentence_p_q_process2(p, p_char, q_pad, q_char_pad):
    ## 处理多个p和多个q的函数
    data_len, seq_len, char_len = np.array(q_char_pad).shape
    q_flags_pad = []
    p_flags_pad = []
    p_pad = []
    p_char_pad = []
    p_char_ = []
    def char_pad(p_char_all):
        is_list = []
        for p_char in p_char_all:
            i_list = []
            for i in p_char:
                if len(i)>=char_len:
                    i = i[:char_len]
                else:
                    pad_i = [0]*(char_len-len(i))
                    i = i+pad_i
                i_list.append(i)
            is_list.append(i_list)
        return is_list
    for x, q_sentence in enumerate(q_pad):
        p_flags = []
        q_flags = []
        for i in q_sentence:
            q_flags.append(1 if i in p[x] and i!=unknownToken else 0)
            if len(q_flags)<seq_len:
                q_flags = q_flags+[0]*(seq_len-len(q_flags))
            else:
                q_flags = q_flags[:seq_len]
        for j in p[x]:
            p_flags.append(1 if j in q_sentence and j!=unknownToken else 0)
            if len(p_flags)<seq_len:
                p_flags = p_flags+[0]*(seq_len-len(p_flags))
            else:
                p_flags = p_flags[:seq_len]
        q_flags_pad.append(q_flags)
        p_flags_pad.append(p_flags)
        if len(p[x]) >= seq_len:
            p_pad.append(p[x][:seq_len])
            p_char_.append(p_char[x][:seq_len])
        else:
            p_pad_ = [0] * (seq_len - len(p[x]))
            p_char_pad_ = [[0]] * (seq_len - len(p[x]))
            p_pad.append(p[x] + p_pad_)
            p_char_.append(p_char[x]+p_char_pad_)
    p_char_pad = char_pad(p_char_)

    
    return p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad


def build_char_dictionary(sentences):
    # Turn sentences which are list of strings into list of words
    texts = [x.lower() for x in sentences]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation.replace('\'', '')) for x in texts]
    # Remove numbers
    # texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    split_sentence = [nltk.word_tokenize(sentence) for sentence in texts]
    # Turn sentences which are list of strings into list of words
    split_word = [w.split() for s in split_sentence for w in s]

    # print(split_word)
    split_words = []
    for w in split_word:
        [word] = w
        for char in word:
            split_words.append(char)
    # print(len(split_words))
    # Initial list of [word, word_count] for each word, starting with unknown
    # count = [('RARE', -1)]
    count = []
    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    # print('vcab_len',len(collections.Counter(split_words).most_common()))
    count.extend(collections.Counter(split_words).most_common(49))
    # count.extend(collections.Counter(split_words).most_common(10000))
    # Now create the dictionary
    char2id = {'padToken':0}
    id2char = {0:'padToken'}
    # For each word that we want in the dictionary, add it, then make it the value of the prior dictionary length
    for word, word_count in count:
        word_id = len(char2id)
        char2id[word] = word_id
        id2char[word_id] = word
    return char2id, id2char

def build_char_data(char2id, filename):
    trainingSamples = []

    def sen2idx(sentence):
        if sentence == '':
            return None
        # 分词
        tokens = nltk.word_tokenize(sentence)
        # 将每个单词转化为id
        wordIds = []

        for token in tokens:
            charIds = []
            for char in token:
                charIds.append(char2id.get(char, padToken))
            wordIds.append(charIds)
        return wordIds

    with open(filename, 'r') as f:
        for line in f:
            data_pairs = line.split('\t')[1:3]
            sentence_0 = data_pairs[0].lower()
            [sentence_0] = [''.join(c for c in sentence_0 if c not in string.punctuation.replace('\'', ''))]
            wordIds_0 = sen2idx(str(sentence_0))
            sentence_1 = data_pairs[1].lower()
            [sentence_1] = [''.join(c for c in sentence_1 if c not in string.punctuation.replace('\'', ''))]
            wordIds_1 = sen2idx(str(sentence_1))
            if sentence_1 and sentence_0 and wordIds_0 != [] and wordIds_1 != []:
                # print('data0:',data_pairs[0],'\nsentence0:',sentence_0,'\ndata1:',data_pairs[1],'\nsentence1:',sentence_1)
                id_pairs = [wordIds_0, wordIds_1]
                trainingSamples.append(id_pairs)
    return trainingSamples

def build_char_data_snli(char2id, filename):
    trainingSamples = []

    def sen2idx(sentence):
        if sentence == '':
            return None
        # 分词
        tokens = nltk.word_tokenize(sentence)
        # 将每个单词转化为id
        wordIds = []
        for token in tokens:
            charIds = []
            for char in token:
                charIds.append(char2id.get(char, padToken))
            wordIds.append(charIds)
        return wordIds
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        sentence0 = data['sentence1']
        sentence1 = data['sentence2']
        label = data['label']
    for i in range(len(label)):
        sentence_0 = sentence0[i].lower()
        [sentence_0] = [''.join(c for c in sentence_0 if c not in string.punctuation.replace('\'', ''))]
        wordIds_0 = sen2idx(str(sentence_0))
        sentence_1 = sentence1[i].lower()
        [sentence_1] = [''.join(c for c in sentence_1 if c not in string.punctuation.replace('\'', ''))]
        wordIds_1 = sen2idx(str(sentence_1))
        if sentence_1 and sentence_0 and wordIds_0 != [] and wordIds_1 != []:
            # print('data0:',data_pairs[0],'\nsentence0:',sentence_0,'\ndata1:',data_pairs[1],'\nsentence1:',sentence_1)
            id_pairs = [wordIds_0, wordIds_1]
            trainingSamples.append(id_pairs)
    return trainingSamples

def build_predict_char_data(char2id, sen_list):
    samples_char = []
    def sen2idx(sentence):
        if sentence == '':
            return None
        # 分词
        tokens = nltk.word_tokenize(sentence)
        # 将每个单词转化为id
        wordIds = []
        for token in tokens:
            charIds = []
            for char in token:
                charIds.append(char2id.get(char, padToken))
            wordIds.append(charIds)
        return wordIds
    for sen in sen_list:
        sen = ''.join(c for c in sen if c not in string.punctuation.replace('\'', ''))
        wordIds = sen2idx(str(sen))
        if wordIds != []:
            samples_char.append(wordIds)
    return samples_char


def build_flags(data):
    samples_flags = []
    for sample in data:
        p_flags = []
        q_flags = []
        p = sample[0]
        q = sample[1]
        for i in p:
            p_flags.append(1 if i in q and i!=unknownToken else 0)
        for j in q:
            q_flags.append(1 if j in p and j!=unknownToken else 0)
        samples_flags.append([p_flags, q_flags])
    return samples_flags


def sample_pad(samples, labels):
    max_p_length = 249
    max_q_length = 249
    # i = 0
    # print(samples[75])
    p_inputs = []
    q_inputs = []

    for sample in samples:
        # 将source进行反序并PAD值本batch的最大长度
        p = sample[0]
        # i+=1
        # print(i)
        if len(p) >= max_p_length:
            p_inputs.append(p[:max_p_length])
        else:
            pad_p = [padToken] * (max_p_length - len(p))
            p_inputs.append(p + pad_p)
        # 将target进行PAD，并添加END符号
        q = sample[1]
        if len(q) >= max_q_length:
            q_inputs.append(q[:max_q_length])
        else:
            pad_q = [padToken] * (max_q_length - len(q))
            q_inputs.append(q + pad_q)
        # batch.target_inputs.append([goToken] + target + pad[:-1])
    targets = []
    for label in labels:
        # print(label)
        if label == 0:
            targets.append([1, 0, 0])
            # print(1)
        elif label == 1:
            targets.append([0, 1, 0])
        elif label == 2:
            targets.append([0, 0, 1])
    return p_inputs, q_inputs, targets

def predict_sample_pad(samples):
    max_q_length = 249
    q_inputs = []
    for sample in samples:
        q = sample
        if len(q) >= max_q_length:
            q_inputs.append(q[:max_q_length])
        else:
            pad_q = [padToken] * (max_q_length - len(q))
            q_inputs.append(q + pad_q)
    return q_inputs

def predict_sample_char_pad(samples):
    max_seq_length = 249
    max_char_length = 116
    q_inputs_char = []

    def char_pad(sample_i):
        if len(sample_i) < max_seq_length:
            pad = [[padToken] * max_char_length] * (max_seq_length - len(sample_i))
            sample_i = sample_i + pad
        sample_pad = []
        for word in sample_i:
            if len(word) < max_char_length:
                pad_w = [padToken] * (max_char_length - len(word))
                word = word + pad_w
            sample_pad.append(word)
        return sample_pad

    for sample in samples:
        q = sample
        q_pad = char_pad(q)
        q_inputs_char.append(q_pad)
    return q_inputs_char



def sample_char_pad(samples):
    max_seq_length = 249
    max_char_length = 116
    p_inputs_char = []
    q_inputs_char = []
    def char_pad(sample_i):
        if len(sample_i) < max_seq_length:
            pad = [[padToken] * max_char_length] * (max_seq_length - len(sample_i))
            sample_i = sample_i + pad
        sample_pad = []
        for word in sample_i:
            if len(word) < max_char_length:
                pad_w = [padToken] * (max_char_length - len(word))
                word = word + pad_w
            sample_pad.append(word)
        return sample_pad

    for sample in samples:
        p, q = sample[0], sample[1]
        p_pad = char_pad(p)
        q_pad = char_pad(q)
        p_inputs_char.append(p_pad)
        q_inputs_char.append(q_pad)
    return p_inputs_char, q_inputs_char

def sample_flags_pad(samples):
    max_seq_length = 249

    def flags_pad(sample_i):
        if len(sample_i) < max_seq_length:
            pad_flags = [0] * (max_seq_length - len(sample_i))
            sample_i = sample_i + pad_flags
        return sample_i
    p_flags = []
    q_flags = []
    for sample in samples:
        p, q = sample[0], sample[1]
        p_pad = flags_pad(p)
        q_pad = flags_pad(q)
        p_flags.append(p_pad)
        q_flags.append(q_pad)
    return p_flags, q_flags

def loadDataset_pad(filename):
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
        p_inputs = data['p_inputs']
        q_inputs = data['q_inputs']
        targets = data['targets']
        p_inputs_char = data['p_inputs_char']
        q_inputs_char = data['q_inputs_char']
        p_flags = data['p_flags']
        q_flags = data['q_flags']
    return p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags

def loadDataset_predict(q_path, q_char_path, p):
    p, p_char = sentence_p_process(p)
    word2id, id2word, q_samples, q_samples_char = loadDataset_predict_(q_path, q_char_path)
    p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad = sentence_p_q_process(p, p_char, q_samples,
                                                                                          q_samples_char)
    return p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad



if __name__ == '__main__':
    p = 'maybe because you used to be aloof......or that you\'re really sarcastic'
    p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad = loadDataset_predict(
        'data/predict_data_pad.pkl', 'data/predict_data_char_pad.pkl', p)
    print(np.array(p_pad).shape, np.array(p_char_pad).shape)
    # print('p_pad:', p_pad.shape, 'p_char_pad:', p_char_pad.shape, 'p_flags_pad:', p_flags_pad.shape)
    # word2id, id2word, q_samples, q_samples_char = loadDataset_predict_('data/predict_data_pad.pkl', 'data/predict_data_char_pad.pkl')
    # p, p_char = sentence_p_process(p)
    # p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad = sentence_p_q_process(p, p_char, q_samples, q_samples_char)
    # print(np.array(p_pad).shape, np.array(p_char_pad).shape, np.array(p_flags_pad).shape, np.array(q_pad).shape, np.array(q_char_pad).shape, np.array(q_flags_pad).shape)
    # p_pad, p_char_pad are 1d. othters are nd.
    # print(len(q_samples), len(q_samples_char))

    # word2id, id2word, predictSamples = loadDataset_predict('data/predict_data.pkl')
    # samples = predict_sample_pad(predictSamples)
    # all_data = {'word2id': word2id, 'id2word': id2word,
    #             'predictSamples_pad': samples}
    # output_file = open('data/predict_data_pad.pkl', 'wb')
    # pickle.dump(all_data, output_file)

    # with open('data/predict_data_char.pkl', 'rb') as handle:
    #     data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
    #     predict_sample_char = data['predictSamples_char']
    # char_pad = predict_sample_char_pad(predict_sample_char)
    # all_data = {'predictSamples_char_pad': char_pad}
    # output_file = open('data/predict_data_char_pad.pkl', 'wb')
    # pickle.dump(all_data, output_file)


    # with open('data/predict_data_char.pkl', 'rb') as handle:
    #     data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
    #     predict_sample_char = data['predictSamples_char']
    # print(predict_sample_char[:2])

    # char2id, id2char = load_dictionary_char()
    # sen_list, id_list = build_q_input('data/sen2id.txt')
    # predict_samples_char = build_predict_char_data(char2id, sen_list)
    # all_data = {'char2id': char2id, 'id2char': id2char,
    #             'predictSamples_char': predict_samples_char}
    # output_file = open('data/predict_data_char.pkl', 'wb')
    # pickle.dump(all_data, output_file)


    # word2id, id2word, trainingSamples, trainingLabels = loadDataset_test('sick_data/dev_data_n.pkl')
    # # trainingSamples_char = loadDataset_char('sick_data/dev_data_char.pkl')
    # # flags = loadDataset_flags('sick_data/dev_data_flags.pkl')
    # p_inputs, q_inputs, targets = sample_pad(trainingSamples, trainingLabels)
    # # p_inputs_char, q_inputs_char = sample_char_pad(trainingSamples_char)
    # # p_flags, q_flags = sample_flags_pad(flags)
    # all_data = {'p_inputs':np.array(p_inputs), 'q_inputs':np.array(q_inputs), 'targets':np.array(targets)}
    # # all_data_char = {'p_inputs_char':p_inputs_char, 'q_inputs_char':q_inputs_char}
    # # all_data_flags = {'p_flags':p_flags, 'q_flags':q_flags}
    # output_file = open('sick_data/dev_data_pad.pkl', 'wb')
    # pickle.dump(all_data, output_file)
    # # output_file_char = open('sick_data/dev_data_char_pad.pkl', 'wb')
    # # pickle.dump(all_data_char, output_file_char)
    # # output_file_flags = open('sick_data/dev_data_flags_pad.pkl', 'wb')
    # # pickle.dump(all_data_flags, output_file_flags)

    # word2id, id2word, trainingSamples, trainingLabels = loadDataset('data/train_data_n.pkl')
    # trainingSamples_char = loadDataset_char('data/train_data_char.pkl')
    # batches = getBatches(trainingSamples, trainingLabels, 100)
    # random_index = batches[0].random_index
    # batches = getBatches_char(trainingSamples_char, 100, random_index)
    # batch = batches[0]
    # print(np.array(batch.p_inputs_char).shape, np.array(batch.q_inputs_char).shape)

    # with open('snli_data/train_data_pad.pkl', 'rb') as handle:
    #     data = pickle.load(handle)
    #     p_inputs = data['p_inputs']
    #     q_inputs = data['q_inputs']
    #     targets = data['targets']
    # print(targets[:10])
    # word2id, id2word, testingSamples, testingLabels = loadDataset_test('data/test_data_n.pkl')
    # word2id, id2word, devingSamples, devingLabels = loadDataset_test('data/dev_data_n.pkl')
    # word2id, id2word, trainingSamples, trainingLabels = loadDataset('data/train_data_n.pkl')
    # i = 0
    # for j in trainingSamples:
    #     if len(j[0])>30 or len(j[1])>30:
    #         i+=1
    # print(i, len(trainingSamples), trainingSamples[:10])
    # trainingSamples_flags = loadDataset_flags('data/train_data_flags.pkl')
    # testingSamples_flags = loadDataset_flags('data/test_data_flags.pkl')
    # devingSamples_flags = loadDataset_flags('data/dev_data_flags.pkl')
    # trainingSamples_char = loadDataset_char('data/train_data_char.pkl')
    # testingSamples_char = loadDataset_char('data/test_data_char.pkl')
    # devingSamples_char = loadDataset_char('data/dev_data_char.pkl')
    # print('train_sample:', len(trainingSamples), 'train_flags:', len(trainingSamples_flags), 'train_char', len(trainingSamples_char),
    #       '\ntest_sample:', len(testingSamples), 'test_flags:', len(testingSamples_flags), 'test_char', len(testingSamples_char),
    #       '\ndev_sample:', len(devingSamples), 'dev_flags:', len(devingSamples_flags), 'dev_char', len(devingSamples_char))
    # char_size = 0
    # for i in testingSamples:
    #     for j in i:
    #         char_size = max(char_size, len(j))
    # print(char_size)

    # word2id, id2word, trainingSamples, trainingLabels = loadDataset_test('sick_data/test_data_n.pkl')
    # # print(trainingSamples[:2])
    # samples_flags = build_flags(trainingSamples)
    # # print(samples_flags)
    # all_data = {'testingSamples_flags': samples_flags}
    # output_file = open('sick_data/test_data_flags.pkl', 'wb')
    # pickle.dump(all_data, output_file)

    # word2id, id2word, trainingSamples, trainingLabels = loadDataset('sick_data/train_data_n.pkl')
    # samples_flags = build_flags(trainingSamples)
    # print(samples_flags)
    # all_data = {'trainingSamples_flags':samples_flags}
    # output_file = open('sick_data/train_data_flags.pkl', 'wb')
    # pickle.dump(all_data, output_file)

    # with open('data/train.tsv', 'r') as f:
    #     sentences = []
    #     for line in f:
    #         data_pairs = line.split('\t')[1:3]
    #         sentences.append(data_pairs[0])
    #         sentences.append(data_pairs[1])
    # char2id, id2char = build_char_dictionary(sentences)
    # all_data = {'char2id': char2id, 'id2char': id2char}
    # output_file = open('data/dictionary_char.pkl', 'wb')
    # pickle.dump(all_data, output_file)

    # with open('data/dictionary_char.pkl', 'rb') as handle:
    #     data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
    #     char2id = data['char2id']
    #     id2char = data['id2char']
    # # print(char2id, id2char)
    # trainingSamples = build_char_data_snli(char2id, 'sick_data/test.pkl')
    # # print(trainingSamples)
    # all_data = {'char2id': char2id, 'id2char': id2char,
    #            'testingSamples_char': trainingSamples}
    # output_file = open('sick_data/dev_data_char.pkl', 'wb')
    # pickle.dump(all_data, output_file)


    # build_txtsamples('txt_data/sen2id.txt')
    # word2id, id2word, predictSamples = loadDataset_predict('data/predict_data.pkl')
    # print(predictSamples[:2])
    #
    # # print(len(word2id))
    # p = 'Oh, great, they haven\'t seen the place since I moved in.'
    #
    # batches = getBatches_predict(predictSamples, p, 100)
    # for batch in batches:
    #     print(batch.q_inputs, batch.q_inputs_length)

    # build_id2vec()


    # with open('data/train.tsv', 'r') as f:
    #     sentences = []
    #     for line in f:
    #         data_pairs = line.split('\t')[1:3]
    #         sentences.append(data_pairs[0])
    #         sentences.append(data_pairs[1])
    # word2id, id2word = build_dictionary(sentences)
    # print(len(word2id), word2id)
    # all_data = {'word2id': word2id, 'id2word': id2word}
    # output_file = open('data/dictionary.pkl', 'wb')
    # pickle.dump(all_data, output_file)

    # with open('data/dictionary.pkl', 'rb') as handle:
    #     data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
    #     word2id = data['word2id']
    #     id2word = data['id2word']
    # print(len(word2id))

    # word2id, id2word = load_dictionary()
    # print(word2id)
    # trainingSamples, trainingLabels = process_data_snli('sick_data/train.pkl', word2id)
    # print(len(trainingSamples))
    # all_data = {'word2id': word2id, 'id2word': id2word,
    #            'trainingSamples': trainingSamples,
    #            'trainingLabels': trainingLabels}
    # output_file = open('sick_data/train_data_n.pkl', 'wb')
    # pickle.dump(all_data, output_file)

    # testingSamples, testingLabels = process_data_snli('sick_data/test.pkl', word2id)
    # all_data = {'word2id': word2id, 'id2word': id2word,
    #             'testingSamples': testingSamples,
    #             'testingLabels': testingLabels}
    # output_file = open('sick_data/test_data_n.pkl', 'wb')
    # pickle.dump(all_data, output_file)
    # testingSamples, testingLabels = process_data(word2id, 'data/dev.tsv')
    # all_data = {'word2id': word2id, 'id2word': id2word,
    #             'testingSamples': testingSamples,
    #             'testingLabels': testingLabels}
    # output_file = open('data/dev_data_n.pkl', 'wb')
    # pickle.dump(all_data, output_file)

    # word2id, id2word, testingSamples, testingLabels = loadDataset_test('data/dev_data_n.pkl')
    # batches = getBatches(testingSamples, testingLabels, 100)

    # word2id, id2word, trainingSamples, trainingLabels = loadDataset('data/train_data_n.pkl')
    # print(len(trainingSamples), trainingLabels)
    # batches = getBatches(trainingSamples, trainingLabels, 100)

    # batch_inputs = batches[0].p_inputs[2:4]
    # batch_inputs_length = batches[0].p_inputs_length[2:4]
    # batch_labels = batches[0].targets[0:10]
    # print(batch_inputs, batch_inputs_length, batch_labels)
    # for i, batch in enumerate(batches):
    #     print(batch.p_inputs)
    #         if (len(batch.p_inputs[j]) != 150):
    #             print(len(batch.p_inputs[j]))
