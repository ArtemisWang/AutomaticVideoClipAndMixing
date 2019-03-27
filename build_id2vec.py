import data_helper
import pickle
import numpy as np
word2vec = {}
id2vec = {}
word2id, id2word, trainingSamples, trainingLabels = data_helper.loadDataset('data/train_data.pkl')
with open('data/glove_300d.txt','r') as f:
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
identity_sum = [0.]*300
for word in word2id.keys():
    if word not in word2vec.keys():
        word2vec[word] = identity_sum
        id2vec[word2id[word]] = identity_sum
# print(len(word2vec), len(word2id), len(word2id.keys()), len(id2vec))
word_vec = []
for i in range(len(id2vec.keys())):
    word_vec.append(id2vec[i])
# print(word_vec)
all_data = {'word2vec':word2vec, 'id2vec':id2vec, 'word_vec':word_vec}
output_file = open('data/vector.pkl', 'wb')
pickle.dump(all_data, output_file)

with open('data/vector.pkl', 'rb') as handle:
    data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
    word2vec = data['word2vec']
    id2vec = data['id2vec']
    word_vec = data['word_vec']
    # for i in range(len(word_vec)):
    #     if np.shape(word_vec[i]) != (300, ):
    #         print(i, np.shape(word_vec[i]))
    #         print(id2vec[i])
print(len(word_vec), np.shape(word_vec[:]), word_vec[4])