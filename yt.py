import numpy as np
import torch as t
import tensorflow as tf
import pickle
import random
import visdom
from torch.nn import functional as F
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import pandas as pd
import data_helper

# x = tf.placeholder(tf.int32, [None, None], name='x')
# y = tf.placeholder(tf.int32, [None, None], name='y')
# z_0 = tf.concat((x,y), -1)
# z_1 = tf.concat((x,y), 1)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# feed_dict = {x:[[1,2,3],[1,111,1]],
#              y:[[4,5,6],[6,6,6]]}
# z0, z1 = sess.run([z_0, z_1], feed_dict=feed_dict)
# print(z0.shape, z1.shape)

# a = t.randn(4,4)
# print(t.sum(a, 1).shape)

# print([[1,3,2],[2],[3]]+[[0]]*3)
# i =0
# if i< 2:
#     print(i)
#     i+=1
# prob_temp_list = [1,2,3,4,7,3,4]
# max_prob = max(t.Tensor(prob_temp_list))
# max_i = t.argmax(t.Tensor(prob_temp_list))
# print(max_prob, max_i, prob_temp_list[max_i])

import random
# p = np.array([0,1,2])
# print(np.nonzero(p))


def _cos_block(p_input, q_input):
    eps = 1e-6
    p_norm = t.sqrt(t.clamp(t.sum(t.pow(p_input, 2), -1), min=eps))
    q_norm = t.sqrt(t.clamp(t.sum(t.pow(q_input, 2), -1), min=eps))
    p_input_norm = t.div(p_input, t.unsqueeze(p_norm, 2))
    q_input_norm = t.div(q_input, t.unsqueeze(q_norm, 2))
    cosine_matrix = t.unsqueeze(t.bmm(q_input_norm, p_input_norm.transpose(1, 2)), 1)  ## [batch_size, 1, q_len, p_len]
    return cosine_matrix


# a = t.Tensor([[[1,0,1],[0,1,1]],[[2,1,0],[0,1,2]]])
# b = t.Tensor([[[[0,1,0],[0,1,2],[0,1,2]],[[0,1,0],[0,1,2],[0,1,2]]],[[[0,1,0],[0,1,2],[0,1,2]],[[0,1,0],[0,1,2],[0,1,2]]]])
# # x = _cos_block(a,b)
# e = t.eye(2,3)
# x = t.ones(2,3)
# print(a*b-x)
# with open('data/vector_n.pkl', 'rb') as handle:
#     data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
#     word_vec = t.from_numpy(np.array(data['word_vec']))
#     print(word_vec.shape)
# print(a.shape, b.shape)
# mask = a!=0
# # y = np.ma.array(a, mask=mask)
# mask = t.unsqueeze(mask, 3)
# print((mask).float()*b)
# print(mask.shape)
# t.manual_seed(2)
# for i in range(10):
#     beg = random.randint(0, 10)
#     print(beg)
# y = list(range(0,10))
# s = random.sample(y, 5)
# z = [y[a] for a in s]
# print(y, s, z)

# vis = visdom.Visdom(env=u'test1')
# for ii in range(0,10):
#     x = t.Tensor([ii])
#     y=x
#     vis.line(X=x, Y=y, win='polynormal', update='append' if ii>0 else None)

# a = t.Tensor([9,3,4,2,1])
# b = t.Tensor([1,2,4,3,9])
# # vis.line(X=a, Y=b, win='polynormal', update='append')
# with open('result.txt', 'w') as f:
#     f.write(str(a.numpy()))


# out = F.softmax(t.Tensor([89.66, 79.61, 79.18, 72.65, 78.91]))
# print(out*80)
# q = [[0.1]*100, [0.2]*100, [0.3]*100]*3
# q = np.array(q)
# pca = PCA(n_components=2)
# pca_q = pca.fit_transform(q)
# print(pca_q)
# w = np.isnan(q)
# print(np.array(q).shape)
# tsne = TSNE(n_components=2)
# embed_vis_fitted = tsne.fit_transform(q)
# print(embed_vis_fitted)
# tsne=TSNE()
# tsne.fit_transform(data_zs)  #进行数据降维,降成两维
# #a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
# tsne=pd.DataFrame(tsne.embedding_,index=data_zs.index) #转换数据格式
#
# import matplotlib.pyplot as plt
#
# d=tsne[r[u'聚类类别']==0]
# plt.plot(d[0],d[1],'r.')
#
# d=tsne[r[u'聚类类别']==1]
# plt.plot(d[0],d[1],'go')
#
# d=tsne[r[u'聚类类别']==2]
# plt.plot(d[0],d[1],'b*')
# li = []
# for i in range(5):
#     a = t.Tensor([i,i,i])
#     li.append(a.numpy())
# b = np.array(li)
# c = np.array([b])
# print(t.from_numpy(c))
# print(c.shape)
# a = [[0,1,2]]
# b = [[0]]*2
# print(a+b)
# char_len = 3
# def char_pad(p_char):
#     i_list = []
#     for i in p_char:
#         if len(i) >= char_len:
#             i = i[:char_len]
#         else:
#             pad_i = [0] * (char_len - len(i))
#             i = i + pad_i
#         i_list.append(i)
#     return i_list
# print(char_pad(a+b))

p = [[1]*2, [2]*3]
p_char = [[[1]*2, [2]*3], [[1]*2, [2]*3, [2]*3]]
q = [[4]*3, [5]*3]
q_char = [[[4]*3, [5]*3], [[4]*3, [5]*3]]
print(np.array(q_char).shape)
p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad = data_helper.sentence_p_q_process2(p, p_char, q, q_char)
print(p_char_pad)
print(np.array(p_char_pad).shape)