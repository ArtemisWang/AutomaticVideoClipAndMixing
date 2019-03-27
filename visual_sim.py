import data_helper
import codecs
import os
import torch as t
import torch.nn as nn
import model_pytorch_2 as mp
import numpy as np
import pickle
from tqdm import tqdm
import build_dataset_pytorch as build_dataset_pytorch
from torch.utils.data import DataLoader
from matplotlib.patches import Circle
import data_helper
from data_helper import sentence_p_process
import matplotlib.pyplot as plt
import visdom
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sympy import *
from sympy.abc import x,y,z
def senid2videoid(id_list, sen_list, senid_list):
    indexs = []
    sen_lists = []
    for i in senid_list:
        indexs.append(id_list[int(i)].replace('\'', '').strip(':[]').split(',')[0])
        sen_lists.append(sen_list[int(i)])
    return indexs, sen_lists

def predict(p, use_gpu, model_path):
    mm2 = mp.MatchModel(vocab_size=100000,tr_w_embed_dim=300,tr_w_emb_dim_flag=100,char_size=50,char_embed_dim=20,output_dim=100,
                       hidden_dim=200,use_gpu=use_gpu,yt_layer_num=1, mode='test')
    mm2.load_state_dict(t.load(model_path, map_location='cpu'))
    mm2.eval()
    if use_gpu: mm2.cuda()
    def data_convert(*data):
        datas = []
        for i in data:
            datas.append(i.cuda())
        return datas
    x = mm2.tr_w_embedding(t.Tensor(np.array(p)).long())
    return x

def pad_sen(p):
    seq_len = 30
    if len(p)>=seq_len:
        p_pad = p[:seq_len]
    else:
        p_pad_ = [0]*(seq_len-len(p))
        p_pad = p+p_pad_
    return p_pad

def _cos_block(p_input, q_input):
    eps = 1e-6
    p_norm = t.sqrt(t.clamp(t.sum(t.pow(p_input, 2), -1), min=eps))
    q_norm = t.sqrt(t.clamp(t.sum(t.pow(q_input, 2), -1), min=eps))
    p_input_norm = t.div(p_input, t.unsqueeze(p_norm, 2))
    q_input_norm = t.div(q_input, t.unsqueeze(q_norm, 2))
    cosine_matrix = t.bmm(q_input_norm, p_input_norm.transpose(1, 2)) ## [batch_size, 1, q_len, p_len]
    return cosine_matrix

def _vis_sim(id_list, sen_list, a):
    from sympy.abc import x, y, z
    indexs, sen_lists = senid2videoid(id_list, sen_list, a)
    # print(sen_lists)
    # with open('result/its_perfect.txt', 'w') as f:
    #     for i in sen_lists:
    #         f.write(i+'\n')

    id_li = []
    for i in sen_lists:
        p = sentence_p_process(i)[0]
        p = pad_sen(p)
        id_li.append(p)
    p_mask = t.unsqueeze((t.Tensor(id_li) != 0).float(), 2)
    p = predict(id_li, False, 'model/mm-important1_14.pth')
    # q = sentence_p_process('Neural networks (or deep learning) have garnered considerable attention for retrieval-based systems. Notably, the dominant state-of-the-art systems for many benchmarks are now neural models, almost completely dispensing with traditional feature engineering techniques altogether. In these systems, convolutional or recurrent networks are empowered with recent techniques such as neural attention, achieving very competitive results on standard benchmarks. The key idea of attention is to extract only the most relevant information that is useful for prediction. In the context of textual data, attention learns to weight words and sub-phrases within documents based on how important they are. In the same vein, co-attention mechanisms are a form of attention mechanisms that learn joint pairwise attentions, with respect to both document and query.')[0]
    # q = sentence_p_process('It\'s perfect.')[0]
    q = sentence_p_process('I love you.')[0]
    # q = sentence_p_process('I hate you.')[0]
    # q = sentence_p_process('I have a dream.')[0]
    q = predict([q], False, 'model/mm-important1_14.pth')
    q = t.squeeze(q)
    # print(q.shape, q)
    pca = PCA(n_components=2)
    pca_q = pca.fit_transform(np.array(q))
    pca_q = np.transpose(pca_q)
    plt.scatter(pca_q[0], pca_q[1], marker='o')
    # print(pca_q[0].shape)
    if pca_q[0].shape == (3,):
        # print(pca_q[0][1], pca_q[1])
        [a_1, a_2, a_3] = [round(pca_q[0][i], 2) for i in range(3)]
        [b_1, b_2, b_3] = [round(pca_q[1][i], 2) for i in range(3)]
        a0, b0, r0 = solve([(a_1 - x) ** 2 + (b_1-y) ** 2 - z ** 2, (a_2-x) ** 2 + (b_2 - y) ** 2 - z ** 2, (a_3-x) ** 2 + (b_3 - y) ** 2 - z ** 2],[x, y, z])[1]
        print(a0, b0, r0)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = a0 + r0 * np.cos(theta)
        y = b0 + r0 * np.sin(theta)
        plt.plot(x,y,color='yellowgreen')
    if pca_q[0].shape == (4,):
        # print(pca_q[0][1], pca_q[1])
        [a_1, a_2, a_3, a_4] = [round(pca_q[0][i], 2) for i in range(4)]
        [b_1, b_2, b_3, b_4] = [round(pca_q[1][i], 2) for i in range(4)]
        a0, b0, r0 = solve([(a_1 - x) ** 2 + (b_1-y) ** 2 - z ** 2, (a_2-x) ** 2 + (b_2 - y) ** 2 - z ** 2, (a_4-x) ** 2 + (b_4 - y) ** 2 - z ** 2],[x, y, z])[1]
        print(a0, b0, r0)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = a0 + r0 * np.cos(theta)
        y = b0 + r0 * np.sin(theta)
        plt.plot(x,y,color='yellowgreen')

    for i in range(len(a)):
        num_x = t.sum(p_mask[i]).int()
        pca_p = pca.fit_transform(np.array(p[i][:num_x]))
        pca_p = np.transpose(pca_p)
        if i == 0:
            x = pca_p[0]
            y = pca_p[1]
        else:
            # print(pca_p.shape, i)
            if len(pca_p) ==1:
                continue
            x = np.concatenate([x, pca_p[0]])
            y = np.concatenate([y, pca_p[1]])
    plt.scatter(x, y, marker='.')

    # q = q.repeat((len(a), 1, 1))
    # out = _cos_block(q, p)
    # out = p_mask * out
    # pca = PCA(n_components=2)
    # x = []
    # y = []
    # for i in range(len(a)):
    #     num = t.sum(p_mask[i]).int()
    #     pca_q = pca.fit_transform(np.array(t.squeeze(out)[i][:num]))
    #     pca_q = np.transpose(pca_q)
    #     if i == 0:
    #         x = pca_q[0]
    #         y = pca_q[1]
    #     else:
    #         print(pca_q[0], pca_q[1])
    #         x = np.concatenate([x, pca_q[0]])
    #         y = np.concatenate([y, pca_q[1]])
    # plt.scatter(x, y, marker='o')

if __name__ == '__main__':
    # a = np.concatenate([np.array([1,2,3]),np.array([2,3,4])])
    # print(a)
    with t.no_grad():
        sen_list, id_list = data_helper.build_p_input('data/sen2id.txt')
        # a = sorted([7856, 8700, 36588, 8146, 27101, 16537, 7698, 52051, 17873, 45575, 7695, 19122, 26166])
        a = [7695, 7696, 7698, 7699, 7700, 7856, 7857, 7858, 8146, 8147, 8148, 8700, 8701, 8702, 16537, 16538, 16539, 17873, 17874, 42057, 42058, 42059, 19122, 19123, 19124, 27101, 27102, 27103, 36588, 36589, 36590, 45575, 45576, 45577, 52051, 52052, 52053]
        b = [9181, 9182, 9996, 9997, 9998, 18607, 18608, 18609, 36588, 36589, 36590, 37342, 37343, 37344, 43103, 43104, 43105, 45328, 45329, 45330, 8822, 8823, 61195, 61196, 61197, 47729, 47730, 47731, 51793, 51794, 51795, 60363, 60364, 60365, 63482, 63483, 63484, 64605, 64606, 64607, 68625, 68626, 68627, 69149, 69150, 69151]
        c = [2803, 2804, 9181, 9182, 9183, 9185, 9186, 9187, 9996, 9997, 9998, 18607, 18608, 18609, 24001, 24002, 24003, 28101, 28102, 28103, 31150, 31151, 53359, 53360, 53361, 34172, 34173, 34174, 35621, 35622, 35623, 36588, 36589, 36590, 37342, 37343, 37344, 43103, 43104, 43105, 45328, 45329, 45330, 8822, 8823, 61195, 61196, 61197, 47729, 47730]
        d = [917, 918, 1562, 1563, 1564, 2803, 2804, 2805, 2843, 2844, 2845, 4674, 4675, 4676, 7695, 7696, 7697, 9997, 9998, 9999, 9181, 9182, 9183, 9185, 9186, 9187, 17558, 17559, 17560, 18607, 18608, 18609, 24001, 24002, 24003, 27137, 27138, 27139, 28101, 28102, 28103, 31150, 31151, 53359, 53360, 53361, 34172, 34173, 34174, 35621, 35622]
        e = [421, 422, 471, 472, 473, 917, 918, 919, 1562, 1563, 1564, 2803, 2804, 2805, 2843, 2844, 2845, 4674, 4675, 4676, 5037, 5038, 5039, 5041, 5042, 5043, 7695, 7696, 7697, 9997, 9998, 9999, 7856, 7857, 7858, 8483, 8484, 8485, 9181, 9182, 9183, 9185, 9186, 9187, 13699, 13700, 13701, 14168, 14169]
        f = [52051, 52052, 51794, 19122, 19123, 19124, 19125, 36588, 36589, 36590, 36591, 39192, 39193, 39194, 39195, 39399, 39400, 39401, 39402, 47729, 47730, 47731, 47732, 56042, 56043, 56044, 56045, 69149, 69150, 69151, 31531, 68077, 68078, 68079, 68080, 60206, 60207]
        # i love you 0.8
        g = [54191, 54192, 54193, 7856, 7857, 44515, 66135, 11786, 41099, 27138, 57326, 57327, 57328, 9001, 9002, 9003, 9004, 9181, 9182, 9183, 9185, 9186, 9187, 9188, 11806, 11807, 11808, 11809, 12920, 12921, 12922, 12923, 16537, 16538, 16539, 16540, 17024, 17025, 17026, 17027, 51794, 51795, 51797, 37832, 37833, 18607, 18608, 18609, 18611]
        # 0.4
        h = [54191, 54192, 13637, 1562, 1563, 1564, 1565, 30011, 30012, 30013, 30014, 5037, 5038, 5039, 5041, 5042, 5043, 5044, 57741, 37832, 37833, 5475, 5476, 5477, 46831, 6174, 6175, 6176, 6177, 51794, 51795,  51797, 6584, 6585, 6586, 6587, 7695, 7696, 7697, 57582, 55173, 44554, 21992, 27138, 14187, 25712, 7701, 7702, 7703]
        # 0.2
        i = [56042, 56043, 56044, 56045, 56046, 69149, 69150, 69151, 57582, 57583, 57584, 57585, 57586]
        # i hate you 0.8
        j = [54191, 54192, 54193, 54194, 44559, 44560, 44561, 44562, 44563, 471, 472, 473, 474, 475, 909, 910, 911, 912, 913, 917, 918, 919, 921, 922, 923, 924, 925, 1043, 1044, 1045, 1046, 1047, 1060, 1061, 1062, 1063, 1064, 1111, 1112, 1113, 1114, 28552, 28553, 67551, 67552, 67553, 67554, 67555, 1559, 1560]
        # i have a dream 0.8
        k = [21791, 21792, 21793, 21794, 21795, 430, 431, 432, 433, 434, 917, 918, 919, 920, 3722, 3723, 3724, 3725, 3726, 23275, 23276, 23277, 23278, 23280, 23281, 23282, 23283, 23284, 949, 950, 951, 952, 953, 957, 958, 959, 960, 961, 988, 989, 990, 991, 992, 1058, 1059, 1060, 1061, 1062, 1119, 1120]
        # it's perfect
        l = [38555, 38556, 38557, 38558, 38559, 56042, 56043, 56044, 56045, 56046]
        # basic(2)_0.8 i love you
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # ax1.set_title('Relationship between theme sentence and result sentences')
        # ax1.set_title('It\'s perfect.')
        ax1.set_title('I love you.')
        # ax1.set_title('I hate you.')
        # ax1.set_title('I have a dream.')
        # _vis_sim(id_list, sen_list, a)
        # _vis_sim(id_list, sen_list, b)
        # _vis_sim(id_list, sen_list, c)
        # _vis_sim(id_list, sen_list, d)
        # _vis_sim(id_list, sen_list, e)
        _vis_sim(id_list, sen_list, f)
        # _vis_sim(id_list, sen_list, g)
        # _vis_sim(id_list, sen_list, h)
        # _vis_sim(id_list, sen_list, i)
        # _vis_sim(id_list, sen_list, j)
        # _vis_sim(id_list, sen_list, k)
        # _vis_sim(id_list, sen_list, l)
        plt.legend('123')
        plt.axis('off')
        # plt.savefig('iloveu.pdf', dpi=600)
        # plt.savefig('ihateu.pdf', dpi=600)
        # plt.savefig('ihavedream.pdf', dpi=600)
        # plt.savefig('itsperfect.pdf', dpi=600)
        plt.show()
        # _vis_sim(id_list, sen_list, k)