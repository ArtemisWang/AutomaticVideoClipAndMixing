import torch as t
import re
import torch.nn as nn
import visualize_model as mp
import numpy as np
import pickle
from tqdm import tqdm
import build_dataset_pytorch as build_dataset_pytorch
from torch.utils.data import DataLoader
import visdom


if __name__ == '__main__':
    vis = visdom.Visdom(env=u'test2')
    a = None
    b = []
    with open('result/accall.txt', 'r') as f:
        for i in f:
            if not a:
                a = i
            else:
                a = a+i
        a = re.sub('[\n\[\]]', '', a)
        a = a.split('\t')
        for j in a:
            b.append(list(map(float, j.split())))
        # print(b)
    x = t.arange(1, 11)
    y0 = t.Tensor(b[0])
    y1 = t.Tensor(b[1])
    vis.line(X=x, Y=y0, win='explore1', opts=dict(title='Without CNN', legend=['all_train'], xlabel='epochs', ylabel='acc(%)'))
    vis.updateTrace(X=x, Y=y1, win='explore1', name='all_test')
    a = None
    b = []
    with open('result/acc0cnn.txt', 'r') as f:
        for i in f:
            if not a:
                a = i
            else:
                a = a+i
        a = re.sub('[\n\[\]]', '', a)
        a = a.split('\t')
        for j in a:
            b.append(list(map(float, j.split())))
    y0 = t.Tensor(b[0])
    y1 = t.Tensor(b[1])
    vis.updateTrace(X=x, Y=y0, win='explore1', name='0cnn_train')
    vis.updateTrace(X=x, Y=y1, win='explore1', name='0cnn_test')

    a = None
    b = []
    with open('result/accall.txt', 'r') as f:
        for i in f:
            if not a:
                a = i
            else:
                a = a + i
        a = re.sub('[\n\[\]]', '', a)
        a = a.split('\t')
        for j in a:
            b.append(list(map(float, j.split())))
        # print(b)
    x = t.arange(1, 11)
    y0 = t.Tensor(b[0])
    y1 = t.Tensor(b[1])
    vis.line(X=x, Y=y0, win='explore2',
             opts=dict(title='Without RNN', legend=['all_train'], xlabel='epochs', ylabel='acc(%)'))
    vis.updateTrace(X=x, Y=y1, win='explore2', name='all_test')
    a = None
    b = []
    with open('result/acc0rnn.txt', 'r') as f:
        for i in f:
            if not a:
                a = i
            else:
                a = a + i
        a = re.sub('[\n\[\]]', '', a)
        a = a.split('\t')
        for j in a:
            b.append(list(map(float, j.split())))
    y0 = t.Tensor(b[0])
    y1 = t.Tensor(b[1])
    vis.updateTrace(X=x, Y=y0, win='explore2', name='0rnn_train')
    vis.updateTrace(X=x, Y=y1, win='explore2', name='0rnn_test')


    a = None
    b = []
    with open('result/accall.txt', 'r') as f:
        for i in f:
            if not a:
                a = i
            else:
                a = a + i
        a = re.sub('[\n\[\]]', '', a)
        a = a.split('\t')
        for j in a:
            b.append(list(map(float, j.split())))
        # print(b)
    x = t.arange(1, 11)
    y0 = t.Tensor(b[0])
    y1 = t.Tensor(b[1])
    vis.line(X=x, Y=y0, win='explore3',
             opts=dict(title='Without attention', legend=['all_train'], xlabel='epochs', ylabel='acc(%)'))
    vis.updateTrace(X=x, Y=y1, win='explore3', name='all_test')
    a = None
    b = []
    with open('result/acc0att.txt', 'r') as f:
        for i in f:
            if not a:
                a = i
            else:
                a = a + i
        a = re.sub('[\n\[\]]', '', a)
        a = a.split('\t')
        for j in a:
            b.append(list(map(float, j.split())))
    y0 = t.Tensor(b[0])
    y1 = t.Tensor(b[1])
    vis.updateTrace(X=x, Y=y0, win='explore3', name='0att_train')
    vis.updateTrace(X=x, Y=y1, win='explore3', name='0att_test')


    a = None
    b = []
    with open('result/accall.txt', 'r') as f:
        for i in f:
            if not a:
                a = i
            else:
                a = a + i
        a = re.sub('[\n\[\]]', '', a)
        a = a.split('\t')
        for j in a:
            b.append(list(map(float, j.split())))
        # print(b)
    x = t.arange(1, 11)
    y0 = t.Tensor(b[0])
    y1 = t.Tensor(b[1])
    vis.line(X=x, Y=y0, win='explore4',
             opts=dict(title='Without residual module', legend=['all_train'], xlabel='epochs', ylabel='acc(%)'))
    vis.updateTrace(X=x, Y=y1, win='explore4', name='all_test')
    a = None
    b = []
    with open('result/acc0res.txt', 'r') as f:
        for i in f:
            if not a:
                a = i
            else:
                a = a + i
        a = re.sub('[\n\[\]]', '', a)
        a = a.split('\t')
        for j in a:
            b.append(list(map(float, j.split())))
    y0 = t.Tensor(b[0])
    y1 = t.Tensor(b[1])
    vis.updateTrace(X=x, Y=y0, win='explore4', name='0res_train')
    vis.updateTrace(X=x, Y=y1, win='explore4', name='0res_test')


    a = None
    b = []
    with open('result/accall.txt', 'r') as f:
        for i in f:
            if not a:
                a = i
            else:
                a = a + i
        a = re.sub('[\n\[\]]', '', a)
        a = a.split('\t')
        for j in a:
            b.append(list(map(float, j.split())))
        # print(b)
    x = t.arange(1, 11)
    y0 = t.Tensor(b[0])
    y1 = t.Tensor(b[1])
    vis.line(X=x, Y=y0, win='explore5',
             opts=dict(title='Without retention', legend=['all_train'], xlabel='epochs', ylabel='acc(%)'))
    vis.updateTrace(X=x, Y=y1, win='explore5', name='all_test')
    a = None
    b = []
    with open('result/acc0ret.txt', 'r') as f:
        for i in f:
            if not a:
                a = i
            else:
                a = a + i
        a = re.sub('[\n\[\]]', '', a)
        a = a.split('\t')
        for j in a:
            b.append(list(map(float, j.split())))
    y0 = t.Tensor(b[0])
    y1 = t.Tensor(b[1])
    vis.updateTrace(X=x, Y=y0, win='explore5', name='0ret_train')
    vis.updateTrace(X=x, Y=y1, win='explore5', name='0ret_test')


    # a = None
    # b = []
    # with open('result/accall.txt', 'r') as f:
    #     for i in f:
    #         if not a:
    #             a = i
    #         else:
    #             a = a + i
    #     a = re.sub('[\n\[\]]', '', a)
    #     a = a.split('\t')
    #     for j in a:
    #         b.append(list(map(float, j.split())))
    #     # print(b)
    # x = t.arange(0, 10)
    # y0 = t.Tensor(b[0])
    # y1 = t.Tensor(b[1])
    # vis.line(X=x, Y=y0, win='explore6',
    #          opts=dict(title='Exploring of MRJACR6', legend=['all_train'], xlabel='epochs', ylabel='acc(%)'))
    # vis.updateTrace(X=x, Y=y1, win='explore6', name='all_test')
    # a = None
    # b = []
    # with open('result/acc1att.txt', 'r') as f:
    #     for i in f:
    #         if not a:
    #             a = i
    #         else:
    #             a = a + i
    #     a = re.sub('[\n\[\]]', '', a)
    #     a = a.split('\t')
    #     for j in a:
    #         b.append(list(map(float, j.split())))
    # y0 = t.Tensor(b[0])
    # y1 = t.Tensor(b[1])
    # vis.line(X=x, Y=y0, win='explore3',
    #          opts=dict(title='Exploring of MRJACR3', legend=['1att_train'], xlabel='epochs', ylabel='acc(%)'))
    # vis.updateTrace(X=x, Y=y1, win='explore3', name='1att_test')
    # a = None
    # b = []
    # with open('result/acc1rnn.txt', 'r') as f:
    #     for i in f:
    #         if not a:
    #             a = i
    #         else:
    #             a = a + i
    #     a = re.sub('[\n\[\]]', '', a)
    #     a = a.split('\t')
    #     for j in a:
    #         b.append(list(map(float, j.split())))
    # y0 = t.Tensor(b[0])
    # y1 = t.Tensor(b[1])
    # vis.updateTrace(X=x, Y=y0, win='explore3', name='1rnn_train')
    # vis.updateTrace(X=x, Y=y1, win='explore3', name='1rnn_test')
    # a = None
    # b = []
    # with open('result/acc1cnn.txt', 'r') as f:
    #     for i in f:
    #         if not a:
    #             a = i
    #         else:
    #             a = a + i
    #     a = re.sub('[\n\[\]]', '', a)
    #     a = a.split('\t')
    #     for j in a:
    #         b.append(list(map(float, j.split())))
    # y0 = t.Tensor(b[0])
    # y1 = t.Tensor(b[1])
    # vis.updateTrace(X=x, Y=y0, win='explore3', name='1cnn_train')
    # vis.updateTrace(X=x, Y=y1, win='explore3', name='1cnn_test')
    # a = None
    # b = []
    # with open('result/accall.txt', 'r') as f:
    #     for i in f:
    #         if not a:
    #             a = i
    #         else:
    #             a = a + i
    #     a = re.sub('[\n\[\]]', '', a)
    #     a = a.split('\t')
    #     for j in a:
    #         b.append(list(map(float, j.split())))
    # y0 = t.Tensor(b[0])
    # y1 = t.Tensor(b[1])
    # vis.updateTrace(X=x, Y=y0, win='explore3', name='all_train')
    # vis.updateTrace(X=x, Y=y1, win='explore3', name='all_test')
    # a = None
    # b = []
    # with open('result/acc1cnn1rnn.txt', 'r') as f:
    #     for i in f:
    #         if not a:
    #             a = i
    #         else:
    #             a = a + i
    #     a = re.sub('[\n\[\]]', '', a)
    #     a = a.split('\t')
    #     for j in a:
    #         b.append(list(map(float, j.split())))
    # y0 = t.Tensor(b[0])
    # y1 = t.Tensor(b[1])
    # vis.updateTrace(X=x, Y=y0, win='explore3', name='c&r_train')
    # vis.updateTrace(X=x, Y=y1, win='explore3', name='c&r_test')