import torch as t
import torch.nn as nn
from torch.autograd import Variable as V
from torch.nn import functional as F
import numpy as np
import data_helper
import pickle

class MatchModel(nn.Module):
    def __init__(self, vocab_size=100000, emb_dim=100, use_gpu=False, mode='train', batch_size=100):
        super(MatchModel, self).__init__()  ##initialise the nn.Module
        self.emb = nn.Embedding(vocab_size, emb_dim, max_norm=1.0, norm_type=2, padding_idx=0)
        self.use_gpu = use_gpu
        self.mode = mode
        self.with_abs = True
        self.dropout = 0.2
        self.hidden_dim = emb_dim*2
        self.emb_dim = emb_dim
        self.use_init_emb = False
        self.batch_size = batch_size
        if self.use_init_emb:
            with open('data/vector_n.pkl', 'rb') as handle:
                data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
                word_vec = t.from_numpy(np.array(data['word_vec']))
            if use_gpu:
                word_vec = word_vec.cuda()
            self.emb.weight = nn.Parameter(word_vec)
        self.tanh = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                   nn.Tanh(),
                                   nn.Linear(self.hidden_dim, self.hidden_dim),
                                   nn.Tanh(),
                                   nn.Linear(self.hidden_dim, self.hidden_dim),
                                   nn.Tanh())
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim, 2),
                                nn.Softmax())
        self.lstm1 = nn.LSTM(emb_dim, emb_dim, num_layers=3, batch_first=True, dropout=0.1,
               bidirectional=False)
        self.lstm2 = nn.LSTM(emb_dim, emb_dim, num_layers=3, batch_first=True, dropout=0.1,
                             bidirectional=False)
        self.squeeze_p = nn.Sequential(nn.Linear(30*emb_dim, 10*emb_dim),
                                       nn.ReLU(),
                                       nn.Linear(10*emb_dim, emb_dim),
                                       nn.ReLU())
        self.squeeze_q = nn.Sequential(nn.Linear(30 * emb_dim, 10 * emb_dim),
                                       nn.ReLU(),
                                       nn.Linear(10 * emb_dim, emb_dim),
                                       nn.ReLU())

    def _attention_block(self, p_input, q_input):
        eps = 1e-6
        p_norm = t.sqrt(t.clamp(t.sum(t.pow(p_input, 2), -1), min = eps))
        q_norm = t.sqrt(t.clamp(t.sum(t.pow(q_input, 2), -1), min = eps))
        p_input_norm = t.div(p_input, t.unsqueeze(p_norm, 2))
        q_input_norm = t.div(q_input, t.unsqueeze(q_norm, 2))
        cosine_matrix = t.bmm(q_input_norm, p_input_norm.transpose(1,2))  ## [batch_size, q_len, p_len]
        q_attention = t.bmm(cosine_matrix, p_input) ## [batch_size, q_len, 100]
        p_attention = t.bmm(t.transpose(cosine_matrix, 1, 2), q_input)  ## [batch_size, p_len, 100]
        return p_attention, q_attention

    def _cos_block(self, p_input, q_input):
        eps = 1e-8
        p_norm = t.sqrt(t.clamp(t.sum(t.pow(p_input, 2), -1), min = eps))
        q_norm = t.sqrt(t.clamp(t.sum(t.pow(q_input, 2), -1), min = eps))
        p_input_norm = t.div(p_input, t.unsqueeze(p_norm, 2))
        q_input_norm = t.div(q_input, t.unsqueeze(q_norm, 2))
        cosine_matrix = t.squeeze(t.bmm(q_input_norm, p_input_norm.transpose(1, 2)))
        # cosine_matrix = t.unsqueeze(t.bmm(q_input_norm, p_input_norm.transpose(1,2)), 1) ## [batch_size, 1, q_len, p_len]
        return cosine_matrix

    def _find_embedding(self, input, fix_embedding):
        for i, input_i in enumerate(input):
            input_vec_i = t.unsqueeze(t.index_select(fix_embedding, 0, input_i), 0)
            if i == 0:
                input_emb = input_vec_i
            else:
                input_emb = t.cat([input_emb, input_vec_i], 0)
        return input_emb

    def _pre_emb(self, x):
        _, s_len, emb_dim = x.size()
        a = self._cos_block(x, x)
        e = t.eye(s_len)
        if self.use_gpu:
            e = e.cuda()
        a = a - e
        x_ = t.bmm(a, x)
        gate = self.gate(x)
        mem = self.mem(x_)
        up = self.up(x)
        one = t.ones(s_len, emb_dim)
        if self.use_gpu:
            one = one.cuda()
        y = gate * mem+(one-gate)*up
        y = t.cat([x, y], 2)
        return y

    def _att_process(self, x, xy):
        _, s_len, emb_dim = x.size()
        gate = self.gate(xy)
        mem = self.mem(xy)
        up = self.up(x)
        one = t.ones(s_len, emb_dim)
        if self.use_gpu:
            one = one.cuda()
        y = gate * mem + (one - gate) * up
        y = t.cat([x, y], 2)
        return y


    def forward(self, p_input, q_input):
        sample_batch_p = p_input[0]
        sample_batch_q = q_input[0]
        p_len = p_input[3]
        q_len = q_input[3]
        if self.mode == 'train':
            p = self.emb(V(sample_batch_p)).float()
            q = self.emb(V(sample_batch_q)).float()
        else:
            with t.no_grad():
                p = self.emb(V(sample_batch_p)).float()
                q = self.emb(V(sample_batch_q)).float()
        p, _ = self.lstm1(p)
        q, _ = self.lstm2(q)
        p = t.reshape(p, [-1, self.emb_dim])
        q = t.reshape(q, [-1, self.emb_dim])
        p_len_list = []
        q_len_list = []
        seq_len = sample_batch_q.shape[1]
        for i, len_i in enumerate(p_len):
            p_len_list.append(len_i+i*seq_len-1)
            q_len_list.append(q_len[i]+i*seq_len-1)
        p_len_list = t.Tensor(p_len_list).long()
        q_len_list = t.Tensor(q_len_list).long()
        if self.use_gpu:
            p = p.cpu()
            q = q.cpu()
        p = t.index_select(p, 0, p_len_list)
        q = t.index_select(q, 0, q_len_list)
        if self.use_gpu:
            p = p.cuda()
            q = q.cuda()
        pq = t.cat([p, q], -1)
        y = self.tanh(pq)
        y = self.fc(y)
        return y



