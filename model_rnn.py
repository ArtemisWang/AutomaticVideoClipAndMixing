import torch as t
import torch.nn as nn
from torch.autograd import Variable as V
from torch.nn import functional as F
import numpy as np
import data_helper
import pickle

class MatchModel(nn.Module):
    def __init__(self, vocab_size=100000, emb_dim=300, use_gpu=False, mode='train'):
        super(MatchModel, self).__init__()  ##initialise the nn.Module
        self.emb = nn.Embedding(vocab_size, emb_dim, max_norm=1.0, norm_type=2, padding_idx=0)
        self.use_gpu = use_gpu
        self.mode = mode
        self.with_abs = True
        self.dropout = 0.2
        self.hidden_dim = emb_dim
        self.use_init_emb = False
        if self.use_init_emb:
            with open('data/vector_n.pkl', 'rb') as handle:
                data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
                word_vec = t.from_numpy(np.array(data['word_vec']))
            if use_gpu:
                word_vec = word_vec.cuda()
            self.emb.weight = nn.Parameter(word_vec)
        self.gate = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                  nn.Sigmoid())
        self.mem = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                 nn.Tanh())
        self.up = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                nn.Tanh())
        self.p_gru = nn.GRU(self.hidden_dim*2, self.hidden_dim, num_layers=2, batch_first=True, dropout=self.dropout, bidirectional=True)
        self.q_gru = nn.GRU(self.hidden_dim*2, self.hidden_dim, num_layers=2, batch_first=True, dropout=self.dropout, bidirectional=True)
        self.p_gru2 = nn.GRU(self.hidden_dim*2, 150, num_layers=2, batch_first=True, dropout=self.dropout, bidirectional=True)
        self.q_gru2 = nn.GRU(self.hidden_dim*2, 150, num_layers=2, batch_first=True, dropout=self.dropout, bidirectional=True)
        self.p_squ = nn.Sequential(nn.Linear(self.hidden_dim, 30),
                                   nn.Tanh())
        self.q_squ = nn.Sequential(nn.Linear(self.hidden_dim, 30),
                                   nn.Tanh())
        self.fc = nn.Sequential(nn.Linear(4*30, 20),
                                nn.Tanh(),
                                nn.Linear(20, 2))

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
        gate = self.gate(x_)
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
        sample_batch_p = p_input
        sample_batch_q = q_input
        if self.mode == 'train':
            p = self.emb(V(sample_batch_p)).float()
            q = self.emb(V(sample_batch_q)).float()
        else:
            with t.no_grad():
                p = self.emb(V(sample_batch_p)).float()
                q = self.emb(V(sample_batch_q)).float()
        if self.with_abs:
            p = self._pre_emb(p)
            q = self._pre_emb(q)
        else:
            p = t.cat([p, p], 2)
            q = t.cat([q, q], 2)
        p, _ = self.p_gru(p)
        q, _ = self.q_gru(q)
        p, q = self._attention_block(p, q)
        p, _ = self.p_gru2(p)
        q, _ = self.q_gru2(q)
        p = self.p_squ(p[:, -1, :])
        q = self.q_squ(q[:, -1, :])
        y = t.cat([p, q, p-q, p+q], -1)
        y = self.fc(y.view(y.size()[0], -1))
        return y











