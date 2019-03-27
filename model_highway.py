import torch as t
import torch.nn as nn
from torch.autograd import Variable as V
from torch.nn import functional as F
import numpy as np
import data_helper
import pickle

class MatchModel(nn.Module):
    def __init__(self,vocab_size,tr_w_embed_dim,tr_w_emb_dim_flag,char_size, char_embed_dim,output_dim,hidden_dim,use_gpu, yt_layer_num, mode = 'train'):
        super(MatchModel, self).__init__()  ##initialise the nn.Module
        self.layer_num = yt_layer_num
        self.use_gpu = use_gpu
        self.mode = mode
        self.tr_w_emb_dim_flag = tr_w_emb_dim_flag
        self.tr_w_embedding = nn.Embedding(vocab_size, tr_w_embed_dim)  # 输入： LongTensor (N, W), N = mini-batch, W = 每个mini-batch中提取的下标数
        if self.tr_w_emb_dim_flag != 0:
            self.tr_w_embedding1 = nn.Embedding(vocab_size, tr_w_emb_dim_flag)
        self.c_embedding = nn.Embedding(char_size, char_embed_dim)
        self.embedding_dim = tr_w_embed_dim + tr_w_emb_dim_flag+char_embed_dim+1 ## 651
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = 0.3
        fc_dim = 40 * 300
        self.emb_dim = emb_dim = self.output_dim * 2 + self.hidden_dim * 2
        self.ytlayer0_p = nn.ModuleList(self._build_multi_layers(1, self.embedding_dim))
        self.ytlayer0_q = nn.ModuleList(self._build_multi_layers(1, self.embedding_dim))
        self.ytlayers_p = nn.ModuleList(self._build_multi_layers(yt_layer_num, emb_dim*2))
        self.ytlayers_q = nn.ModuleList(self._build_multi_layers(yt_layer_num, emb_dim*2))
        self.fc_p = nn.ModuleList(self._make_fclayer(emb_dim))
        self.fc_q = nn.ModuleList(self._make_fclayer(emb_dim))
        self.fc_high_g = nn.Sequential(nn.Linear(300, 300),
                                       nn.Sigmoid())
        self.fc_high_t = nn.Sequential(nn.Linear(300, 300),
                                       nn.Tanh())
        self.fc = nn.Sequential(nn.Linear(fc_dim, 800),
                                nn.Dropout(p=self.dropout),
                                nn.ReLU(inplace=False),
                                nn.Linear(800, 200),
                                nn.Dropout(p=self.dropout),
                                nn.ReLU(inplace=False),
                                nn.Linear(200, 2))

    def _make_fclayer(self, input_dim):
        high_g = nn.Sequential(nn.Linear(input_dim, input_dim),
                               nn.Sigmoid())
        high_t = nn.Sequential(nn.Linear(input_dim, input_dim),
                               nn.Tanh())
        fc = nn.Sequential(nn.Linear(input_dim, 200),
                           nn.Dropout(p=self.dropout),
                           nn.ReLU(inplace=False),
                           nn.Linear(200, 50))
        return [high_g, high_t, fc]


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
        eps = 1e-6
        p_norm = t.sqrt(t.clamp(t.sum(t.pow(p_input, 2), -1), min = eps))
        q_norm = t.sqrt(t.clamp(t.sum(t.pow(q_input, 2), -1), min = eps))
        p_input_norm = t.div(p_input, t.unsqueeze(p_norm, 2))
        q_input_norm = t.div(q_input, t.unsqueeze(q_norm, 2))
        cosine_matrix = t.unsqueeze(t.bmm(q_input_norm, p_input_norm.transpose(1,2)), 1) ## [batch_size, 1, q_len, p_len]
        return cosine_matrix

    def _find_embedding(self, input, fix_embedding):
        for i, input_i in enumerate(input):
            input_vec_i = t.unsqueeze(t.index_select(fix_embedding, 0, input_i), 0)
            if i == 0:
                input_emb = input_vec_i
            else:
                input_emb = t.cat([input_emb, input_vec_i], 0)
        return input_emb

    def _make_ytlayer(self, input_dim):
        gru1 = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2, batch_first=True, dropout=self.dropout, bidirectional=True)
        gru2 = nn.GRU(self.emb_dim, self.output_dim, num_layers=2, batch_first=True, dropout=self.dropout, bidirectional=False)
        conv = nn.Sequential(nn.Conv2d(1, 1, (1,51)),
                             nn.Dropout(p=self.dropout),
                             nn.ReLU(inplace=False),
                             nn.MaxPool2d((1, 2)),
                             nn.Conv2d(1, 1, (1,36)),
                             nn.Dropout(p=self.dropout),
                             nn.ReLU(inplace=False),
                             nn.MaxPool2d((1, 2)),
                             nn.Conv2d(1,1,(1,21)))
        linear1 = nn.Linear(input_dim, self.emb_dim)
        highway_gate = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim),
                                     nn.Sigmoid())
        highway_trans = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim),
                                      nn.Tanh())
        return linear1, gru1, gru2, conv, highway_gate, highway_trans

    def _build_multi_layers(self, layer_num, input_dim):
        yt_layers = []
        for i in range(layer_num):
            x = self._make_ytlayer(input_dim)
            yt_layers = yt_layers+[*x]
        return yt_layers


    def _multi_emb(self, input, input_char, pretr_w_embedding=None, flags=None):
        batch_size, seq_len, word_len = input_char.data.shape
        if pretr_w_embedding is not None:
            pretr_w_embedding = self._find_embedding(input, pretr_w_embedding)
            if self.mode == 'train':
                self.fix_w_embedding = V(pretr_w_embedding, requires_grad = False).float()
            else:
                with t.no_grad():
                    self.fix_w_embedding = V(pretr_w_embedding).float()
        if self.use_gpu: self.fix_w_embedding = self.fix_w_embedding.cuda()
        tr_w_embedding = self.tr_w_embedding(input).float()
        if self.tr_w_emb_dim_flag != 0:
            tr_w_embedding1 = self.tr_w_embedding1(input).float()
            flags_squ = t.squeeze(flags)
            tr_w_embedding_flags = t.mul(tr_w_embedding1, t.softmax(flags, 1))
        # print('tr_w_embedding_flags:', tr_w_embedding_flags.shape)
        input_char = input_char.view([batch_size, seq_len*word_len])
        c_embedding = self.c_embedding(input_char)  # input_char:[batch_size, seq_len*word_len]
        c_embedding = c_embedding.view([batch_size, seq_len, word_len, -1])
        c_embedding = F.max_pool2d(c_embedding, kernel_size=(c_embedding.size()[-2], 1), stride=1)
        c_embedding = c_embedding.squeeze()  # c_embedding:[batch_size, seq_len, char_dim]
        multi_embedding = t.cat([tr_w_embedding, c_embedding, flags], 2)
        if self.tr_w_emb_dim_flag != 0:
            multi_embedding = t.cat([tr_w_embedding, tr_w_embedding_flags, c_embedding, flags], 2)
        return multi_embedding

    def _highway_layer(self, input, gate_f, trans_f):
        gate = gate_f(input)
        trans = trans_f(input)
        out = trans*gate + input*(1.0-gate)
        return out

    def _ytlayer(self, input_p, input_q, i):
        batch_size, seq_len, emb_dim = input_p.shape
        # print('seq_len:', seq_len, 'emb_dim:', emb_dim)
        if emb_dim == self.embedding_dim:
            linear1_p, gru1_p, gru2_p, conv_p, highway_gate_p, highway_trans_p = \
                self.ytlayer0_p[0], self.ytlayer0_p[1], self.ytlayer0_p[2], self.ytlayer0_p[3], self.ytlayer0_p[4], \
                self.ytlayer0_p[5]
            linear1_q, gru1_q, gru2_q, conv_q, highway_gate_q, highway_trans_q = \
                self.ytlayer0_q[0], self.ytlayer0_q[1], self.ytlayer0_q[2], self.ytlayer0_q[3], self.ytlayer0_q[4], \
                self.ytlayer0_q[5]
            self.res_p = input_p = linear1_p(input_p)
            self.res_q = input_q = linear1_q(input_q)
            p_g, _ = gru1_p(input_p)
            q_g, _ = gru1_q(input_q)
            p_c = t.squeeze(conv_p(t.unsqueeze(input_p, 1)))
            q_c = t.squeeze(conv_q(t.unsqueeze(input_q, 1)))
            p_a, q_a = self._attention_block(input_p, input_q)
            p_a, _ = gru2_p(p_a)
            q_a, _ = gru2_q(q_a)
            out_p = t.cat([p_g, p_a, p_c], 2)
            out_q = t.cat([q_g, q_a, q_c], 2)
            out_p = self._highway_layer(out_p, highway_gate_p, highway_trans_p)
            out_q = self._highway_layer(out_q, highway_gate_q, highway_trans_q)
        else:
            linear1_p, gru1_p, gru2_p, conv_p, highway_gate_p, highway_trans_p = \
                self.ytlayers_p[6*i+0], self.ytlayers_p[6*i+1], self.ytlayers_p[6*i+2], self.ytlayers_p[6*i+3], self.ytlayers_p[6*i+4], \
                self.ytlayers_p[6*i+5]
            linear1_q, gru1_q, gru2_q, conv_q, highway_gate_q, highway_trans_q = \
                self.ytlayers_q[6*i+0], self.ytlayers_q[6*i+1], self.ytlayers_q[6*i+2], self.ytlayers_q[6*i+3], self.ytlayers_q[6*i+4], \
                self.ytlayers_q[6*i+5]
            input_p = t.cat([input_p, self.res_p], 2)
            input_q = t.cat([input_q, self.res_q], 2)
            input_p = linear1_p(input_p)
            input_q = linear1_q(input_q)
            p_g, _ = gru1_p(input_p)
            q_g, _ = gru1_q(input_q)
            p_c = t.squeeze(conv_p(t.unsqueeze(input_p, 1)))
            q_c = t.squeeze(conv_q(t.unsqueeze(input_q, 1)))
            p_a, q_a = self._attention_block(input_p, input_q)
            p_a, _ = gru2_p(p_a)
            q_a, _ = gru2_q(q_a)
            out_p = t.cat([p_g, p_a, p_c], 2)
            out_q = t.cat([q_g, q_a, q_c], 2)
            out_p = self._highway_layer(out_p, highway_gate_p, highway_trans_p)
            out_q = self._highway_layer(out_q, highway_gate_q, highway_trans_q)
        return out_p, out_q

    def _fclayer(self, p, q):
        high_g_p, high_t_p, fc_p = self.fc_p[0], self.fc_p[1], self.fc_p[2]
        high_g_q, high_t_q, fc_q = self.fc_q[0], self.fc_q[1], self.fc_q[2]
        p = self._highway_layer(p, high_g_p, high_t_p)
        q = self._highway_layer(q, high_g_q, high_t_q)
        p = fc_p(p)
        q = fc_q(q)
        p_attention, q_attention = self._attention_block(p, q)
        fc = t.cat([p, q, p_attention, q_attention, p - q, p + q], 2)
        batch_size, seq_len, hidden_dim = fc.shape
        fc = self._highway_layer(fc, self.fc_high_g, self.fc_high_t)
        output = self.fc(fc.view([batch_size, seq_len * hidden_dim]))
        return output

    def forward(self, output_dim, p_input, q_input):
        [sample_batch_p, sample_batch_char_p, word_vec, sample_batch_flags_p] = p_input
        [sample_batch_q, sample_batch_char_q, sample_batch_flags_q] = q_input
        if self.mode == 'train':
            multi_embedding_p = self._multi_emb(V(sample_batch_p), V(sample_batch_char_p), word_vec, V(sample_batch_flags_p))
            multi_embedding_q = self._multi_emb(V(sample_batch_q), V(sample_batch_char_q), None, V(sample_batch_flags_q))
        else:
            with t.no_grad():
                multi_embedding_p = self._multi_emb(V(sample_batch_p), V(sample_batch_char_p), word_vec,
                                                V(sample_batch_flags_p))
                multi_embedding_q = self._multi_emb(V(sample_batch_q), V(sample_batch_char_q), None,
                                                V(sample_batch_flags_q))
        # return multi_embedding_p
        p, q = self._ytlayer(multi_embedding_p, multi_embedding_q, None)
        # print('b:',p.shape, q.shape)
        for i in range(self.layer_num):
            p, q = self._ytlayer(p, q, i)
        # print('c:',p.shape, q.shape)
        output = self._fclayer(p, q)
        # output = t.softmax(output, 1)
        return output
