import torch as t
import torch.nn as nn
from torch.autograd import Variable as V
from torch.nn import functional as F
import numpy as np
import data_helper
import pickle
import torch.nn.init as init

class MatchModel(nn.Module):
    def __init__(self,vocab_size,tr_w_embed_dim,tr_w_emb_dim_flag,char_size, char_embed_dim,output_dim,hidden_dim,use_gpu, yt_layer_num, mode = 'train'):
        super(MatchModel, self).__init__()  ##initialise the nn.Module
        self.layer_num = yt_layer_num
        self.use_gpu = use_gpu
        self.mode = mode
        self.tr_w_emb_dim_flag = tr_w_emb_dim_flag
        self.tr_w_embedding = nn.Embedding(vocab_size, tr_w_embed_dim, padding_idx=0)  # 输入： LongTensor (N, W), N = mini-batch, W = 每个mini-batch中提取的下标数
        if self.tr_w_emb_dim_flag != 0:
            self.tr_w_embedding1 = nn.Embedding(vocab_size, tr_w_emb_dim_flag, padding_idx=0)
        self.c_embedding = nn.Embedding(char_size, char_embed_dim, padding_idx=0)
        self.embedding_dim = tr_w_embed_dim + tr_w_emb_dim_flag+char_embed_dim+1 ## 651
        self.output_dim = output_dim  ## 200
        self.hidden_dim = hidden_dim  ## 200
        emb_dim_fc = self.output_dim * 3 + self.hidden_dim * 2
        self.dropout = 0.1
        self.seq_len = 30
        self.layer0_lstm_p, self.layer0_linear1_p, self.layer0_linear2_p, self.layer0_linear0_p = self._make_layer(self.embedding_dim)
        self.layer0_lstm_q, self.layer0_linear1_q, self.layer0_linear2_q, self.layer0_linear0_q = self._make_layer(self.embedding_dim)
        self.yt_layers_p = nn.ModuleList(self._build_multi_layers(self.layer_num, emb_dim_fc))
        self.yt_layers_q = nn.ModuleList(self._build_multi_layers(self.layer_num, emb_dim_fc))
        self.p_fc0 = nn.Sequential(nn.Linear(emb_dim_fc, 200),
                                   nn.Dropout(p=self.dropout),
                                   nn.ReLU(inplace=False),
                                   nn.Linear(200, 50))
        self.q_fc0 = nn.Sequential(nn.Linear(emb_dim_fc, 200),
                                   nn.Dropout(p=self.dropout),
                                   nn.ReLU(inplace=False),
                                   nn.Linear(200, 50))
        self.fc = nn.Sequential(nn.Linear(self.seq_len * 300, 800),
                                nn.Dropout(p=self.dropout),
                                nn.ReLU(inplace=False),
                                nn.Linear(800, 200),
                                nn.Dropout(p=self.dropout),
                                nn.ReLU(inplace=False),
                                nn.Linear(200, 2))
        self.bn_emb_p = nn.BatchNorm1d(self.seq_len, momentum=0.5)
        self.bn_emb_q = nn.BatchNorm1d(self.seq_len, momentum=0.5)
        # self.conv_p = nn.Sequential(nn.Conv2d(1, 2, 5),
        #                             nn.Dropout(p=self.dropout),
        #                             nn.ReLU(inplace=False),
        #                             nn.MaxPool2d((2, 2)),
        #                             nn.Conv2d(2, 4, 4),
        #                             nn.Dropout(p=self.dropout),
        #                             nn.ReLU(inplace=False),
        #                             nn.MaxPool2d((2, 2)))
        # self.conv_q = nn.Sequential(nn.Conv2d(1, 2, 5),
        #                             nn.Dropout(p=self.dropout),
        #                             nn.ReLU(inplace=False),
        #                             nn.MaxPool2d((2, 2)),
        #                             nn.Conv2d(2, 4, 4),
        #                             nn.Dropout(p=self.dropout),
        #                             nn.ReLU(inplace=False),
        #                             nn.MaxPool2d((2, 2)))
        # self.conv_p = nn.Sequential(nn.Conv2d(1,64,(1,5)),
        #                             nn.MaxPool2d(()))
        self.bns_p = []
        for i in range(yt_layer_num+1):
            bni_p = []
            for j in range(5):
                bn = nn.BatchNorm1d(self.seq_len, momentum=0.5)
                setattr(self, 'bn_p%d_%d'%(i, j), bn)
                self._set_init(bn)
                bni_p.append(bn)
            self.bns_p.append(bni_p)
        self.bns_q = []
        for i in range(yt_layer_num + 1):
            bni_q = []
            for j in range(5):
                bn = nn.BatchNorm1d(self.seq_len, momentum=0.5)
                setattr(self, 'bn_q%d_%d' % (i, j), bn)
                self._set_init(bn)
                bni_q.append(bn)
            self.bns_q.append(bni_q)

    def _set_init(self, layer):
        init.normal(layer.weight, mean=0., std=.1)
        init.constant(layer.bias, 1.)

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

    def _make_layer(self, input_dim):
        lstm = nn.GRU(input_dim, self.hidden_dim, num_layers=2, batch_first=True,dropout=0.5, bidirectional=True)
        linear1 = nn.Linear(input_dim, self.output_dim)
        linear2 = nn.Linear(input_dim, self.output_dim)
        linear0 = nn.Linear(self.embedding_dim, self.output_dim)
        return lstm, linear1, linear2, linear0

    def _make_ytlayer(self, input_dim):
        lstm = nn.GRU(input_dim, self.hidden_dim, num_layers=2, batch_first=True,dropout=0.5, bidirectional=True)
        linear1 = nn.Linear(input_dim, self.output_dim)
        linear2 = nn.Linear(input_dim, self.output_dim)
        linear0 = nn.Linear(self.embedding_dim, self.output_dim)
        # conv = nn.Sequential(nn.Conv2d(1,2,5),
        #                         nn.ReLU(inplace=False),
        #                         nn.MaxPool2d((2,2)),
        #                         nn.Conv2d(2, 4, 4),
        #                         nn.ReLU(inplace=False),
        #                         nn.MaxPool2d((2, 2)))
        # conv = nn.Sequential(nn.Conv2d(1, 2, 5),
        #                      nn.ReLU(inplace=False),
        #                      nn.MaxPool2d((2, 2)),
        #                      nn.Conv2d(2, 4, 5),
        #                      nn.ReLU(inplace=False),
        #                      nn.MaxPool2d((2, 2)),
        #                      nn.Conv2d(4, 4, 3))
        return lstm, linear1, linear2, linear0

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
            tr_w_embedding_flags = t.mul(tr_w_embedding1, flags)
        # print('tr_w_embedding_flags:', tr_w_embedding_flags.shape)
        input_char = input_char.view([batch_size, seq_len*word_len])
        c_embedding = self.c_embedding(input_char)  # input_char:[batch_size, seq_len*word_len]
        c_embedding = c_embedding.view([batch_size, seq_len, word_len, -1])
        c_embedding = F.max_pool2d(c_embedding, kernel_size=(c_embedding.size()[-2], 1), stride=1)
        # print(c_embedding.shape)
        c_embedding = c_embedding.squeeze(2)  # c_embedding:[batch_size, seq_len, char_dim]
        multi_embedding = t.cat([tr_w_embedding, c_embedding, flags], 2)
        if self.tr_w_emb_dim_flag != 0:
            multi_embedding = t.cat([tr_w_embedding, tr_w_embedding_flags, c_embedding, flags], 2)
        return multi_embedding

    def _ytlayer(self, input_p, input_q, i):
        batch_size, seq_len, emb_dim = input_p.shape
        # print('seq_len:', seq_len, 'emb_dim:', emb_dim)
        if emb_dim == self.embedding_dim: ##n51
            self.res_p = input_p
            self.res_q = input_q
            output_lstm_p, _ = self.layer0_lstm_p(input_p)
            # print('input_p_pack:', input_p_pack.shape, output_lstm_p.shape)
            output_linear1_p = F.relu(self.layer0_linear1_p(input_p))
            res_p = self.layer0_linear0_p(input_p)
            output_lstm_q, _ = self.layer0_lstm_q(input_q)
            res_q = self.layer0_linear0_q(input_q)
            output_linear1_q = F.relu(self.layer0_linear1_q(input_q))
            output_attention_p, output_attention_q = self._attention_block(input_p, input_q)
            # conv = self._cos_block(input_p, input_q)
            # conv_p = self.conv_p(conv)
            # conv_p = conv_p.view(conv_p.size()[0], 1, -1)
            # conv_p = conv_p.repeat(1, self.seq_len, 1)
            # conv_q = self.conv_q(conv)
            # conv_q = conv_q.view(conv_q.size()[0], 1, -1)
            # conv_q = conv_q.repeat(1, self.seq_len, 1)
            output_linear2_p = F.relu(self.layer0_linear2_p(output_attention_p))
            output_linear2_q = F.relu(self.layer0_linear2_q(output_attention_q))
            # print(output_lstm_p.shape, output_linear2_p.shape, output_linear1_p.shape, self.res_p.shape)
            output_p = t.cat([output_lstm_p, output_linear2_p, output_linear1_p, res_p], 2)
            output_q = t.cat([output_lstm_q, output_linear2_q, output_linear1_q, res_q], 2)
            # output_p = t.cat([output_lstm_p, conv_p], 2)
            # output_q = t.cat([output_lstm_q, conv_q], 2)
        else: ## emb_dim = 800
            layer1_lstm_p, layer1_linear1_p, layer1_linear2_p, layer1_linear0_p = self.yt_layers_p[i*4],self.yt_layers_p[i*4+1],self.yt_layers_p[i*4+2],self.yt_layers_p[i*4+3]
            layer1_lstm_q, layer1_linear1_q, layer1_linear2_q, layer1_linear0_q = self.yt_layers_q[i*4],self.yt_layers_q[i*4+1],self.yt_layers_q[i*4+2],self.yt_layers_p[i*4+3]
            output_lstm_p, _ = layer1_lstm_p(input_p)
            output_linear1_p = F.relu(layer1_linear1_p(input_p))
            output_lstm_q, _ = layer1_lstm_q(input_q)
            output_linear1_q = F.relu(layer1_linear1_q(input_q))
            output_attention_p, output_attention_q = self._attention_block(input_p, input_q)
            output_linear2_p = F.relu(layer1_linear2_p(output_attention_p))
            output_linear2_q = F.relu(layer1_linear2_q(output_attention_q))
            res_p = layer1_linear0_p(self.res_p)
            res_q = layer1_linear0_q(self.res_q)
            # conv = self._cos_block(input_p, input_q)
            # conv_p = layer1_conv_p(conv)
            # conv_p = conv_p.view(conv_p.size()[0], 1, -1)
            # conv_p = conv_p.repeat(1, self.seq_len, 1)
            # conv_q = layer1_conv_q(conv)
            # conv_q = conv_q.view(conv_q.size()[0], 1, -1)
            # conv_q = conv_q.repeat(1, self.seq_len, 1)
            output_p = t.cat([output_lstm_p, output_linear2_p, output_linear1_p, res_p], 2)
            output_q = t.cat([output_lstm_q, output_linear2_q, output_linear1_q, res_q], 2)
            # output_p = t.cat([output_lstm_p, conv_p], 2)
            # output_q = t.cat([output_lstm_q, conv_q], 2)
        return output_p, output_q

    def _normalization(self, input, bni):
        in_0 = bni[0](input[:,:,:400])
        # print('len(bn):', len(bni))
        for i in range(len(bni)-2):
            # print('aaa')
            in_1 = bni[i+1](input[:,:,400+i*100:500+i*100])
            in_0 = t.cat([in_0, in_1], 2)
        out = bni[len(bni)-1](in_0)
        return out

    def _fclayer(self, p, q):
        p = self.p_fc0(p)
        q = self.q_fc0(q)
        p_attention, q_attention = self._attention_block(p, q)
        fc = t.cat([p, q, p_attention, q_attention, p-q, p+q], 2)  ## [batch_size, p_len, 300]
        batch_size, seq_len, hidden_dim = fc.shape
        output = self.fc(fc.view([batch_size, seq_len*hidden_dim]))
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
        multi_embedding_p = self.bn_emb_p(multi_embedding_p)
        multi_embedding_q - self.bn_emb_q(multi_embedding_q)
        out_p = []
        out_q = []
        cos = []
        p, q = self._ytlayer(multi_embedding_p, multi_embedding_q, None)
        p = self._normalization(p, self.bns_p[0])
        q = self._normalization(q, self.bns_q[0])
        out_p.append(p)
        out_q.append(q)
        cos.append(self._cos_block(p, q))
        # print('b:',p.shape, q.shape)
        for i in range(self.layer_num):
            # print(i)
            p, q = self._ytlayer(p, q, i)
            p = self._normalization(p, self.bns_p[i+1])
            q = self._normalization(q, self.bns_q[i+1])
            out_p.append(p)
            out_q.append(q)
            cos.append(self._cos_block(p, q))
        # print('c:',p.shape, q.shape)
        # output = self._fclayer(p, q)
        # output = t.softmax(output, 1)
        return out_p, out_q, cos
