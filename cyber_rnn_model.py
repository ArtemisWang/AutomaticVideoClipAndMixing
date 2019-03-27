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
        self.use_init_emb = False
        self.p_mask = None
        self.q_mask = None
        self.mask = False
        if self.use_init_emb:
            with open('data/vector_n.pkl', 'rb') as handle:
                data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
                word_vec = t.from_numpy(np.array(data['word_vec']))
            if use_gpu:
                word_vec = word_vec.cuda()
            self.tr_w_embedding.weight = nn.Parameter(word_vec)
        if self.tr_w_emb_dim_flag != 0:
            self.tr_w_embedding1 = nn.Embedding(vocab_size, tr_w_emb_dim_flag)
        self.c_embedding = nn.Embedding(char_size, char_embed_dim)
        self.embedding_dim = tr_w_embed_dim + tr_w_emb_dim_flag+char_embed_dim+1 ## 651
        self.output_dim = output_dim  ## 200
        self.hidden_dim = hidden_dim  ## 200
        self.dropout = 0.1
        self.rnn_p = nn.RNN(self.embedding_dim, self.hidden_dim, num_layers=2, batch_first=True,dropout=self.dropout, bidirectional=True)
        self.rnn_q = nn.RNN(self.embedding_dim, self.hidden_dim, num_layers=2, batch_first=True,dropout=self.dropout, bidirectional=True)
        self.linear = nn.Sequential(nn.Linear(30*2*hidden_dim, 2000),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(2000, 200),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(200,2))


    def _attention_block(self, p_input, q_input):
        eps = 1e-6
        p_norm = t.sqrt(t.clamp(t.sum(t.pow(p_input, 2), -1), min = eps))
        q_norm = t.sqrt(t.clamp(t.sum(t.pow(q_input, 2), -1), min = eps))
        p_input_norm = t.div(p_input, t.unsqueeze(p_norm, 2))
        q_input_norm = t.div(q_input, t.unsqueeze(q_norm, 2))
        cosine_matrix = t.bmm(q_input_norm, p_input_norm.transpose(1,2))  ## [batch_size, q_len, p_len]
        q_attention = t.bmm(cosine_matrix, p_input)  ## [batch_size, q_len, 100]
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
        lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=2, batch_first=True,dropout=self.dropout, bidirectional=True)
        linear1 = nn.Linear(input_dim, self.output_dim)
        linear2 = nn.Linear(input_dim, self.output_dim)
        return lstm, linear1, linear2

    def _make_ytlayer(self, input_dim):
        lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=2, batch_first=True,dropout=self.dropout, bidirectional=True)
        linear1 = nn.Linear(input_dim, self.output_dim)
        linear2 = nn.Linear(input_dim, self.output_dim)
        conv = nn.Sequential(nn.Conv2d(1,16,5),
                             nn.Dropout(p=self.dropout),
                             nn.ReLU(inplace=False),
                             nn.MaxPool2d((2,2)),
                             nn.Conv2d(16, 4, 4),
                             nn.Dropout(p=self.dropout),
                             nn.ReLU(inplace=False),
                             nn.MaxPool2d((2, 2)))
        # conv = nn.Sequential(nn.Conv2d(1, 2, 5),
        #                      nn.ReLU(inplace=False),
        #                      nn.MaxPool2d((2, 2)),
        #                      nn.Conv2d(2, 4, 5),
        #                      nn.ReLU(inplace=False),
        #                      nn.MaxPool2d((2, 2)),
        #                      nn.Conv2d(4, 4, 3))
        return lstm, linear1, linear2, conv

    def _build_multi_layers(self, layer_num, input_dim):
        yt_layers = []
        for i in range(layer_num):
            x = self._make_ytlayer(input_dim)
            yt_layers = yt_layers+[*x]
        return yt_layers

    def _multi_emb(self, input, input_char, flags=None):
        batch_size, seq_len, word_len = input_char.data.shape
        # if self.use_gpu: self.fix_w_embedding = self.fix_w_embedding.cuda()
        tr_w_embedding = self.tr_w_embedding(input).float()
        if self.tr_w_emb_dim_flag != 0:
            tr_w_embedding1 = self.tr_w_embedding1(input).float()
            # flags_squ = t.squeeze(flags)
            # print('tr_w_emb:', tr_w_embedding1.shape, 'flags:', flags.shape)
            tr_w_embedding_flags = t.mul(tr_w_embedding1, t.softmax(flags, 1))
        # print('tr_w_embedding_flags:', tr_w_embedding_flags.shape)
        input_char = input_char.view([batch_size, seq_len*word_len])
        c_embedding = self.c_embedding(input_char)  # input_char:[batch_size, seq_len*word_len]
        c_embedding = c_embedding.view([batch_size, seq_len, word_len, -1])
        c_embedding = F.max_pool2d(c_embedding, kernel_size=(c_embedding.size()[-2], 1), stride=1)
        c_embedding = c_embedding.view([batch_size, seq_len, -1])  # c_embedding:[batch_size, seq_len, char_dim]
        multi_embedding = t.cat([tr_w_embedding, c_embedding, flags], 2)
        if self.tr_w_emb_dim_flag != 0:
            multi_embedding = t.cat([tr_w_embedding, tr_w_embedding_flags, c_embedding, flags], 2)
        return multi_embedding

    def _ytlayer(self, input_p, input_q, i):
        batch_size, seq_len, emb_dim = input_p.shape
        # print('seq_len:', seq_len, 'emb_dim:', emb_dim)
        if emb_dim == self.embedding_dim: ##n51
            output_lstm_p, _ = self.layer0_lstm_p(input_p)
            # print('input_p_pack:', input_p_pack.shape, output_lstm_p.shape)
            self.res_p = self.layer0_linear1_p(input_p)
            output_linear1_p = F.relu(F.dropout(self.res_p, p=self.dropout))
            output_lstm_q, _ = self.layer0_lstm_q(input_q)
            self.res_q = self.layer0_linear1_q(input_q)
            output_linear1_q = F.relu(F.dropout(self.res_q, p=self.dropout))
            output_attention_p, output_attention_q = self._attention_block(input_p, input_q)
            conv = self._cos_block(input_p, input_q)
            conv_p = self.conv_p(conv)
            conv_p = conv_p.view(conv_p.size()[0], 1, -1)
            conv_p = conv_p.repeat(1, 30, 1)
            conv_q = self.conv_q(conv)
            conv_q = conv_q.view(conv_q.size()[0], 1, -1)
            conv_q = conv_q.repeat(1, 30, 1)
            output_linear2_p = F.relu(self.layer0_linear2_p(output_attention_p))
            output_linear2_q = F.relu(self.layer0_linear2_q(output_attention_q))
            # print(output_lstm_p.shape, output_linear2_p.shape, output_linear1_p.shape, self.res_p.shape)
            output_p = t.cat([output_lstm_p, output_linear2_p, output_linear1_p, self.res_p, conv_p], 2)
            output_q = t.cat([output_lstm_q, output_linear2_q, output_linear1_q, self.res_q, conv_q], 2)
            # output_p = t.cat([output_lstm_p, conv_p], 2)
            # output_q = t.cat([output_lstm_q, conv_q], 2)
        else: ## emb_dim = 800
            layer1_lstm_p, layer1_linear1_p, layer1_linear2_p, layer1_conv_p = self.yt_layers_p[i*4],self.yt_layers_p[i*4+1],self.yt_layers_p[i*4+2],self.yt_layers_p[i*4+3]
            layer1_lstm_q, layer1_linear1_q, layer1_linear2_q, layer1_conv_q = self.yt_layers_q[i*4],self.yt_layers_q[i*4+1],self.yt_layers_q[i*4+2],self.yt_layers_q[i*4+3]
            output_lstm_p, _ = layer1_lstm_p(input_p)
            output_linear1_p = F.relu(F.dropout(layer1_linear1_p(input_p), p=self.dropout))
            output_lstm_q, _ = layer1_lstm_q(input_q)
            output_linear1_q = F.relu(F.dropout(layer1_linear1_q(input_q), p=self.dropout))
            output_attention_p, output_attention_q = self._attention_block(input_p, input_q)
            output_linear2_p = F.relu(layer1_linear2_p(output_attention_p))
            output_linear2_q = F.relu(layer1_linear2_q(output_attention_q))
            conv = self._cos_block(input_p, input_q)
            conv_p = layer1_conv_p(conv)
            conv_p = conv_p.view(conv_p.size()[0], 1, -1)
            conv_p = conv_p.repeat(1, 30, 1)
            conv_q = layer1_conv_q(conv)
            conv_q = conv_q.view(conv_q.size()[0], 1, -1)
            conv_q = conv_q.repeat(1, 30, 1)
            output_p = t.cat([output_lstm_p, output_linear2_p, output_linear1_p, self.res_p, conv_p], 2)
            output_q = t.cat([output_lstm_q, output_linear2_q, output_linear1_q, self.res_q, conv_q], 2)
            # output_p = t.cat([output_lstm_p, conv_p], 2)
            # output_q = t.cat([output_lstm_q, conv_q], 2)
        return output_p, output_q

    def _mask(self, p, q):
        if self.mask:
            p = self.p_mask*p
            q = self.q_mask*q
        return p, q


    def _fclayer(self, p, q):
        p = self.p_fc0(p)
        q = self.q_fc0(q)
        p_attention, q_attention = self._attention_block(p, q)
        fc = t.cat([p, q, p_attention, q_attention, p-q, p+q], 2)  ## [batch_size, p_len, 300]
        batch_size, seq_len, hidden_dim = fc.shape
        output = self.fc(fc.view([batch_size, seq_len*hidden_dim]))
        return output

    def forward(self, output_dim, p_input, q_input):
        [sample_batch_p, sample_batch_char_p, sample_batch_flags_p] = p_input
        [sample_batch_q, sample_batch_char_q, sample_batch_flags_q] = q_input
        self.p_mask = t.unsqueeze((sample_batch_p!=0).float(), 2)
        self.q_mask = t.unsqueeze((sample_batch_q!=0).float(), 2)
        if self.mode == 'train':
            multi_embedding_p = self._multi_emb(V(sample_batch_p), V(sample_batch_char_p), V(sample_batch_flags_p))
            multi_embedding_q = self._multi_emb(V(sample_batch_q), V(sample_batch_char_q), V(sample_batch_flags_q))
        else:
            with t.no_grad():
                multi_embedding_p = self._multi_emb(V(sample_batch_p), V(sample_batch_char_p),
                                                V(sample_batch_flags_p))
                multi_embedding_q = self._multi_emb(V(sample_batch_q), V(sample_batch_char_q),
                                                V(sample_batch_flags_q))
        # return multi_embedding_p
        multi_embedding_p, multi_embedding_q = self._mask(multi_embedding_p, multi_embedding_q)
        p, _ = self.rnn_p(multi_embedding_p)
        q, _ = self.rnn_p(multi_embedding_q)
        # y = t.cat([p, q], -1)
        y = t.reshape(q, [sample_batch_p.size()[0],-1])
        y = self.linear(y)

        return y
