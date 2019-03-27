import torch as t
import torch.nn as nn
import model_pytorch_2 as mp
# import model_20190311 as mp
import numpy as np
import pickle
from tqdm import tqdm
import build_dataset_pytorch as build_dataset_pytorch
from torch.utils.data import DataLoader
import data_helper

def train(use_gpu, model_path, lr, epochs, model_prefix, batch_size):
    dataset = build_dataset_pytorch.QuoraDataset_train('data/train_data_pad.pkl','data/train_data_char_pad.pkl',
                                                       'data/train_data_flags_pad.pkl')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=1)
    mm = mp.MatchModel(vocab_size=100000,tr_w_embed_dim=300,tr_w_emb_dim_flag=100,char_size=50,char_embed_dim=20,output_dim=100,
                       hidden_dim=200,use_gpu=use_gpu,yt_layer_num=1)
    # for name, param in mm.named_parameters():
    #     print(name, param.size())
    optimizer = t.optim.Adam(mm.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    t.manual_seed(2)
    if model_path:
        mm.load_state_dict(t.load(model_path))
    if use_gpu:
        # device_ids = [1,2]
        mm.cuda()
        # mm = nn.DataParallel(mm, device_ids = device_ids)
        # optimizer = nn.DataParallel(optimizer, device_ids = device_ids)
        criterion.cuda()
        t.cuda.manual_seed_all(2)
    # loss_meter = meter.AverageValueMeter()

    with open('data/vector_n.pkl', 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word_vec = t.from_numpy(np.array(data['word_vec']))

    def data_convert(*data):
        datas = []
        for i in data:
            datas.append(i.cuda())
        return datas

    max_test = 0
    max_dev = 0
    for e in range(0, epochs):
        correct_all = 0
        data_num = 0
        # loss_meter.reset()
        print("----- Epoch {}/{} -----".format(e + 1, epochs))
        # optimizer.zero_grad()
        for i, data_ in tqdm(enumerate(dataloader), desc="Training"):
            p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags = data_
            if use_gpu:
                [p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags] = \
                    data_convert(p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags)
            optimizer.zero_grad()
            p_input = [p_inputs, p_inputs_char, p_flags]
            q_input = [q_inputs, q_inputs_char, q_flags]
            output = mm(100, p_input, q_input)
            loss = criterion(output, t.argmax(targets, 1))
            # print(loss)
            # print(output)
            if use_gpu:
                output = output.cpu()
                targets = targets.cpu()
            correct_num = (t.argmax(output, 1) == t.argmax(targets, 1)).numpy()
            correct_sum = correct_num.sum()
            # print('correct_sum:', correct_num)
            accuracy = correct_sum/len(correct_num)
            data_num = data_num+len(correct_num)
            correct_all = correct_all+correct_sum
            loss.backward()
            optimizer.step()
            # loss_meter.add(loss.data[0])
            if (i+1)%600 == 0:
                tqdm.write("----- Epoch %d Batch %d -- Loss %.2f -- Acc %.3f" % (e+1, i+1, loss, accuracy))
        tqdm.write("----- epoch %d -- Loss %.2f -- Acc %.3f" % (e + 1, loss, correct_all/data_num))
        t.save(mm.state_dict(), '%s_%s.pth' % (model_prefix, e))
        dev_acc = dev(True, 50, '%s_%s.pth' % (model_prefix, e))
        test_acc = test(True, 50, '%s_%s.pth' % (model_prefix, e))
        max_dev = max(max_dev, dev_acc)
        max_test = max(max_test, test_acc)
    print('max test acc:', max_test, '\nmax dev acc:', max_dev)

def test(use_gpu, batch_size, model_path):
    dataset = build_dataset_pytorch.QuoraDataset_train('data/test_data_pad.pkl', 'data/test_data_char_pad.pkl',
                                                       'data/test_data_flags_pad.pkl')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=1)
    mm2 = mp.MatchModel(vocab_size=100000,tr_w_embed_dim=300,tr_w_emb_dim_flag=100,char_size=50,char_embed_dim=20,output_dim=100,
                        hidden_dim=200,use_gpu=use_gpu,yt_layer_num=1,mode='test')
    mm2.load_state_dict(t.load(model_path))
    mm2.eval()
    with open('data/vector_n.pkl', 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word_vec = t.from_numpy(np.array(data['word_vec']))
    if use_gpu: mm2.cuda()
    def data_convert(*data):
        datas = []
        for i in data:
            datas.append(i.cuda())
        return datas
    data_num = 0
    correct_all = 0
    for i, data in enumerate(dataloader):
        p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags = data
        if use_gpu:
            [p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec] = \
                data_convert(p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec)
        p_input = [p_inputs, p_inputs_char, word_vec, p_flags]
        q_input = [q_inputs, q_inputs_char, q_flags]
        output = mm2(100, p_input, q_input)
        if use_gpu:
            output = output.cpu()
            targets = targets.cpu()
        # print(output, targets)
        correct_num = (t.argmax(output, 1) == t.argmax(targets, 1)).numpy()
        correct_sum = correct_num.sum()
        data_num = data_num+len(correct_num)
        correct_all = correct_all+correct_sum
    print("-- Test Acc %.3f" % (correct_all/data_num))
    return correct_all/data_num

def dev(use_gpu, batch_size, model_path):
    dataset = build_dataset_pytorch.QuoraDataset_train('data/dev_data_pad.pkl', 'data/dev_data_char_pad.pkl',
                                                       'data/dev_data_flags_pad.pkl')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=1)
    mm1 = mp.MatchModel(vocab_size=100000,tr_w_embed_dim=300,tr_w_emb_dim_flag=100,char_size=50,char_embed_dim=20,output_dim=100,
                        hidden_dim=200,use_gpu=use_gpu,yt_layer_num=1,mode='test')
    mm1.load_state_dict(t.load(model_path))
    mm1.eval()
    with open('data/vector_n.pkl', 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word_vec = t.from_numpy(np.array(data['word_vec']))
    if use_gpu: mm1.cuda()
    def data_convert(*data):
        datas = []
        for i in data:
            datas.append(i.cuda())
        return datas
    data_num = 0
    correct_all = 0
    for i, data in enumerate(dataloader):
        p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags = data
        if use_gpu:
            [p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec] = \
                data_convert(p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec)
        p_input = [p_inputs, p_inputs_char, word_vec, p_flags]
        q_input = [q_inputs, q_inputs_char, q_flags]
        output = mm1(100, p_input, q_input)
        if use_gpu:
            output = output.cpu()
            targets = targets.cpu()
        # print(output, targets)
        correct_num = (t.argmax(output, 1) == t.argmax(targets, 1)).numpy()
        correct_sum = correct_num.sum()
        data_num = data_num+len(correct_num)
        correct_all = correct_all+correct_sum
    print("-- Dev Acc %.3f" % (correct_all/data_num))
    return correct_all/data_num

def predict(key_sentence, batch_size, use_gpu, model_path):
    index_list = []
    print('build the sen_list...')
    sen_list, id_list = data_helper.build_q_input('data/sen2id.txt')
    eps_list = []
    part_i = 0
    part_i_len = 0
    brick_len = 5
    part_len = 10
    eps_len = 50
    p = ''
    # with open('data/vector_n.pkl', 'rb') as handle:
    #     data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
    #     word_vec = t.from_numpy(np.array(data['word_vec']))
    mm2 = mp.MatchModel(vocab_size=100000,tr_w_embed_dim=300,tr_w_emb_dim_flag=100,char_size=50,char_embed_dim=20,output_dim=100,
                       hidden_dim=200,use_gpu=use_gpu,yt_layer_num=1, mode='test')
    mm2.load_state_dict(t.load(model_path))
    mm2.eval()
    if use_gpu: mm2.cuda()
    def data_convert(*data):
        datas = []
        for i in data:
            datas.append(i.cuda())
        return datas
    while p != '-1' and len(eps_list)< eps_len:
        if index_list == []:
            p = key_sentence
        dataset = build_dataset_pytorch.QuoraDataset_predict(p, 'data/predict_data_pad.pkl',
                                                                 'data/predict_data_char_pad.pkl')
        dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=1)
        prob_temp_list = []
        index_temp_list = []
        for i, data in enumerate(dataloader):
            p_inputs, q_inputs, p_inputs_char, q_inputs_char, p_flags, q_flags, index = data
            if use_gpu:
                [p_inputs, q_inputs, p_inputs_char, q_inputs_char, p_flags, q_flags] = \
                    data_convert(p_inputs, q_inputs, p_inputs_char, q_inputs_char, p_flags, q_flags)
            p_input = [p_inputs, p_inputs_char, p_flags]
            q_input = [q_inputs, q_inputs_char, q_flags]
            output = mm2(100, p_input, q_input)
            output = t.softmax(output, 1)
            if use_gpu:
                output = output.cpu()
            for ii, prob_ii in enumerate(output):
                if prob_ii[1].data>0.8:  ## miu
                    print(prob_ii)
                    prob_temp_list.append(prob_ii[1].data)
                    index_temp_list.append(index[ii].item())
                # print(prob_temp_list)
        if index_list == []:
            index_list = index_temp_list
            if index_list == []:break
            eps_list.append(index_list[part_i])
            eps_list.append(index_list[part_i]+1)
            part_i_len+=2
            p = sen_list[index_list[part_i]+1]+key_sentence
            continue
        if prob_temp_list != []:
            # max_prob = max(t.Tensor(prob_temp_list))
            max_i = t.argmax(t.Tensor(prob_temp_list))
            max_index = index_temp_list[max_i]
            if max_index+1 not in eps_list:
                if (len(eps_list)>=brick_len and max_index+1-eps_list[-brick_len] != brick_len) or len(eps_list)<brick_len :
                    if part_i_len < part_len:
                        eps_list.append(max_index + 1)
                        p = sen_list[max_index + 1] + key_sentence
                        part_i_len+=1
                        continue
        if prob_temp_list == [] and len(eps_list)>brick_len and eps_list[-1]-eps_list[-brick_len]!= brick_len-1:
            if part_i_len != part_len:
                eps_list.append(eps_list[-1] + 1)
                p = sen_list[eps_list[-1] + 1] + key_sentence
                part_i_len += 1
                continue
        part_i += 1
        part_i_len = 0
        while part_i != len(index_list) and (index_list[part_i] in eps_list or index_list[part_i]+1 in eps_list):
            part_i += 1
            part_i_len = 0
        if part_i == len(index_list):break
        eps_list.append(index_list[part_i])
        eps_list.append(index_list[part_i]+1)
        p = sen_list[index_list[part_i]+1]+key_sentence
        part_i_len = part_i_len+2
    return eps_list

def predict_1filter(key_sentence, batch_size, use_gpu, model_path):
    index_list = []
    print('build the sen_list...')
    sen_list, id_list = data_helper.build_q_input('data/sen2id.txt')
    print(len(sen_list))
    eps_list = []
    part_i = 0
    part_i_len = 0
    brick_len = 5
    part_len = 10
    eps_len = 50
    p = ''
    # with open('data/vector_n.pkl', 'rb') as handle:
    #     data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
    #     word_vec = t.from_numpy(np.array(data['word_vec']))
    mm2 = mp.MatchModel(vocab_size=100000, tr_w_embed_dim=300, tr_w_emb_dim_flag=100, char_size=50, char_embed_dim=20,
                        output_dim=100,
                        hidden_dim=200, use_gpu=use_gpu, yt_layer_num=1, mode='test')
    mm2.load_state_dict(t.load(model_path))
    mm2.eval()
    mm3 = mp.MatchModel(vocab_size=100000, tr_w_embed_dim=300, tr_w_emb_dim_flag=100, char_size=50, char_embed_dim=20,
                        output_dim=100,
                        hidden_dim=200, use_gpu=use_gpu, yt_layer_num=1, mode='test')
    mm3.load_state_dict(t.load(model_path))
    mm3.eval()
    if use_gpu:
        mm2.cuda()
        mm3.cuda()

    def data_convert(*data):
        datas = []
        for i in data:
            datas.append(i.cuda())
        return datas

    def list2tensor(*data):
        datas = []
        for i in data:
            a = np.array(i)
            datas.append(t.from_numpy(a))
        return datas

    while p != '-1' and len(eps_list) < eps_len:
        print('eps_list:', eps_list)
        if index_list == []:
            p = key_sentence
        dataset = build_dataset_pytorch.QuoraDataset_predict(p, 'data/predict_data_pad.pkl',
                                                             'data/predict_data_char_pad.pkl')
        dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=1)
        prob_temp_list = []
        index_temp_list = []
        for i, data in enumerate(dataloader):
            p_inputs, q_inputs, p_inputs_char, q_inputs_char, p_flags, q_flags, index = data
            if use_gpu:
                [p_inputs, q_inputs, p_inputs_char, q_inputs_char, p_flags, q_flags] = \
                    data_convert(p_inputs, q_inputs, p_inputs_char, q_inputs_char, p_flags, q_flags)
            p_input = [p_inputs, p_inputs_char, p_flags]
            q_input = [q_inputs, q_inputs_char, q_flags]
            output = mm2(100, p_input, q_input)
            output = t.softmax(output, 1)
            if use_gpu:
                output = output.cpu()
            for ii, prob_ii in enumerate(output):
                if prob_ii[1].data > 0.8:  ## miu
                    print(prob_ii)
                    prob_temp_list.append(prob_ii[1].data)
                    index_temp_list.append(index[ii].item())
                # print(prob_temp_list)
        if index_list == []:
            index_list = index_temp_list
            if index_list == []: break
            p_ = []
            p_char_ = []
            q_ = []
            q_char_ = []
            print(len(prob_temp_list), len(index_temp_list))
            for x in range(len(prob_temp_list)):
                print(x, index_temp_list[x])
                if index_temp_list[x]+1 == len(sen_list):
                    p_.append(p_[-1])
                    p_char_.append(p_char_[-1])
                    q_.append(q_[-1])
                    q_char_.append(q_char_[-1])
                    continue
                p_i = sen_list[index_temp_list[x] + 1] + key_sentence
                _, q_i, _, q_i_char, _, _, _ = dataset.__getitem__(index_temp_list[x]+1)
                p_i, p_i_char = data_helper.sentence_p_process(p_i)
                p_.append(p_i)
                p_char_.append(p_i_char)
                q_.append(q_i.numpy())
                q_char_.append(q_i_char.numpy())

            # print(np.array(p_).shape, np.array(p_char_).shape, np.array(q_).shape, np.array(q_char_).shape)
            p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad = data_helper.sentence_p_q_process2(p_,
                                                                                                               p_char_,
                                                                                                               q_,
                                                                                                               q_char_)
            [p_pad, p_char_pad, q_pad, q_char_pad] = list2tensor(p_pad, p_char_pad, q_pad, q_char_pad)
            p_flags_pad = t.unsqueeze(t.from_numpy(np.array(p_flags_pad).astype(np.float32)), 2)
            q_flags_pad = t.unsqueeze(t.from_numpy(np.array(q_flags_pad).astype(np.float32)), 2)
            if use_gpu:
                [p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad] = \
                    data_convert(p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad)
            p_input = [p_pad, p_char_pad, p_flags_pad]
            q_input = [q_pad, q_char_pad, q_flags_pad]
            output = mm3(len(p_), p_input, q_input)
            output = t.softmax(output, 1)
            if use_gpu:
                output = output.cpu()
            max_i = t.argmax(output[:, 1].data)
            max_index = index_temp_list[max_i]
            eps_list.append(max_index)
            eps_list.append(max_index + 1)
            part_i_len = 2
            p = sen_list[index_list[max_i] + 1] + key_sentence
            continue
        if prob_temp_list != []:
            # max_prob = max(t.Tensor(prob_temp_list))
            p_n = []
            p_char_n = []
            q_n = []
            q_char_n = []
            for x in range(len(prob_temp_list)):
                if index_temp_list[x]+1 == len(sen_list):
                    p_n.append(p_n[-1])
                    p_char_n.append(p_char_n[-1])
                    q_n.append(q_n[-1])
                    q_char_n.append(q_char_n[-1])
                    continue
                p_i = sen_list[index_temp_list[x] + 1] + key_sentence
                _, q_i, _, q_i_char, _, _, _ = dataset.__getitem__(index_temp_list[x]+1)
                p_i, p_i_char = data_helper.sentence_p_process(p_i)
                p_n.append(p_i)
                p_char_n.append(p_i_char)
                q_n.append(q_i.numpy())
                q_char_n.append(q_i_char.numpy())
            # print(np.array(p_).shape, np.array(p_char_).shape, np.array(q_).shape, np.array(q_char_).shape)
            p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad = data_helper.sentence_p_q_process2(p_n,
                                                                                                               p_char_n,
                                                                                                               q_n,
                                                                                                               q_char_n)
            [p_pad, p_char_pad, q_pad, q_char_pad] = list2tensor(p_pad, p_char_pad, q_pad, q_char_pad)
            p_flags_pad = t.unsqueeze(t.from_numpy(np.array(p_flags_pad).astype(np.float32)), 2)
            q_flags_pad = t.unsqueeze(t.from_numpy(np.array(q_flags_pad).astype(np.float32)), 2)
            if use_gpu:
                [p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad] = \
                    data_convert(p_pad, p_char_pad, p_flags_pad, q_pad, q_char_pad, q_flags_pad)
            p_input = [p_pad, p_char_pad, p_flags_pad]
            q_input = [q_pad, q_char_pad, q_flags_pad]
            output = mm3(len(p_n), p_input, q_input)
            # print('output.shape:', output.shape)
            output = t.softmax(output, 1)
            if use_gpu:
                output = output.cpu()
            max_i = t.argmax(output.data, dim=0)
            # max_i = t.argmax(t.Tensor(prob_temp_list))
            # print('max_i:', max_i, 'len(prob_list):', len(prob_temp_list))
            max_index = index_temp_list[max_i[1]]
            if max_index + 1 not in eps_list:
                if (len(eps_list) >= brick_len and max_index + 1 - eps_list[-brick_len] != brick_len) or (len(eps_list) < brick_len and eps_list[-1] - eps_list[0]+1 != brick_len):
                    if part_i_len < part_len:
                        eps_list.append(max_index + 1)
                        p = sen_list[max_index + 1] + key_sentence
                        part_i_len = 1
                        continue
        if prob_temp_list == [] and ((len(eps_list) > brick_len and eps_list[-1] - eps_list[-brick_len] != brick_len - 1) or (len(eps_list) < brick_len and eps_list[-1] - eps_list[0]+1 != brick_len)):
            if part_i_len != part_len:
                eps_list.append(eps_list[-1] + 1)
                p = sen_list[eps_list[-1] + 1] + key_sentence
                part_i_len += 1
                continue
        part_i += 1
        part_i_len = 0
        while part_i < len(index_list) and (index_list[part_i] in eps_list or index_list[part_i] + 1 in eps_list):
            part_i += 1
            part_i_len = 0
        if part_i >= len(index_list): break
        eps_list.append(index_list[part_i])
        eps_list.append(index_list[part_i] + 1)
        p = sen_list[index_list[part_i] + 1] + key_sentence
        part_i_len = part_i_len + 2
    return eps_list



if __name__ == '__main__':
    # train(use_gpu=True, model_path=None, lr=0.0001, epochs=10, model_prefix='model/mm-conv', batch_size=50)
    # test(use_gpu=True, batch_size=200, model_path='model/mm-0_9.pth')
    # index = predict(key_sentence='i love you.', batch_size=100, use_gpu=True, model_path='model/mm-important1_14.pth')
    index = predict_1filter(key_sentence='i love you.', batch_size=100, use_gpu=True, model_path='model/mm-important1_13.pth')####
    # index = predict_1filter(key_sentence='i love you.', batch_size=100, use_gpu=True, model_path='model/mm-20190311_2.pth')
    print(index)
