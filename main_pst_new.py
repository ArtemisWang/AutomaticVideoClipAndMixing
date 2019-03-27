import torch as t
import torch.nn as nn
import model_pst_new as mp
import numpy as np
import pickle
from tqdm import tqdm
import build_dataset_pytorch as build_dataset_pytorch
from torch.utils.data import DataLoader

def train(use_gpu, model_path, lr, epochs, model_prefix, batch_size):
    dataset = build_dataset_pytorch.QuoraDataset_train('data/train_data_pad.pkl','data/train_data_char_pad.pkl',
                                                       'data/train_data_flags_pad.pkl')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    mm = mp.MatchModel(vocab_size=100000,tr_w_embed_dim=300,tr_w_emb_dim_flag=100,char_size=50,char_embed_dim=20,output_dim=100,
                       hidden_dim=200,use_gpu=use_gpu,yt_layer_num=2)
    # for name, param in mm.named_parameters():
    #     print(name, param.size())
    optimizer = t.optim.Adam(mm.parameters(), lr = lr, weight_decay=0.0001)
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

    max_dev = 0
    max_test = 0
    for e in range(0, epochs):
        correct_all = 0
        data_num = 0
        # loss_meter.reset()
        print("----- Epoch {}/{} -----".format(e + 1, epochs))
        # optimizer.zero_grad()
        for i, data_ in tqdm(enumerate(dataloader), desc="Training"):
            p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags = data_
            if use_gpu:
                [p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec] = \
                    data_convert(p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec)
            optimizer.zero_grad()
            p_input = [p_inputs, p_inputs_char, word_vec, p_flags]
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
        #     if (i + 1) % 600 == 0:
        #         tqdm.write("----- Epoch %d Batch %d -- Loss %.4f -- Acc %.3f" % (e + 1, i + 1, loss, accuracy))
        #         t.save(mm.state_dict(), '%s_%s.pth' % (model_prefix, e))
        #         dev_acc = dev(True, 50, '%s_%s.pth' % (model_prefix, e))
        #         test_acc = test(True, 50, '%s_%s.pth' % (model_prefix, e))
        #         max_dev = max(max_dev, dev_acc)
        #         max_test = max(max_test, test_acc)
        #     optimizer.step()
        # tqdm.write("----- epoch %d -- Loss %.4f -- Acc %.3f\n" % (e + 1, loss, correct_all / data_num))
        # print('max test acc:', max_test, '\nmax dev acc:', max_dev)
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

def dev(use_gpu, batch_size, model_path):
    dataset = build_dataset_pytorch.QuoraDataset_train('data/dev_data_pad.pkl', 'data/dev_data_char_pad.pkl',
                                                       'data/dev_data_flags_pad.pkl')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    mm1 = mp.MatchModel(vocab_size=100000,tr_w_embed_dim=300,tr_w_emb_dim_flag=100,char_size=50,char_embed_dim=20,output_dim=100,
                        hidden_dim=200,use_gpu=use_gpu,yt_layer_num=2,mode='test')
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

def test(use_gpu, batch_size, model_path):
    dataset = build_dataset_pytorch.QuoraDataset_train('data/test_data_pad.pkl', 'data/test_data_char_pad.pkl',
                                                       'data/test_data_flags_pad.pkl')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    mm2 = mp.MatchModel(vocab_size=100000,tr_w_embed_dim=300,tr_w_emb_dim_flag=100,char_size=50,char_embed_dim=20,output_dim=100,
                        hidden_dim=200,use_gpu=use_gpu,yt_layer_num=2,mode='test')
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

if __name__ == '__main__':
    train(True, None, 0.0001, 10, 'model/mm-pstn', 100)
    # test(True, 200, 'model/mm-2_9.pth')
