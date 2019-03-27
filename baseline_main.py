import torch as t
import torch.nn as nn
import baseline_model as mp
import numpy as np
import pickle
from tqdm import tqdm
import dataset as build_dataset_pytorch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

def train(use_gpu, model_path, lr, epochs, model_prefix, batch_size):
    dataset = build_dataset_pytorch.QuoraDataset_train('data/train_data_pad.pkl','data/train_data_char_pad.pkl',
                                                       'data/train_data_flags_pad.pkl')
    # weights = [5 if t.argmax(targets)==1 else 1 for p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags in dataset]
    # sampler = WeightedRandomSampler(weights, num_samples=380000, replacement=False)
    # dataloader = DataLoader(dataset, batch_size, num_workers=0, sampler=sampler)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    # for j in range(2):
    #     for i in dataloader:
    #         p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags = i
    #         print(p_inputs)
    mm = mp.MatchModel(vocab_size=100000,emb_dim=100,use_gpu=use_gpu,mode='train',batch_size=batch_size)
    # for name, param in mm.named_parameters():
    #     print(name, param.size())
    optimizer = t.optim.Adam(mm.parameters(), lr = lr, weight_decay=0)
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
    learning_rate = lr
    for e in range(3, epochs):
        correct_all = 0
        data_num = 0
        # loss_meter.reset()
        print("----- Epoch {}/{} -----".format(e + 1, epochs))
        # optimizer.zero_grad()
        for i, data_ in tqdm(enumerate(dataloader), desc="Training"):
            p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, p_inputs_length, q_inputs_length = data_
            if use_gpu:
                [p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec] = \
                    data_convert(p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec)
            optimizer.zero_grad()
            p_input = [p_inputs, p_inputs_char, p_flags, p_inputs_length]
            q_input = [q_inputs, q_inputs_char, q_flags, q_inputs_length]
            output = mm(p_input, q_input)
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
            if (i + 1) % 600 == 0:
                tqdm.write("----- Epoch %d Batch %d -- Loss %.2f -- Acc %.3f" % (e + 1, i + 1, loss, accuracy))
        tqdm.write("----- epoch %d -- Loss %.2f -- Acc %.3f" % (e + 1, loss, correct_all / data_num))
        t.save(mm.state_dict(), '%s_%s.pth' % (model_prefix, e))
        dev_acc = dev(True, batch_size, '%s_%s.pth' % (model_prefix, e))
        test_acc = test(True, batch_size, '%s_%s.pth' % (model_prefix, e))
        max_dev = max(max_dev, dev_acc)
        max_test = max(max_test, test_acc)
    print('max test acc:', max_test, '\nmax dev acc:', max_dev)

def dev(use_gpu, batch_size, model_path):
    dataset = build_dataset_pytorch.QuoraDataset_test('data/dev_data_pad.pkl', 'data/dev_data_char_pad.pkl',
                                                       'data/dev_data_flags_pad.pkl')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    mm1 = mp.MatchModel(vocab_size=100000,emb_dim=100,use_gpu=use_gpu,mode='test')
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
        p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, p_inputs_length, q_inputs_length = data
        if use_gpu:
            [p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec] = \
                data_convert(p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec)
        p_input = [p_inputs, p_inputs_char, p_flags, p_inputs_length]
        q_input = [q_inputs, q_inputs_char, q_flags, q_inputs_length]
        output = mm1(p_input, q_input)
        if use_gpu:
            output = output.cpu()
            targets = targets.cpu()
        # print(output, targets)
        correct_num = (t.argmax(output, 1) == t.argmax(targets, 1)).numpy()
        correct_sum = correct_num.sum()
        data_num = data_num+len(correct_num)
        correct_all = correct_all+correct_sum
    mm1.train()
    print("-- Dev Acc %.3f" % (correct_all/data_num))
    return correct_all/data_num

def test(use_gpu, batch_size, model_path):
    dataset = build_dataset_pytorch.QuoraDataset_test('data/test_data_pad.pkl', 'data/test_data_char_pad.pkl',
                                                       'data/test_data_flags_pad.pkl')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    mm2 = mp.MatchModel(vocab_size=100000,emb_dim=100,use_gpu=use_gpu,mode='test')
    mm2.load_state_dict(t.load(model_path))
    mm2.eval()
    if use_gpu: mm2.cuda()
    def data_convert(*data):
        datas = []
        for i in data:
            datas.append(i.cuda())
        return datas
    data_num = 0
    correct_all = 0
    for i, data in enumerate(dataloader):
        p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, p_inputs_length, q_inputs_length = data
        if use_gpu:
            [p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags] = \
                data_convert(p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags)

        p_input = [p_inputs, p_inputs_char, p_flags, p_inputs_length]
        q_input = [q_inputs, q_inputs_char, q_flags, q_inputs_length]
        output = mm2(p_input, q_input)
        if use_gpu:
            output = output.cpu()
            targets = targets.cpu()
        # print(output, targets)
        correct_num = (t.argmax(output, 1) == t.argmax(targets, 1)).numpy()
        correct_sum = correct_num.sum()
        data_num = data_num+len(correct_num)
        correct_all = correct_all+correct_sum
    mm2.train()
    print("-- Test Acc %.3f" % (correct_all/data_num))
    return correct_all/data_num

if __name__ == '__main__':
    train(True, 'model/mm_rnn_3.pth', 0.001, 10, 'model/mm_rnn', 100)
    # test(True, 200, 'model/mm_rnn_3.pth')
