import torch as t
import torch.nn as nn
import visualize_model as mp
import numpy as np
import pickle
from tqdm import tqdm
import build_dataset_pytorch as build_dataset_pytorch
from torch.utils.data import DataLoader
import visdom



def dev(use_gpu, batch_size, model_path):
    dataset = build_dataset_pytorch.QuoraDataset_train('data/dev_data_pad.pkl', 'data/dev_data_char_pad.pkl',
                                                       'data/dev_data_flags_pad.pkl')
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    mm1 = mp.MatchModel(vocab_size=100000,tr_w_embed_dim=300,tr_w_emb_dim_flag=100,char_size=50,char_embed_dim=20,output_dim=100,
                        hidden_dim=200,use_gpu=use_gpu,yt_layer_num=3,mode='test')
    mm1.load_state_dict(t.load(model_path, map_location='cpu'))
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
        if i == 5:
            p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags = data
            print(t.nonzero(p_inputs).shape[0], t.nonzero(q_inputs).shape[0])
            if use_gpu:
                [p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec] = \
                    data_convert(p_inputs, q_inputs, targets, p_inputs_char, q_inputs_char, p_flags, q_flags, word_vec)
            p_input = [p_inputs, p_inputs_char, word_vec, p_flags]
            q_input = [q_inputs, q_inputs_char, q_flags]
            out_p, out_q, cos = mm1(100, p_input, q_input)
            vis = visdom.Visdom(env=u'test1')
            for x in range(len(out_p)):
                # print('\n%d:'%(x), cos[x].detach().numpy())
                # vis.image(out_p[x].detach().numpy()[:,:400], win='p_layer_%d_%d'%(x, 0), opts={'title':'p_layer%d'%(x)})
                # vis.image(out_q[x].detach().numpy()[:,:400], win='q_layer_%d_%d'%(x, 0), opts={'title':'q_layer%d'%(x)})
                vis.image(cos[x].detach().numpy(), win='q_layer_%d_%d' % (x, 0), opts={'title':'cos_layer%d'%(x)})
            break
                # for y in range(0,4):
                #     vis.image(out_p[x].detach().numpy()[:,400+y*100:500+y*100], win='p_layer_%d_%d' % (x, y+1))
                #     vis.image(out_q[x].detach().numpy()[:,400+y*100:500+y*100], win='q_layer_%d_%d' % (x, y+1))

            # if use_gpu:
            #     output = output.cpu()
            #     targets = targets.cpu()
            # print(output, targets)
    #         correct_num = (t.argmax(output, 1) == t.argmax(targets, 1)).numpy()
    #         correct_sum = correct_num.sum()
    #         data_num = data_num+len(correct_num)
    #         correct_all = correct_all+correct_sum
    # print("-- Dev Acc %.3f" % (correct_all/data_num))
    # return correct_all/data_num

if __name__ == '__main__':
    dev(False, 1, 'model/mm-1_15.pth')
    # vis = visdom.Visdom(env=u'test1')
    # x = t.arange(1,30,0.01)
    # y = t.sin(x)
    # vis.line(X=x, Y=y, win='sinx', opts={'title':'y=sin(x)'})
    # print(t.randn(3,64,64))
    # vis.image(t.randn(64,64).numpy())
    # vis.image(t.randn(3,64,64).numpy(), win='random2')
    # vis.images(t.randn(36,3,64,64).numpy(), nrow=6, win='random3', opts={'title':'random_imgs'})