from MyFaceID_GAN.My_Face_ID_model import Face_ID_GAN_Model
import MyFaceID_GAN.My_loss as losses
import MyFaceID_GAN.config_train as config

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch import optim as opt
import torch
import numpy as np

import matplotlib.pyplot as plt

import time


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y_C = hist['C_losses']
    y_D = hist['D_losses']
    y_G = hist['G_losses']

    plt.plot(x, y_C, label='C_losses')
    plt.plot(x, y_D, label='D_losses')
    plt.plot(x, y_G, label='G_losses')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def train():

    conf = config.get_config()


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(conf.path_root_train, transform)
    print('dataset classes: %d', len(dataset.classes))
    assert len(dataset.classes) == 10575
    model = Face_ID_GAN_Model(len(dataset.classes))
    optim_c = opt.Adam(filter(lambda x:x.requires_grad is not False, model.c.parameters()), lr=conf.lr, weight_decay=0.0005) #权重衰减是否有必要
    optim_d = opt.Adam(filter(lambda x: x.requires_grad is not False, model.d.parameters()), lr=conf.lr,
                       weight_decay=0.0005)  # 权重衰减是否有必要
    optim_g = opt.Adam(filter(lambda x: x.requires_grad is not False, model.g.parameters()), lr=conf.lr,
                       weight_decay=0.0005)  # 权重衰减是否有必要

    model.cuda()

    #对应model.eval()
    model.train()
    k = torch.FloatTensor([0]).cuda()#D的损失函数系数

    train_hist = {}
    train_hist['C_losses'] = []
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []

    #总共的图片数是491542个
    loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, drop_last=True)

    steps = 0
    #开始训练
    for epoch in range(conf.epoch):
        print('%d epoch training...'%epoch)
        D_losses = []
        G_losses = []
        C_losses = []
        time_start_epoch = time.time()
        for _, data in enumerate(loader):

            inputs, labels= data
            inputs = inputs.cuda()
            labels = labels.cuda()
            #print(data, _)

            r_x_s, r_x_r, f_p_s, f_p_t, f_id_s, f_id_r, c_x_r, c_x_s = model(inputs)
            steps += 1
            lamda = losses.update_lamda(steps)
            k = losses.update_k(k, r_x_r, r_x_s)
            ld, lc, lg = losses.get_loss(r_x_s, r_x_r, f_p_s, f_p_t, f_id_s, f_id_r, c_x_r, c_x_s, labels, k, lamda)


            # 记录损失
            print('[%d epoch|%d iter]. train_loss c, d, g: %.4f, %.4f, %.4f'%(epoch+1, steps,
                                                                            lc.item(), ld.item(), lg.item()))
            D_losses.append(ld.item())
            G_losses.append(lg.item())
            C_losses.append(lc.item())


            #正向传播+反向传播
            optim_d.zero_grad()
            ld.backward(retain_graph=True)
            optim_d.step()

            optim_c.zero_grad()
            lc.backward(retain_graph=True)
            optim_c.step()

            optim_g.zero_grad()
            lg.backward()
            optim_g.step()




            if (steps + 1) % (50000*1.5) == 0:
                for param_group in optim_g.param_groups:
                    param_group['lr'] = param_group['lr'] - 0.0002
                for param_group in optim_d.param_groups:
                    param_group['lr'] = param_group['lr'] - 0.0002

            if optim_g.param_groups[0]['lr'] <= 0:
                print('lr of G <0, break the training')
                break


            if steps + 1 == (150000*1.5):
                for param_group in optim_c.param_groups:
                    param_group['lr'] = 0.0005
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['C_losses'].append(torch.mean(torch.FloatTensor(C_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

        time_end_epoch = time.time()
        per_epoch_time = time_end_epoch - time_start_epoch
        print('[%d epoch|%d total] - training-time: %.2f, loss_c:%.4f, loss_d:%.4f,'
              ' loss_g:%.4f'%((epoch+1), conf.epoch, per_epoch_time, torch.mean(torch.FloatTensor(C_losses)),
                              torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses)))
              )
    print('Training finsh!... save training results')
    #存储方式用pkl和pth都是一样的
    torch.save(model.c.state_dict(), 'save_models/c_param.pkl')
    torch.save(model.d.state_dict(), 'save_models/d_param.pkl')
    torch.save(model.g.state_dict(), 'save_models/g_param.pkl')

    #还可以把train_hist存起来
    show_train_hist(train_hist, save=True)

if __name__ == '__main__':
    train()