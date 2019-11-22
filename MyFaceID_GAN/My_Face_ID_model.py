#Boundary Equilibrium GANs 生成高清图像用BeGAN最简单合适
from BeGAN.began import G,D
from Classifier.resnet import resnet50 as C
from train_p.resnet import resnet18 as P
#可以看到基础的G、D、C、P都是使用的已有的网络结构

import torch
from torch import nn
#import torchvision.models.resnet
import numpy as np

def initial_model(class_num):
    #TODO C、P、G、D
    classifer = C(pretrained=False,**{'size_feature':256, 'num_class':class_num})
    p = P(pretrained=False,**{'size_feature':235})
    p_state_dict = torch.load('../train_p/saved_model/2_pnet.pth')['state_dict']
    p.load_state_dict(p_state_dict)
    #h和inputs的维度是一样的
    #如果输入是112*112.则G的输入是603
    #如果输入是128*128,则G的输入是619
    g = G(h=619,n=64,output_dim=(3,128,128))
    d = D(h=619,n=64,input_dim=(3,128,128)) #d的输出也是（3, 128, 128吗）,为什么d不用2分类器
    total_num_params = 0
    for m in [classifer,p,g,d]:
        for pa in m.parameters():
            total_num_params += pa.numel()
    print('number of all models\' parameters are %d'%(total_num_params))
    return classifer,p,g,d


def transform_func(v, smile, silent):
    #从一个均匀的随机采样中初始化weight
    weight = np.random.uniform(0, 1)
    #随机选择一个角度和表情
    yaw_angle = torch.Tensor(v.shape[0], 1).uniform_(-0.3, 0.3)
    #lerp是线性插值 (smile+silent)*weight
    new_exp_v = torch.lerp(smile, silent, weight)
    v[:, -29:] = new_exp_v
    v[:, 1] = yaw_angle[:, 0]
    # v = t.cat((yaw_angle,v[:,-228:]))
    return v

class Face_ID_GAN_Model(nn.Module):
    def __init__(self, people_num=10575):
        #people_num代表训练集里有多少类人
        #初始化父类nn.Module
        super(Face_ID_GAN_Model, self).__init__()
        #初始化每一个组件, C、P、G、D
        #p直接用的预训练模型
        self.c, self.p, self.g, self.d = initial_model(people_num)

        #TODO get pretrained model

        #self.p.load_state_dict(torch.load('../train_p/saved_model/2_pnet.pth'))


        #p不需要进行参数更新
        for i in self.p.parameters():
            i.requires_grad = False

        #temp是用来干嘛的,temp是矩阵对应位系数
        self.smile_vector = torch.from_numpy(np.loadtxt('../train_p/1.txt', dtype=np.float32))[-29:]
        self.silent_vector = torch.from_numpy(np.loadtxt('../train_p/0.txt', dtype=np.float32))[-29:]
        #矩阵系数有235个，对应特征向量也是235
        temp_vector = [0] + [1] + [0] * 5 + [1] * 228
        self.temp_vector = torch.from_numpy(np.array(temp_vector, dtype=np.float32))

    def forward(self, x):
        #batch, class, height, weight, x==xr
        b, c, h, w = x.shape

        #c_x_r是分类信息，f_id_r是人脸特征，分类器C动作
        c_x_r, f_id_r = self.c(x)

        #f_p_r是p网络输出的pose特征,需要乘对应系数，在某些特征上有注意力
        f_p_r = self.p(x)
        f_p_r = f_p_r.mul(self.temp_vector.cuda())

        #噪声z
        z = torch.Tensor(b, 128).uniform_(-1, 1).cuda()

        #将f_id_r+z+f_p_r 放入G
        #首先f_p_r转为固定的角度和表情的pose特征，结果是229维
        f_p_t = transform_func(f_p_r, self.smile_vector.cuda(), self.silent_vector.cuda())

        #拼接, 生成器g动作
        g_inputs = torch.cat((f_id_r, f_p_t, z), dim=1)
        xs = self.g(g_inputs)
        #注意生成的xs的规格一定是2的幂，因此应该将数据集输入置为128
        assert xs.shape[-1] == 128

        #a = self.d(xs)
        #判别器d动作
        r_x_s = torch.sqrt(torch.dist(self.d(xs), xs, p=1))
        #a = self.d(x)
        r_x_r = torch.sqrt(torch.dist(self.d(x), x, p=1))

        #姿势估计p动作
        f_p_s = self.p(xs)
        f_p_s = f_p_s.mul(self.temp_vector.cuda())

        #分类器C动作
        c_x_s, f_id_s = self.c(xs)


        #注意r是真实图像，s是G生成的图像

        return r_x_s, r_x_r, f_p_s, f_p_t, f_id_s, f_id_r, c_x_r, c_x_s