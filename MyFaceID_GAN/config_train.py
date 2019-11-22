from easydict import  EasyDict

def get_config():
    conf = EasyDict()
    conf.path_root_train = '/home/zx/CASIA-algin-128'
    conf.batch_size = 36
    conf.lr = 0.0008
    conf.epoch = 20
    conf.path_p_net = '../train_p/saved_model/2_pnet.pth'

    return conf