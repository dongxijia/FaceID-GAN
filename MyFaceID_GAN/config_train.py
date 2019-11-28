from easydict import  EasyDict

def get_config():
    conf = EasyDict()
    conf.path_root_train = '/home/zx/CASIA-algin-500-128'
    conf.batch_size = 32
    conf.lr = 0.0008
    conf.epoch = 20
    conf.path_p_net = '../train_p/saved_model/2_pnet.pth'

    #model type
    conf.c_type = 'ResNet18'
    conf.d_type = 'BeGAN'
    conf.g_type = 'BeGAN'
    assert conf.c_type in ['ResNet50', 'ResNet18']
    assert conf.d_type in ['BeGAN']
    assert conf.g_type in ['BeGAN']

    conf.show_x_s = 1
    conf.show_d_x = 1

    #val
    conf.path_lfw = '/home/zx/face_verification/LFW/lfw_align_112'
    conf.path_pair_lfw = '/home/zx/face_verification/LFW/pairs.txt'

    return conf