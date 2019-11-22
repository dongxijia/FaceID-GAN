import torch
from torch.nn import CrossEntropyLoss

CE = CrossEntropyLoss()
#这些lamda系数是怎么来的
lamda_1 = 0.000001
lamda_2 = 1
lamda_3 = 1

def get_loss(r_x_s, r_x_r, f_p_s,f_p_t, f_id_s, f_id_r, c_x_r, c_x_s,label,k,lamda ):

    ld = r_x_r-k*r_x_s
    lc = CE(c_x_r,label) + lamda * CE(c_x_s,label)
    lg = lamda_1*r_x_s + lamda_2*cosine_distance(f_id_r,f_id_s) +lamda_3*l2_distance(f_p_s,f_p_t)
    return ld,lc,lg

def cosine_distance(f1,f2):
    #f1xf2为[16*16]
    f1_norm = torch.norm(f1,p=2,dim=1)
    #norm是系数相乘，而矩阵乘法应该用
    f2_norm = torch.norm(f2,p=2,dim=1)
    norm = f1_norm.mul(f2_norm)
    #print('norm shape ',norm.shape)
    d = 1 - torch.mm(f1,f2.t())/norm.unsqueeze(1)
    return torch.sum(torch.diag(d,0))

def l2_distance(f1,f2):
    return torch.sum(torch.pow((f1-f2),2))

def update_k(k,r_x_r,r_x_s):
    with torch.no_grad():
        a1 = r_x_r.cpu().detach().numpy()
        a2 = r_x_s.cpu().detach().numpy()
        change = 0.00001*(a1-a2)
    print('k: %.4f -> k: %.4f'%(k, k+change))
    #TODO k的分段 5000之后，变成原来的0.1
    return k + change#0.000
    #return k

def update_lamda(iters):
    if iters <= 30000:
        return 0.9
    elif iters <= 60000:
        return 0.7
    elif iters <= 90000:
        return 0.5
    elif iters <= 120000:
        return 0.3
    elif iters <= 150000:
        return 0.15
    else:
        return 0.05