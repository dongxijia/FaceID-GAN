import torch
from torch.nn import CrossEntropyLoss

CE = CrossEntropyLoss()
#这些lamda系数是怎么来的
lamda_1 = 100
lamda_2 = 1
lamda_3 = 1

def get_loss(r_x_s, r_x_r, f_p_s,f_p_t, f_id_s, f_id_r, c_x_r, c_x_s,label,k,lamda ):

    ld = r_x_r-k*r_x_s
    #ld = torch.add(r_x_r, torch.mul())
    #ld = r_x_r - 0 * r_x_s
    lc = CE(c_x_r,label) + lamda * CE(c_x_s,label)
    #lc = CE(c_x_r, label) + 0 * CE(c_x_s, label)
    #r_x_s是p1范数
    #lg = lamda_1 * r_x_s
    lg = lamda_1*r_x_s + lamda_2*cosine_distance(f_id_r,f_id_s) +lamda_3*l2_distance(f_p_s,f_p_t)
    #lg = lamda_1 * r_x_s + lamda_2 * torch.mean(torch.cosine_similarity(f_id_r, f_id_s)) + lamda_3 * l2_distance(f_p_s, f_p_t)

    return ld*100,lc,lg

def cosine_distance(f1,f2):
    #f1xf2为[16*16]
    f1_norm = torch.norm(f1,p=2,dim=1)
    #norm是系数相乘，而矩阵乘法应该用
    f2_norm = torch.norm(f2,p=2,dim=1)
    norm = f1_norm.mul(f2_norm)
    #print('norm shape ',norm.shape)
    #
    d = 1 - torch.mm(f1,f2.t())/norm.unsqueeze(1)
    d = torch.sum(d, dim=1)
    #mean_d = torch.mean(d)
    #return torch.sum(torch.diag(d,0))
    return torch.mean(d)

def l2_distance(f1,f2):
    #应该是整个batch的累积和
    #return torch.sum(torch.pow((f1-f2),2))
    l2_d = torch.norm((f1-f2), p=2, dim=1)
    #l2_d = torch.pow(torch.dist(f1, f2, p=2),2)
    l2_d = torch.pow(l2_d, 2)
    return torch.mean(l2_d)

def update_k(k,r_x_r,r_x_s):
    gamma = 1
    change = 0
    temp_k = k + 0.001 * (gamma * r_x_r - r_x_s)
    temp_k = temp_k.item()
    '''
    with torch.no_grad():
        a1 = r_x_r.cpu().detach().numpy()
        a2 = r_x_s.cpu().detach().numpy()
        change = 0.001*(gamma*a1-a2)
    '''
    print('k: %.6f -> k: %.6f'%(k, temp_k))
    #TODO k的分段 5000之后，变成原来的0.1
    #k = min(max(temp_k, 0), 1)
    return min(max(temp_k, 0), 1)
    #return k

def update_lamda(iters):
    if iters <= 30000:
        return 0.1
    elif iters <= 60000:
        return 0.2
    elif iters <= 90000:
        return 0.3
    elif iters <= 120000:
        return 0.4
    elif iters <= 150000:
        return 0.5
    else:
        return 1
    '''
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
    '''