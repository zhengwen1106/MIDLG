import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import random


# load data from files
def load_data(args, date, balance=False, device='cpu'):
    nx = torch.load(args.path + 'nx_'+ date + '.pt').to(device)
    tx = torch.load(args.path + 'tx_'+ date + '.pt').to(device)
    nadj = torch.load(args.path + 'nadj_'+ date + '.pt').to(device)
    tadj = torch.load(args.path + 'tadj_'+ date + '.pt').to(device)
    y = torch.load(args.path + 'ys_'+ date + '.pt').long()

    # balance positive and negative samples
    if balance:
        tmp = y[:, 1]
        index1 = tmp == 1
        index0 = tmp == 0
        nx1, nadj1 = nx[index1].clone(), nadj[index1].clone() 
        tx1, tadj1, y1 = tx[index1].clone(), tadj[index1].clone(), y[index1].clone()
        nx0, nadj0 = nx[index0].clone(), nadj[index0].clone()
        tx0, tadj0, y0 = tx[index0].clone(), tadj[index0].clone(), y[index0].clone()
        index0 = random.sample(range(len(y0)), len(y1))
        nx0, nadj0 = nx0[index0].clone(), nadj0[index0].clone()
        tx0, tadj0, y0 = tx0[index0].clone(), tadj0[index0].clone(), y0[index0].clone()
        nx, nadj = torch.cat((nx0, nx1), dim=0), torch.cat((nadj0, nadj1), dim=0)
        tadj, tx, y = torch.cat((tadj0, tadj1), dim=0), torch.cat((tx0, tx1), dim=0), torch.cat((y0, y1), dim=0)

    return  nx, tx, nadj, tadj, y



# load data from files and split them by day
def get_data(args, list_date, balance=False):
    nxs, txs, nadjs, tadjs, ys= torch.Tensor([]).long(), torch.Tensor([]).long(), torch.Tensor([]).long(), torch.Tensor([]).long(), torch.Tensor([]).long()
    rate, dy = torch.Tensor([]).long(), torch.Tensor([]).long()
    
    for date in list_date:
        nx, tx, nadj, tadj, y = load_data(args, date, balance)
        nxs, txs, nadjs, tadjs = torch.cat((nxs, nx), 0), torch.cat((txs, tx), 0), torch.cat((nadjs, nadj), 0), torch.cat((tadjs, tadj), 0)
        y, r, d =  y[:, :2], y[:, 2]/(y[:, 2] + y[:, 3]), y[:, 4:]
        ys, rate, dy = torch.cat((ys, y)), torch.cat((rate, r)), torch.cat((dy, d), 0)

    return nxs, txs, nadjs, tadjs, ys, rate, dy


# train normalization
def normalize(xs):
    dn, d = xs.shape[-2], xs.shape[-1]
    xs = xs.reshape((-1, d))

    dmin = torch.min(xs, dim=0).values.unsqueeze(0)
    dmax = torch.max(xs, dim=0).values.unsqueeze(0)
    d_min = dmin.repeat((xs.shape[0], 1))
    d_max = dmax.repeat((xs.shape[0], 1))
    xs = (xs - d_min )/(d_max-d_min)
    xs = torch.where(torch.isnan(xs), torch.full_like(xs, 0), xs)

    xs = xs.reshape((-1, dn, d))
    return xs, dmin, dmax
    

# test normalization
def normalize_test_data(xs, dmin, dmax):

    dn, d = xs.shape[-2], xs.shape[-1]
    xs = xs.reshape((-1, d))

    d_min = dmin.repeat((xs.shape[0], 1))
    d_max = dmax.repeat((xs.shape[0], 1))
    
    xs = (xs - d_min)/(d_max-d_min)
    xs = torch.where(torch.isnan(xs), torch.full_like(xs, 0), xs)
    xs = torch.where(torch.isinf(xs), torch.full_like(xs, 0), xs)
    
    xs = xs.reshape((-1, dn, d))
    return xs

# normalize a feature
def normalize_env(envs, length, i):
    xs = []
    for env in envs:
        xs.append(env[i])
    xs = torch.cat(xs, dim=0)
    neighbor, n = xs.shape[-2], xs.shape[-1]
    xs = xs.reshape((-1, n))

    d_min = torch.min(xs, dim=0).values[-1]
    d_max = torch.max(xs, dim=0).values[-1]

    dmin = torch.min(xs, dim=0).values.unsqueeze(0)
    dmax = torch.max(xs, dim=0).values.unsqueeze(0)
    d_min = dmin.repeat((xs.shape[0], 1))
    d_max = dmax.repeat((xs.shape[0], 1))
    xs = (xs - d_min )/(d_max-d_min)
    xs = torch.where(torch.isnan(xs), torch.full_like(xs, 0), xs)

    xs = xs.reshape((-1, neighbor, n))
    xs = torch.split(xs, length, dim=0)
    for j in range(len(envs)):
        envs[j][i] = xs[j]
        
    return envs, dmin, dmax

# adv: concat adversarial labels
def sample_domain(dy):
    N = range(2)
    ntype = dy.shape[-1]-1
    changeY = np.random.choice(N, size=dy.shape[0])
    for i in range(len(changeY)):
        if changeY[i]:
            t = dy[i, :].argmax()
            index = random.randint(0, ntype)
            while index == t:
                index = random.randint(0, ntype)
            dy[i, t] = 0
            dy[i, index] = 1
    y = torch.Tensor([1-changeY, changeY]).T
    return dy, y

# cross-entropy Loss
def mean_nll(logits, y, reduction = 'mean'):
    return nn.functional.cross_entropy(logits, y.float(), reduction = reduction)

# IRM_Loss
def penalty(logits, y):
    scale = torch.tensor(1.).requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

# TopK@ACC
def accuracy_head(out, y, pred, rate=1):
    y_0 = out[:, 0]
    _, idx = torch.sort(y_0, descending=True)
    n = int(out.shape[0] * rate)
    idx = idx[:n]
    head_ys, head_pred = y[idx], pred[idx]
    correct_head = head_pred.eq(head_ys).sum().item() / head_ys.shape[0]
    return correct_head

def normalize_a(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, dim=3)
    rowsum = rowsum.unsqueeze(-1).repeat(1, 1, 1, mx.shape[-1])
    mx = torch.div(mx, rowsum)
    mx = torch.where(torch.isnan(mx), torch.full_like(mx, 0), mx)
    return mx

def loss_weight(loss):
    loss = 4 * torch.pow(loss, 2) + 4 * loss + 1
    return loss
