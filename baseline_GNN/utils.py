import numpy as np
import scipy.sparse as sp
import torch
import random


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 从文件中读取数据，balance参数含义表明获取的是均衡的数据还是非均衡的数据
def load_data(args, date, balance=False, device='cpu'):
    # 读取当前的数据
    nx = torch.load(args.path + 'nx_'+ date + '.pt').to(device)
    tx = torch.load(args.path + 'tx_'+ date + '.pt').to(device)[:, 0, :]
    nadj = torch.load(args.path + 'nadj_'+ date + '.pt').to(device)
    y = torch.load(args.path + 'ys_'+ date + '.pt').long()[:, :2].max(dim=1)[1]

    # 要求均衡数据的处理
    if balance:
        index1 = y == 1
        index0 = y == 0
        nx1, nadj1 = nx[index1].clone(), nadj[index1].clone() 
        tx1, y1 = tx[index1].clone(), y[index1].clone()
        nx0, nadj0 = nx[index0].clone(), nadj[index0].clone()
        tx0, y0 = tx[index0].clone(), y[index0].clone()
        index0 = random.sample(range(len(y0)), len(y1))
        nx0, nadj0 = nx0[index0].clone(), nadj0[index0].clone()
        tx0, y0 = tx0[index0].clone(), y0[index0].clone()
        nx, nadj = torch.cat((nx0, nx1), dim=0), torch.cat((nadj0, nadj1), dim=0)
        tx, y = torch.cat((tx0, tx1), dim=0), torch.cat((y0, y1), dim=0)
    return  nx, tx, nadj, y

# 根据名称加载数据
def get_data(args, list_date, balance=False):
    nxs, txs, nadjs, ys= torch.Tensor([]).long(), torch.Tensor([]).long(), torch.Tensor([]).long(), torch.Tensor([]).long()
    
    for date in list_date:
        # 获取当前训练所需要的数据，由于数据量过大，进行分文件存储
        nx, tx, nadj, y = load_data(args, date, balance)
        nxs, txs, nadjs, ys= torch.cat((nxs, nx), 0), torch.cat((txs, tx), 0), torch.cat((nadjs, nadj), 0), torch.cat((ys, y))
    nx1 = nxs[:, :, 3:15]
    # nx2 = nx[:, :, 21:26]
    nx2 = nxs[:, :, 27:30]
    nx3 = nxs[:, :, 34:49]
    nx4 = nxs[:, :, 53:73]
    nx5 = nxs[:, :, 83:]
    nxs = torch.cat([nx1, nx2, nx3, nx4, nx5], dim=-1)
    return nxs, txs, nadjs, ys

# 对数据进行归一化操作
def normalize_x(xs):
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

# 对test数据进行归一化
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

def normalize_adj(adj):
    # adj:[batch_size, 6, n, n]
    adj = torch.sum(adj, dim=1)
    ones = torch.ones_like(adj)
    zeros = torch.zeros_like(adj)
    adj = torch.where(adj>=1, ones, zeros)
    return normalize_a(adj)

def normalize_a(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, dim=2)
    rowsum = rowsum.unsqueeze(-1).repeat(1, 1, mx.shape[-1])
    mx = torch.div(mx, rowsum)
    mx = torch.where(torch.isnan(mx), torch.full_like(mx, 0), mx)
    return mx

def accuracy_head(out, y, pred, rate=1):
    # 指标计算
    y_0 = out[:, 0]
    _, idx = torch.sort(y_0, descending=True)
    n = int(out.shape[0] * rate)
    idx = idx[:n]
    head_ys, head_pred = y[idx], pred[idx]
    correct_head = head_pred.eq(head_ys).sum().item() / head_ys.shape[0]
    return correct_head


