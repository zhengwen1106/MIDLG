import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
import joblib
import torch 
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/')
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=False)

args = parser.parse_args()

seeds = []
for i in range(5):
    seeds.append(random.randint(1, 100))


def load_data(args, date, balance=False, device='cpu'):
    nx = torch.load(args.path + 'nx_'+ date + '.pt')
    tx = torch.load(args.path + 'tx_'+ date + '.pt')[:, 0, :]
    y = torch.load(args.path + 'ys_'+ date + '.pt').long()[:, :2]
    y =  y.max(dim=1)[1]

    if balance:
        index1 = y == 1
        index0 = y == 0
        nx1, tx1, y1 = nx[index1].clone(), tx[index1].clone(), y[index1].clone()
        nx0, tx0, y0 = nx[index0].clone(), tx[index0].clone(), y[index0].clone()

        index0 = random.sample(range(len(y0)), len(y1))
        nx0, tx0, y0 = nx0[index0].clone(), tx0[index0].clone(), y0[index0].clone()

        nx, tx, y = torch.cat((nx0, nx1), dim=0), torch.cat((tx0, tx1), dim=0), torch.cat((y0, y1), dim=0)

    return  nx, tx, y


def get_data(args, list_date, balance=False):
    nxs, txs, ys= torch.Tensor([]).long(), torch.Tensor([]).long(), torch.Tensor([]).long()
    
    for date in list_date:
        nx, tx, y = load_data(args, date, balance)
        nxs, txs, ys = torch.cat((nxs, nx), 0), torch.cat((txs, tx), 0), torch.cat((ys, y))

    return nxs, txs, ys

def accuracy_head(out, y, pred, rate=1):
    y_0 = out[:, 0]
    _, idx = torch.sort(y_0, descending=True)
    n = int(out.shape[0] * rate)
    idx = idx[:n]
    pred = torch.Tensor(pred)
    y = torch.Tensor(y)
    head_ys, head_pred = y[idx], pred[idx]
    correct_head = head_pred.eq(head_ys).sum().item() / head_ys.shape[0]
    return correct_head

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

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

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


def train(i):
    # load train data
    train_list_date = ['train']
    train_nxs, train_x, train_y = get_data(args, train_list_date, balance=True)
    train_x = train_x.unsqueeze(1)
    train_nxs, dmin_nxs, dmax_nxs = normalize(train_nxs)
    train_ti, train_x = train_x[:, :, :3], train_x[:, :, 3:]
    train_x, dmin_txs, dmax_txs = normalize(train_x)
    ex_uin = train_ti[:, :, 0].unsqueeze(2).repeat(1, 1, train_nxs.shape[2]).long()
    bee_uin = train_ti[:, :, 1].unsqueeze(2).repeat(1, 1, train_nxs.shape[2]).long()
    frc_uin = train_ti[:, :, 2].unsqueeze(2).repeat(1, 1, train_nxs.shape[2]).long()
    ex_x = torch.gather(train_nxs, 1, ex_uin)
    bee_x = torch.gather(train_nxs, 1, bee_uin)
    frc_x = torch.gather(train_nxs, 1, frc_uin)
    train_x = torch.cat((ex_x, bee_x, frc_x, train_x), dim=2)
    train_x = train_x.squeeze()
    x_train, y_train = train_x.numpy(), train_y.numpy()

    # load test data
    test_data = []
    test_list_date = ['test1']
    for date in test_list_date:
        test_nxs, test_x, test_y = get_data(args, [date], balance=False)
        test_x = test_x.unsqueeze(1)
        test_nxs = normalize_test_data(test_nxs, dmin_nxs, dmax_nxs)
        test_ti, test_x = test_x[:, :, :3], test_x[:, :, 3:]
        test_x = normalize_test_data(test_x, dmin_txs, dmax_txs)
        ex_uin = test_ti[:, :, 0].unsqueeze(2).repeat(1, 1, test_nxs.shape[2]).long()
        bee_uin = test_ti[:, :, 1].unsqueeze(2).repeat(1, 1, test_nxs.shape[2]).long()
        frc_uin = test_ti[:, :, 2].unsqueeze(2).repeat(1, 1, test_nxs.shape[2]).long()
        ex_x, bee_x, frc_x = torch.gather(test_nxs, 1, ex_uin), torch.gather(test_nxs, 1, bee_uin), torch.gather(test_nxs, 1, frc_uin)
        test_x = torch.cat((ex_x, bee_x, frc_x, test_x), dim=2).squeeze()
        test_data.append([test_x.numpy(), test_y.numpy()])
    if args.save:
        np.save("svm_data.npy", test_data)
    setup_seed(seeds[i])
    model = svm.SVC(probability = True).fit(x_train, y_train)
    acc, acc2, acc4, acc6, auc, f1 = [], [], [], [], [], []
    for i, test_d in enumerate(test_data):
        x_test, y_test = test_d
        y_pred = model.predict(x_test)
        y_pred_auc = model.predict_proba(x_test)

        y_pred_auc_t = torch.Tensor(y_pred_auc)
        acc_head_2 = accuracy_head(y_pred_auc_t, y_test, y_pred, 0.2)
        acc_head_4 = accuracy_head(y_pred_auc_t, y_test, y_pred, 0.4)
        acc_head_6 = accuracy_head(y_pred_auc_t, y_test, y_pred, 0.6)

        acc.append(metrics.accuracy_score(y_test,y_pred))
        auc.append(metrics.roc_auc_score(y_test,y_pred_auc[:, 1]))
        f1.append(metrics.f1_score(y_test, y_pred, average='macro'))
        acc2.append(acc_head_2), acc4.append(acc_head_4), acc6.append(acc_head_6)
    if args.save_model:
        joblib.dump(model, "svm.m")
    return acc, acc2, acc4, acc6, auc, f1
    
accs, acc2s, acc4s, acc6s, aucs, f1s = [], [], [], [], [], []
for i in range(5):
    acc, acc2, acc4, acc6, auc, f1 = train(i)
    accs.append(np.mean(acc))
    aucs.append(np.mean(auc))
    f1s.append(np.mean(f1))
    acc2s.append(np.mean(acc2))
    acc4s.append(np.mean(acc4))
    acc6s.append(np.mean(acc6))

print('ACC:{}+-{} F1:{}+-{} AUC:{}+-{}'.format(np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s), np.mean(aucs), np.std(aucs) ))
print('ACC2:{}+-{} ACC4:{}+-{} ACC6:{}+-{}'.format(np.mean(acc2s), np.std(acc2s), np.mean(acc4s), np.std(acc4s), np.mean(acc6s), np.std(acc6s) ))

