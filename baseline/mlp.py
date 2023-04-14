from doctest import testfile
import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn.functional as F
import pandas as pd
import argparse
import logging
import numpy as np
import random
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=int, default=0.001)
parser.add_argument('--weight_decay', type=int, default=0.0001)
parser.add_argument('--nfeat', type=int, default=291)
parser.add_argument('--hidden1', type=int, default=16)
parser.add_argument('--hidden2', type=int, default=8)
parser.add_argument('--nclass', type=int, default=2)
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--dropout', type=int, default=0.2)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--nneib', type=int, default=20)
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
seeds = []
for i in range(5):
    seeds.append(random.randint(1, 100))

logging.basicConfig(level=logging.DEBUG,
                    filename='mlp.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def accuracy_head(out, y, pred, rate=1):
    y_0 = out[:, 0]
    _, idx = torch.sort(y_0, descending=True)
    n = int(out.shape[0] * rate)
    idx = idx[:n]
    head_ys, head_pred = y[idx], pred[idx]
    correct_head = head_pred.eq(head_ys).sum().item() / head_ys.shape[0]
    return correct_head



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

class MLP(nn.Module):
    def __init__(self, in_feature, hidden1, hidden2, out_feature, dropout):
        self.dropout = dropout
        super().__init__()
        self.linear1 = nn.Linear(in_feature, hidden1)
        kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden1, hidden2)
        kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(hidden2, out_feature)
        xavier_uniform_(self.linear3.weight)


    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear1(x)
        x = self.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return F.log_softmax(x, dim=1)



train_list_date = ['train']
setup_seed(42)
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

val_list_date = ['val']
val_nxs, val_x, val_y = get_data(args, val_list_date, balance=False)
val_x = val_x.unsqueeze(1)

val_nxs = normalize_test_data(val_nxs, dmin_nxs, dmax_nxs)
val_ti, val_x = val_x[:, :, :3], val_x[:, :, 3:]
val_x = normalize_test_data(val_x, dmin_txs, dmax_txs)

ex_uin = val_ti[:, :, 0].unsqueeze(2).repeat(1, 1, val_nxs.shape[2]).long()
bee_uin = val_ti[:, :, 1].unsqueeze(2).repeat(1, 1, val_nxs.shape[2]).long()
frc_uin = val_ti[:, :, 2].unsqueeze(2).repeat(1, 1, val_nxs.shape[2]).long()
ex_x = torch.gather(val_nxs, 1, ex_uin)
bee_x = torch.gather(val_nxs, 1, bee_uin)
frc_x = torch.gather(val_nxs, 1, frc_uin)
val_x = torch.cat((ex_x, bee_x, frc_x, val_x), dim=2)
val_x = val_x.squeeze()

arg_vals = [[val_x, val_y]]

test_list_dates = [['test1'], ["test2"], ["test3"]]
arg_tests = []
for test_list_date in test_list_dates:
    test_data = []
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
        test_data.append([test_x, test_y])
    arg_tests.append(test_data)


def train():
    model = MLP(args.nfeat, args.hidden1, args.hidden2, args.nclass, args.dropout)
    optimz = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    min_loss, max_auc = 1e10, 0
    patience = 0
    for epoch in range(args.epochs):

        optimz.zero_grad()
        out = model(train_x)
        loss = F.nll_loss(out, train_y)
        loss.backward()
        optimz.step()
        logging.info("epoch: {}, Train loss: {}".format(epoch, loss.item()))

        val = test(arg_vals, model)
        val_loss = val[-1]
        val_auc = val[-3]
        
    
        if epoch > 50 and val_auc > max_auc:
            max_auc = val_auc
            logging.info("Common distribution")
            result1 = test(arg_tests[0], model, p=True)
            logging.info("New distribution1")
            result2 = test(arg_tests[1], model, p=True)
            logging.info("New distribution2")
            result3 = test(arg_tests[2], model, p=True)
            total_result = [result1, result2, result3]
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break
        
    return total_result


        

def test(arg_test, model, p=False):
    model.eval()
    acc, acc2, acc4, acc6, acc8, aucs, f1s, tsts = [], [], [], [], [],[], [], []
    for arg in arg_test:
        test_x, test_y = arg

        with torch.no_grad():

            out = model(test_x)
            pred_auc = out[:, 1]
            pred = out.max(dim=1)[1]
            # print(sum(pred))
            auc = metrics.roc_auc_score(test_y, pred_auc)
            f1 = metrics.f1_score(test_y, pred, average='macro')
            acc_head_2 = accuracy_head(out, test_y, pred, 0.2)
            acc_head_4 = accuracy_head(out, test_y, pred, 0.4)
            acc_head_6 = accuracy_head(out, test_y, pred, 0.6)
            acc_head_8 = accuracy_head(out, test_y, pred, 0.8)

            loss = F.nll_loss(out, test_y)
            correct = pred.eq(test_y).sum().item()/test_x.shape[0]
            loss = loss.item()

            acc.append(correct), acc2.append(acc_head_2), acc4.append(acc_head_4), acc6.append(acc_head_6), acc8.append(acc_head_8)
            aucs.append(auc), f1s.append(f1), tsts.append(loss)
    if p:
        logging.info("Test acc:{} Test acc2:{} Test acc4:{} Test acc6:{} Test acc8:{}".format(np.mean(acc), np.mean(acc2), np.mean(acc4), np.mean(acc6), np.mean(acc8)))
        logging.info("Test AUC:{} Test F1:{} Test loss:{} ".format(np.mean(aucs), np.mean(f1s), np.mean(tsts)))
    best_result = [np.mean(acc), np.mean(acc2), np.mean(acc4), np.mean(acc6), np.mean(acc8), np.mean(aucs), np.mean(f1s), np.mean(tsts)]

    return best_result

acc, acc2, acc4, acc6, acc8, aucs, f1s, tsts = [], [], [], [], [],[], [], []
acc_dis, acc2_dis, acc4_dis, acc6_dis, acc8_dis, aucs_dis, f1s_dis, tsts_dis = [], [], [], [], [],[], [], []
acc_dis1, acc2_dis1, acc4_dis1, acc6_dis1, acc8_dis1, aucs_dis1, f1s_dis1, tsts_dis1 = [], [], [], [], [],[], [], []
for i in range(5):
    setup_seed(seeds[i])
    logging.info('Round:{}'.format(i))
    best_result = train()
    best_acc, best_acc2, best_acc4, best_acc6, best_acc8, best_auc, best_f1, min_tst_loss = best_result[0]
    acc.append(best_acc), acc2.append(best_acc2), acc4.append(best_acc4), acc6.append(best_acc6), acc8.append(best_acc8), aucs.append(best_auc), f1s.append(best_f1), tsts.append(min_tst_loss)

    best_acc, best_acc2, best_acc4, best_acc6, best_acc8, best_auc, best_f1, min_tst_loss = best_result[1]
    acc_dis.append(best_acc), acc2_dis.append(best_acc2), acc4_dis.append(best_acc4), acc6_dis.append(best_acc6), acc8_dis.append(best_acc8), aucs_dis.append(best_auc), f1s_dis.append(best_f1), tsts_dis.append(min_tst_loss)

    best_acc, best_acc2, best_acc4, best_acc6, best_acc8, best_auc, best_f1, min_tst_loss = best_result[2]
    acc_dis1.append(best_acc), acc2_dis1.append(best_acc2), acc4_dis1.append(best_acc4), acc6_dis1.append(best_acc6), acc8_dis1.append(best_acc8), aucs_dis1.append(best_auc), f1s_dis1.append(best_f1), tsts_dis1.append(min_tst_loss)

logging.info("Common distribution")
logging.info("Test acc:{}+-{} Test acc2:{}+-{} Test acc4:{}+-{} Test acc6:{}+-{} Test acc8:{}+-{}".format(np.mean(acc), np.std(acc), np.mean(acc2), np.std(acc2), np.mean(acc4), np.std(acc4), np.mean(acc6), np.std(acc6), np.mean(acc8), np.std(acc8)))
logging.info("Test AUC:{}+-{} Test F1:{}+-{} Test loss:{} ".format(np.mean(aucs), np.std(aucs), np.mean(f1s), np.std(f1s), np.mean(tsts)))

logging.info("New distribution1")
logging.info("Test acc:{}+-{} Test acc2:{}+-{} Test acc4:{}+-{} Test acc6:{}+-{} Test acc8:{}+-{}".format(np.mean(acc_dis), np.std(acc_dis), np.mean(acc2_dis), np.std(acc2_dis), np.mean(acc4_dis), np.std(acc4_dis), np.mean(acc6_dis), np.std(acc6_dis), np.mean(acc8_dis), np.std(acc8_dis)))
logging.info("Test AUC:{}+-{} Test F1:{}+-{} Test loss:{} ".format(np.mean(aucs_dis), np.std(aucs_dis), np.mean(f1s_dis), np.std(f1s_dis), np.mean(tsts_dis)))

logging.info("New distribution2")
logging.info("Test acc:{}+-{} Test acc2:{}+-{} Test acc4:{}+-{} Test acc6:{}+-{} Test acc8:{}+-{}".format(np.mean(acc_dis1), np.std(acc_dis1), np.mean(acc2_dis1), np.std(acc2_dis1), np.mean(acc4_dis1), np.std(acc4_dis1), np.mean(acc6_dis1), np.std(acc6_dis1), np.mean(acc8_dis1), np.std(acc8_dis1)))
logging.info("Test AUC:{}+-{} Test F1:{}+-{} Test loss:{} ".format(np.mean(aucs_dis1), np.std(aucs_dis1), np.mean(f1s_dis1), np.std(f1s_dis1), np.mean(tsts_dis1)))


