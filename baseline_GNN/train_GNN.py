import argparse
import gc
from pickletools import optimize
import sys
import numpy as np
import random
import argparse
import logging
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


from utils import get_data, normalize_x, normalize_test_data, normalize_adj, accuracy_head
from model import GCN, GAT
from dagnn import DaGnn

class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# test
def test(model, test_args, device, p=False):
    model.eval()
    acc, acc2, acc4, acc6, acc8, aucs, f1s, tsts = [], [], [], [], [],[], [], []
    for test_arg in test_args:
        test_nxs, test_tx, test_ti, test_nadjs, test_ys = test_arg
        test_data = TensorDataset(test_nxs, test_tx, test_ti, test_nadjs)
        test_data = DataLoader(test_data, batch_size=1024, shuffle=False)
        tmp = torch.Tensor([])
        for i, j in enumerate(test_data):
            test_nxs, test_tx, test_ti, test_nadjs = j
            test_nxs, test_tx, test_ti, test_nadjs = test_nxs.to(device), test_tx.to(device), test_ti.to(device), test_nadjs.to(device)
            out = model(test_nxs, test_nadjs, test_tx, test_ti)
            out = out.cpu()
            tmp = torch.cat((tmp, out), dim=0)
        out = tmp
        pred_auc = out[:, 1].detach().numpy()
        pred = out.max(dim=1)[1]

        loss = F.nll_loss(out, test_ys).item()
        auc = metrics.roc_auc_score(test_ys, pred_auc)
        f1 = metrics.f1_score(test_ys, pred, average='macro')

        correct = pred.eq(test_ys).sum().item() / test_ys.shape[0]
        acc_head_2 = accuracy_head(out, test_ys, pred, 0.2)
        acc_head_4 = accuracy_head(out, test_ys, pred, 0.4)
        acc_head_6 = accuracy_head(out, test_ys, pred, 0.6)
        acc_head_8 = accuracy_head(out, test_ys, pred, 0.8)
        acc.append(correct), acc2.append(acc_head_2), acc4.append(acc_head_4), acc6.append(acc_head_6), acc8.append(acc_head_8)
        aucs.append(auc), f1s.append(f1), tsts.append(loss)
    if p:
        logging.info("Test acc:{} Test acc2:{} Test acc4:{} Test acc6:{} Test acc8:{}".format(np.mean(acc), np.mean(acc2), np.mean(acc4), np.mean(acc6), np.mean(acc8)))
        logging.info("Test AUC:{} Test F1:{} Test loss:{} ".format(np.mean(aucs), np.mean(f1s), np.mean(tsts)))
    result = [np.mean(acc), np.mean(acc2), np.mean(acc4), np.mean(acc6), np.mean(acc8), np.mean(aucs), np.mean(f1s), np.mean(tsts)]
    return result


# train
def train_and_eval(args, device, seed):
    model = None
    if args.name == 'GCN':
        model = GCN(args.nfeat, args.nhid, args.nclass, args.nedge, args.dropout).to(device)
    elif args.name == 'GAT':
        model = GAT(args.nfeat, args.nhid, args.nclass, args.nedge, args.dropout).to(device)
    elif args.name == 'DaGNN':
        model = DaGnn(args.nfeat, args.nhid, args.nclass, args.nedge, args.dropout, 3).to(device)
    
    if args.name == 'DaGNN':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    setup_seed(42)

    # load train data
    train_list_date = ['train']
    train_nxs, train_tx, train_nadjs, train_ys = get_data(args, train_list_date, balance=True)
    train_ti, train_tx = train_tx[:, :3].unsqueeze(1), train_tx[:, 3:].unsqueeze(1)
    train_nxs, dmin_nxs, dmax_nxs = normalize_x(train_nxs)
    train_tx, dmin_txs, dmax_txs = normalize_x(train_tx)
    train_nadjs = normalize_adj(train_nadjs)

    # load val data
    val_list_date = ['val']
    arg_val = []
    for date in val_list_date:
        val_nxs, val_tx, val_nadjs, val_ys = get_data(args, [date], balance=False)
        val_ti, val_tx = val_tx[:, :3].unsqueeze(1), val_tx[:, 3:].unsqueeze(1)
        val_nxs = normalize_test_data(val_nxs, dmin_nxs, dmax_nxs)
        val_tx = normalize_test_data(val_tx, dmin_txs, dmax_txs)
        val_nadjs = normalize_adj(val_nadjs)
        arg_val.append([val_nxs, val_tx, val_ti, val_nadjs, val_ys])

    # load test data
    test_list_dates = [['test1'], ["test2"], ["test3"]]
    arg_tests = []
    for test_list_date in test_list_dates:
        arg_test = []
        for date in test_list_date:
            test_nxs, test_tx, test_nadjs, test_ys = get_data(args, [date], balance=False)
            test_ti, test_tx = test_tx[:, :3].unsqueeze(1), test_tx[:, 3:].unsqueeze(1)
            test_nxs = normalize_test_data(test_nxs, dmin_nxs, dmax_nxs)
            test_tx = normalize_test_data(test_tx, dmin_txs, dmax_txs)
            test_nadjs = normalize_adj(test_nadjs)
            arg_test.append([test_nxs, test_tx, test_ti, test_nadjs, test_ys])
        arg_tests.append(arg_test)
    
    setup_seed(seed)

    train_data = TensorDataset(train_nxs, train_nadjs, train_tx, train_ti, train_ys)
    train_data = DataLoader(train_data, batch_size=1024, shuffle=False)
    min_loss, max_auc, val_auc = 1e10, 0, 0
    for epoch in range(args.epochs):
        if args.output_to_log:
            logging.info('epoch{} begin'.format(epoch))

        model.train()
        for i, j in enumerate(train_data):
            train_nxs, train_nadjs, train_tx, train_ti, train_ys = j
            train_nxs, train_nadjs, train_tx, train_ti, train_ys = train_nxs.to(device), train_nadjs.to(device), train_tx.to(device), train_ti.to(device), train_ys.to(device)
            optimizer.zero_grad()
            out = model(train_nxs, train_nadjs, train_tx, train_ti)

            loss = F.nll_loss(out, train_ys)
            loss.backward()
            optimizer.step()

        if args.output_to_log:
            logging.info("Training MainModel loss:{}".format(loss.item()))

        result = test(model, arg_val, device)
        val_loss, val_auc = result[-1], result[-3]


        
        if epoch > 20 and val_auc > max_auc:
            max_auc = val_auc
            logging.info("Common distribution")
            result1 = test(model, arg_tests[0], device, p=True)
            logging.info("New distribution1")
            result2 = test(model, arg_tests[1], device, p=True)
            logging.info("New distribution2")
            result3 = test(model, arg_tests[2], device, p=True)
            total_result = [result1, result2, result3]
            
    return total_result


def main(args_str=None):
    seeds = []
    for i in range(5):
        seeds.append(random.randint(1,100))
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/')
    parser.add_argument('--output_to_log', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--nfeat', type=int, default=62)
    parser.add_argument('--nedge', type=int, default=6)
    parser.add_argument('--nhid', type=int, default=8)
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--name', type=str, default='GCN')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # log  
    name = args.name + "-" + str(args.lr)+"-"+str(args.nhid)+"-"+str(args.dropout)+".log"
    logging.basicConfig(level=logging.DEBUG,
                        filename=name,
                        filemode='a',
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )

    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())

    acc, acc2, acc4, acc6, acc8, aucs, f1s, tsts = [], [], [], [], [],[], [], []
    acc_dis, acc2_dis, acc4_dis, acc6_dis, acc8_dis, aucs_dis, f1s_dis, tsts_dis = [], [], [], [], [],[], [], []
    acc_dis1, acc2_dis1, acc4_dis1, acc6_dis1, acc8_dis1, aucs_dis1, f1s_dis1, tsts_dis1 = [], [], [], [], [],[], [], []

    for i in range(5):
        logging.info("Round {}".format(i))
        best_result = train_and_eval(args, device, seeds[i])
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
    logging.info("Test acc:{}+-{} Test acc2:{}+-{} Test acc4:{}+-{} Test acc6:{}+-{} Test acc8:{}+-{}".format(np.mean(acc_dis1), np.std(acc_dis1), np.mean(acc2_dis1), np.std(acc2_dis1), np.mean(acc4_dis1), np.std(acc4_dis1), np.mean(acc6_dis), np.std(acc6_dis), np.mean(acc8_dis), np.std(acc8_dis)))
    logging.info("Test AUC:{}+-{} Test F1:{}+-{} Test loss:{} ".format(np.mean(aucs_dis1), np.std(aucs_dis1), np.mean(f1s_dis1), np.std(f1s_dis1), np.mean(tsts_dis1)))
    
    
    return np.mean(acc), np.mean(aucs), np.mean(f1s), np.mean(tsts)


if __name__ == '__main__':
     print('(%.4f, %.4f, %.4f, %.4f)' % main())
     for _ in range(5):
          gc.collect()
          torch.cuda.empty_cache()