import argparse
import gc
import sys
import os
import numpy as np
import random
import argparse
import logging
import torch
import torch.nn.functional as F
from sklearn import metrics
from model import MainModel, AdvModel
from utils import get_data, normalize, normalize_test_data, penalty, sample_domain, mean_nll, accuracy_head, normalize_env, normalize_a, loss_weight
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# test
def test_arg(model, arg_test, device, p=False):
    model.eval()
    acc, acc2, acc4, acc6, acc8, aucs, f1s, tsts = [], [], [], [], [],[], [], []
    for arg in arg_test:
        test_nxs, test_txs, test_nadjs, test_tadjs, test_ti, test_ys, test_rate, test_dy = arg
        test_data = TensorDataset(test_nxs, test_txs, test_ti, test_nadjs, test_tadjs)
        test_data = DataLoader(test_data, batch_size=1024, shuffle=False)
        tmp = torch.Tensor([])
        for i, j in enumerate(test_data):
            test_nxs, test_txs, test_ti, test_nadjs, test_tadjs = j
            test_nxs, test_txs, test_ti, test_nadjs, test_tadjs = test_nxs.to(device), test_txs.to(device), test_ti.to(device), test_nadjs.to(device), test_tadjs.to(device)
            out, _ = model(test_nxs, test_txs, test_ti, test_nadjs, test_tadjs)
            out = out.cpu()
            tmp = torch.cat((tmp, out), dim=0)
        out = tmp
        out_with_softmax = F.softmax(out, dim=1)
        pred_auc = out_with_softmax[:, 1].detach().numpy()
        pred = out_with_softmax.max(dim=1)[1]

        y = test_ys.max(dim=1)[1]

        loss = mean_nll(out, test_ys).item()
        auc = metrics.roc_auc_score(y, pred_auc)
        f1 = metrics.f1_score(y, pred, average='macro')

        correct = pred.eq(y).sum().item() / test_ys.shape[0]
        acc_head_2 = accuracy_head(out_with_softmax, y, pred, 0.2)
        acc_head_4 = accuracy_head(out_with_softmax, y, pred, 0.4)
        acc_head_6 = accuracy_head(out_with_softmax, y, pred, 0.6)
        acc_head_8 = accuracy_head(out_with_softmax, y, pred, 0.8)
        acc.append(correct), acc2.append(acc_head_2), acc4.append(acc_head_4), acc6.append(acc_head_6), acc8.append(acc_head_8), aucs.append(auc), f1s.append(f1), tsts.append(loss)
    if p:
        logging.info("Test Acc:{} Test Acc2:{} Test Acc4:{} Test Acc6:{} Test Acc8:{}".format(np.mean(acc), np.mean(acc2), np.mean(acc4), np.mean(acc6), np.mean(acc8)))
        logging.info("Test AUC:{} Test F1:{} Test loss:{} ".format(np.mean(aucs), np.mean(f1s), np.mean(tsts)))
    result  = [np.mean(acc), np.mean(acc2), np.mean(acc4), np.mean(acc6), np.mean(acc8), np.mean(aucs), np.mean(f1s), np.mean(tsts)]
    return result

def save_model(model, args, name):
    model.to("cpu")
    state = {"net": model.state_dict()}
    ckpt_path = os.path.join(args.save_dir, name)
    torch.save(state, ckpt_path)

# train
def train_and_eval(args, device, index, seed):
    # load train data
    train_list_date = ['train']
    setup_seed(42)
    envs, nlength, tlength = [], [], []
    for date in train_list_date:
        train_nxs, train_txs, train_nadjs, train_tadjs, train_ys, train_rate, train_sub_y = get_data(args, [date], balance=True)
        train_tadjs, train_txs = train_tadjs[:, :args.nneib, :args.nneib], train_txs[:, :args.nneib, :]
        train_ti, train_txs = train_txs[:, :, :3], train_txs[:, :, 3:]
        train_nadjs = normalize_a(train_nadjs)
        envs.append([train_nxs, train_txs, train_nadjs, train_tadjs, train_ti, train_ys, train_rate, train_sub_y])
        nlength.append(train_nxs.shape[0])
        tlength.append(train_txs.shape[0])
    envs, dmin_nxs, dmax_nxs = normalize_env(envs, nlength, 0)
    envs, dmin_txs, dmax_txs = normalize_env(envs, tlength, 1)

    # load val data
    val_list_date = ['val']
    arg_val = []
    for date in val_list_date:
        val_nxs, val_txs, val_nadjs, val_tadjs, val_ys, val_rate, val_dy = get_data(args, [date], balance=False)
        val_tadjs, val_txs = val_tadjs[:, :args.nneib, :args.nneib], val_txs[:, :args.nneib, :]
        val_ti, val_txs = val_txs[:, :, :3], val_txs[:, :, 3:]
        val_nxs = normalize_test_data(val_nxs, dmin_nxs, dmax_nxs)
        val_nadjs = normalize_a(val_nadjs)
        val_txs = normalize_test_data(val_txs, dmin_txs, dmax_txs)
        arg_val.append([val_nxs, val_txs, val_nadjs, val_tadjs, val_ti, val_ys, val_rate, val_dy])

    # load test data
    test_list_dates = [['test1'], ['test2'], ['test3']]
    arg_tests = []
    for test_list_date in test_list_dates:
        arg_test = []
        for date in test_list_date:
            test_nxs, test_txs, test_nadjs, test_tadjs, test_ys, test_rate, test_dy = get_data(args, [date], balance=False)
            test_tadjs, test_txs = test_tadjs[:, :args.nneib, :args.nneib], test_txs[:, :args.nneib, :]
            test_ti, test_txs = test_txs[:, :, :3], test_txs[:, :, 3:]
            test_nadjs = normalize_a(test_nadjs)
            test_nxs = normalize_test_data(test_nxs, dmin_nxs, dmax_nxs)
            test_txs = normalize_test_data(test_txs, dmin_txs, dmax_txs)
            arg_test.append([test_nxs, test_txs, test_nadjs, test_tadjs, test_ti, test_ys, test_rate, test_dy])
        arg_tests.append(arg_test)
    if args.save_data:
        np.save("MIDLG_data.npy", arg_tests)
        exit()
    setup_seed(seed)

    # 
    for env in envs:
        train_sub_y = env[-1]
        train_sub_y, dy = sample_domain(train_sub_y)
        env[-1] = train_sub_y
        env.append(dy)

    # init model
    mainNet = MainModel(args.nfeat, args.nhid, args.nclass, args.dropout, args.alpha, args.nheads, device).to(device)
    advNet = AdvModel(8 * args.nheads, 8, args.ndomain, args.dropout).to(device)
    optimizerM = torch.optim.Adam(mainNet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizerA = torch.optim.Adam(advNet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    min_loss, max_auc = 1e10, 0.0
    total_result = []
    for epoch in range(args.epochs):
        if args.output_to_log:
            logging.info('epoch{} begin'.format(epoch))
        for _ in range(args.Msteps):
            # train MainModel
            optimizerM.zero_grad()
            loss_nll = []
            loss_irm = []
            loss_adv = []
            # IRM: data in one day is a env
            for env in envs:
                env_nxs, env_txs, env_nadjs, env_tadjs, env_ti, env_ys, env_rate, env_sub_y, env_dy = env
                env_nxs, env_txs, env_nadjs, env_tadjs, env_ti, env_ys, env_rate, env_sub_y, env_dy = env_nxs.to(device), env_txs.to(device), env_nadjs.to(device), env_tadjs.to(device), env_ti.to(device), env_ys.to(device), env_rate.to(device), env_sub_y.to(device), env_dy.to(device)

                out, z = mainNet(env_nxs, env_txs, env_ti, env_nadjs, env_tadjs)
                domain_out = advNet(z, env_sub_y)
                lossM = mean_nll(out, env_ys)
                loss_nll.append(lossM)
                lossI = penalty(out, env_ys)
                loss_irm.append(lossI)
                lossA = mean_nll(domain_out, env_dy)
                loss_adv.append(lossA)
            # loss
            loss_nll = torch.stack(loss_nll).mean()
            loss_irm = torch.stack(loss_irm).mean()
            loss_adv = torch.stack(loss_adv).mean()
            
            loss1 = loss_nll + loss_irm * args.IRM_rate - loss_adv * args.ADV_rate
            loss1.backward()
            optimizerM.step()

            if args.output_to_log:
                logging.info("Training MainModel loss:{}".format(loss1.item()))

        # train AdvModel
        for s in range(args.Asteps):
            optimizerA.zero_grad()
            loss_adv = []
            for env in envs:
                env_nxs, env_txs, env_nadjs, env_tadjs, env_ti, env_ys, env_rate, env_sub_y, env_dy = env
                env_nxs, env_txs, env_nadjs, env_tadjs, env_ti, env_ys, env_rate, env_sub_y, env_dy = env_nxs.to(device), env_txs.to(device), env_nadjs.to(device), env_tadjs.to(device), env_ti.to(device), env_ys.to(device), env_rate.to(device), env_sub_y.to(device), env_dy.to(device)
                out, z = mainNet(env_nxs, env_txs, env_ti, env_nadjs, env_tadjs)
                domain_out = advNet(z, env_sub_y)
                lossA = mean_nll(domain_out, env_dy)
                loss_adv.append(lossA)
            loss2 = torch.stack(loss_adv).mean()
            if args.output_to_log:
                logging.info("Training AdvModel loss:{}".format(loss2.item()))
            loss2.backward()
            optimizerA.step()
            
        # test
        result = test_arg(mainNet, arg_val, device)
        val_loss = result[-1]
        val_auc = result[-3]
        if args.update == 'LOSS':
            flag = val_loss < min_loss
        elif args.update == 'AUC':
            flag = val_auc > max_auc
        if epoch > 10 and flag:
            if args.update == "LOSS":
                min_loss = val_loss
            else:
                max_auc = val_auc
            logging.info("Common distribution")
            result1 = test_arg(mainNet, arg_tests[0], device, p=True)
            logging.info("New distribution1")
            result2 = test_arg(mainNet, arg_tests[1], device, p=True)
            logging.info("New distribution2")
            result3 = test_arg(mainNet, arg_tests[2], device, p=True)
            total_result = [result1, result2, result3]
            if args.save:
                save_model(mainNet, args, "mainNet.pth")
                save_model(advNet, args, "advNet.pth")
                mainNet.to(device)
                advNet.to(device)

    return  total_result


def main(args_str=None):
    seeds = []
    for i in range(5):
        a = random.randint(1, 99)
        while a == 27:
            a = random.randint(1, 99)
        seeds.append(a)
    parser = argparse.ArgumentParser()
    
    # model train
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--Msteps', type=int, default=5)
    parser.add_argument('--Asteps', type=int, default=20)
    parser.add_argument('--nhid', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--IRM_rate', type=float, default=0.0)
    parser.add_argument('--ADV_rate', type=float, default=0.1)
    parser.add_argument('--update', type=str, default="AUC")

    # dataset 
    parser.add_argument('--nfeat', type=int, default=95)
    parser.add_argument('--nheads', type=int, default=2)
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--nneib', type=int, default=20)
    parser.add_argument('--ndomain', type=int, default=46)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--npeople', type=int, default=10)

    # parameters
    parser.add_argument('--path', type=str, default='data/')
    parser.add_argument('--output_to_log', type=bool, default=True)
    parser.add_argument('--name', type=str, default='MIDLG')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='trained_model')
    parser.add_argument('--save_data', type=bool, default=False)

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # log      
    name = args.name + "-" + str(args.IRM_rate)+"IRM-"+str(args.ADV_rate)+"ADV-"+str(args.nhid)+".log"
    logging.basicConfig(level=logging.DEBUG,
                        filename=name,
                        filemode='a',
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )
    logging.info(args)
    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())

    acc, acc2, acc4, acc6, acc8, aucs, f1s, tsts = [], [], [], [], [],[], [], []
    acc_dis, acc2_dis, acc4_dis, acc6_dis, acc8_dis, aucs_dis, f1s_dis, tsts_dis = [], [], [], [], [],[], [], []
    acc_dis1, acc2_dis1, acc4_dis1, acc6_dis1, acc8_dis1, aucs_dis1, f1s_dis1, tsts_dis1 = [], [], [], [], [],[], [], []

    # train 5 times
    for i in range(5):
        logging.info("Round {}".format(i))
        logging.info("Seed {}".format(seeds[i]))
        best_result = train_and_eval(args, device, i, seeds[i])
        # JD
        best_acc, best_acc2, best_acc4, best_acc6, best_acc8, best_auc, best_f1, min_tst_loss = best_result[0]
        acc.append(best_acc), acc2.append(best_acc2), acc4.append(best_acc4), acc6.append(best_acc6), acc8.append(best_acc8), aucs.append(best_auc), f1s.append(best_f1), tsts.append(min_tst_loss)
        # AD
        best_acc, best_acc2, best_acc4, best_acc6, best_acc8, best_auc, best_f1, min_tst_loss = best_result[1]
        acc_dis.append(best_acc), acc2_dis.append(best_acc2), acc4_dis.append(best_acc4), acc6_dis.append(best_acc6), acc8_dis.append(best_acc8), aucs_dis.append(best_auc), f1s_dis.append(best_f1), tsts_dis.append(min_tst_loss)
        # MD
        best_acc, best_acc2, best_acc4, best_acc6, best_acc8, best_auc, best_f1, min_tst_loss = best_result[2]
        acc_dis1.append(best_acc), acc2_dis1.append(best_acc2), acc4_dis1.append(best_acc4), acc6_dis1.append(best_acc6), acc8_dis1.append(best_acc8), aucs_dis1.append(best_auc), f1s_dis1.append(best_f1), tsts_dis1.append(min_tst_loss)

    logging.info("JD distribution")
    logging.info("Test acc:{}+-{} Test acc2:{}+-{} Test acc4:{}+-{} Test acc6:{}+-{} Test acc8:{}+-{}".format(np.mean(acc), np.std(acc), np.mean(acc2), np.std(acc2), np.mean(acc4), np.std(acc4), np.mean(acc6), np.std(acc6), np.mean(acc8), np.std(acc8)))
    logging.info("Test AUC:{}+-{} Test F1:{}+-{} Test loss:{} ".format(np.mean(aucs), np.std(aucs), np.mean(f1s), np.std(f1s), np.mean(tsts)))

    logging.info("AD distribution")
    logging.info("Test acc:{}+-{} Test acc2:{}+-{} Test acc4:{}+-{} Test acc6:{}+-{} Test acc8:{}+-{}".format(np.mean(acc_dis), np.std(acc_dis), np.mean(acc2_dis), np.std(acc2_dis), np.mean(acc4_dis), np.std(acc4_dis), np.mean(acc6_dis), np.std(acc6_dis), np.mean(acc8_dis), np.std(acc8_dis)))
    logging.info("Test AUC:{}+-{} Test F1:{}+-{} Test loss:{} ".format(np.mean(aucs_dis), np.std(aucs_dis), np.mean(f1s_dis), np.std(f1s_dis), np.mean(tsts_dis)))
    
    logging.info("MD distribution")
    logging.info("Test acc:{}+-{} Test acc2:{}+-{} Test acc4:{}+-{} Test acc6:{}+-{} Test acc8:{}+-{}".format(np.mean(acc_dis1), np.std(acc_dis1), np.mean(acc2_dis1), np.std(acc2_dis1), np.mean(acc4_dis1), np.std(acc4_dis1), np.mean(acc6_dis1), np.std(acc6_dis1), np.mean(acc8_dis1), np.std(acc8_dis1)))
    logging.info("Test AUC:{}+-{} Test F1:{}+-{} Test loss:{} ".format(np.mean(aucs_dis1), np.std(aucs_dis1), np.mean(f1s_dis1), np.std(f1s_dis1), np.mean(tsts_dis1)))
    
    return np.mean(acc), np.mean(aucs), np.mean(f1s), np.mean(tsts)


if __name__ == '__main__':
     print('(%.4f, %.4f, %.4f, %.4f)' % main())
     for _ in range(5):
          gc.collect()
          torch.cuda.empty_cache()