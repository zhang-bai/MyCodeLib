import sys
import os
import copy
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from trainer import Trainer
from gnn import GNN
from ramps import *
from losses import *
import loader

def debugger(args):



    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    opt = vars(args)



    net_file = opt['dataset'] + '/net.txt'
    label_file = opt['dataset'] + '/label.txt'
    feature_file = opt['dataset'] + '/feature.txt'
    train_file = opt['dataset'] + '/train.txt'
    dev_file = opt['dataset'] + '/dev.txt'
    test_file = opt['dataset'] + '/test.txt'


    vocab_node = loader.Vocab(net_file, [0, 1])
    vocab_label = loader.Vocab(label_file, [1])
    vocab_feature = loader.Vocab(feature_file, [1])

    opt['num_node'] = len(vocab_node)
    opt['num_feature'] = len(vocab_feature)
    opt['num_class'] = len(vocab_label)


    graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
    label = loader.EntityLabel(file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1])
    feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])
    graph.to_symmetric(opt['self_link_weight'])
    feature.to_one_hot(binary=True)
    adj = graph.get_sparse_adjacency(opt['cuda'])

    with open(train_file, 'r') as fi:
        idx_train = [vocab_node.stoi[line.strip()] for line in fi]
    with open(dev_file, 'r') as fi:
        idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
    with open(test_file, 'r') as fi:
        idx_test = [vocab_node.stoi[line.strip()] for line in fi]
    idx_all = list(range(opt['num_node']))

    idx_unlabeled = list(set(idx_all)-set(idx_train))
    inputs = torch.Tensor(feature.one_hot)
    target = torch.LongTensor(label.itol)
    idx_train = torch.LongTensor(idx_train)
    idx_dev = torch.LongTensor(idx_dev)
    idx_test = torch.LongTensor(idx_test)
    idx_all = torch.LongTensor(idx_all)
    idx_unlabeled = torch.LongTensor(idx_unlabeled)
    inputs_q = torch.zeros(opt['num_node'], opt['num_feature'])
    target_q = torch.zeros(opt['num_node'], opt['num_class'])
    inputs_p = torch.zeros(opt['num_node'], opt['num_class'])
    target_p = torch.zeros(opt['num_node'], opt['num_class'])

    if opt['cuda']:
        inputs = inputs.cuda()
        target = target.cuda()
        idx_train = idx_train.cuda()
        idx_dev = idx_dev.cuda()
        idx_test = idx_test.cuda()
        idx_all = idx_all.cuda()
        idx_unlabeled = idx_unlabeled.cuda()
        inputs_q = inputs_q.cuda()
        target_q = target_q.cuda()
        inputs_p = inputs_p.cuda()
        target_p = target_p.cuda()

    gnn = GNN(opt)
    trainer = Trainer(opt, gnn)

    # import gaux.graph_emb as emb
    options = dict()
    options['lr'] = opt['lr']
    options['weight_decay'] = opt['decay']

    options['input_channels'] = 1
    options['emb1_channels'] = 2
    options['emb2_channels'] = 1
    options['Mix_nb'] = 2
    options['cuda'] = opt['cuda']
    # options['fea_in'] = (opt['num_feature'] + opt['num_class'] - 2*options['Mix_nb'] + 2)   # 1354*1437/2874
    options['fea_in'] = (opt['num_feature'] - 2 * options['Mix_nb'] + 2)  # 1354*1437/2874
    options['fea_out'] = opt['num_feature']
    options['lab_in'] = options['fea_in']
    options['lab_out'] = opt['num_class']
    options['adj_in'] = options['fea_in']
    options['adj_out'] = 1

    import gaux
    import itertools
    g_emb = gaux.graph_emb(options)
    g_fea = gaux.graph_fea(options)
    g_lab = gaux.graph_lab(options)
    g_adj = gaux.graph_adj(options)
    g_para = itertools.chain(g_emb.parameters(), g_fea.parameters(), g_lab.parameters(), g_adj.parameters())
    g_optim = torch.optim.Adam(g_para, lr=0.001, weight_decay=options['weight_decay'])
    g_bce_loss = nn.BCELoss().cuda()
    g_mse_loss = nn.MSELoss().cuda()
    g_softmax = nn.Softmax(dim=1).cuda()

    # total_para = itertools.chain(g_para, gnn.parameters())

    # Build the ema model
    gnn_ema = GNN(opt)

    for ema_param, param in zip(gnn_ema.parameters(), gnn.parameters()):
                ema_param.data= param.data

    for param in gnn_ema.parameters():
                param.detach_()
    trainer_ema = Trainer(opt, gnn_ema, ema = False)


    def init_data():
        inputs_q.copy_(inputs)
        temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
        target_q[idx_train] = temp

    def update_ema_variables(model, ema_model, alpha, epoch):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (epoch + 1), alpha)
        #print (alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def get_current_consistency_weight(final_consistency_weight, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        epoch = epoch - args.consistency_rampup_starts
        return final_consistency_weight *sigmoid_rampup(epoch, args.consistency_rampup_ends - args.consistency_rampup_starts )

    def sharpen(prob, temperature):
        temp_reciprocal = 1.0/ temperature
        prob = torch.pow(prob, temp_reciprocal)
        row_sum = prob.sum(dim=1).reshape(-1,1)
        out = prob/row_sum
        return out

    def train(epoches):
        best = 0.0
        init_data()
        results = []

        if args.consistency_type == 'mse':
            consistency_criterion = softmax_mse_loss
        elif args.consistency_type == 'kl':
            consistency_criterion = softmax_kl_loss

        # 筛选index
        adj_g = adj.to_dense()
        adj_g *= torch.ones_like(adj_g).cuda()-torch.eye(adj_g.shape[0]).cuda()
        f_index = torch.arange(0, inputs_q.shape[0], 2)
        f_index = f_index.type(torch.long)

        # 先只考虑有 label 的样本来选择 idx
        idx_b = idx_train.cpu()
        adj_bbb = adj_g[idx_b]
        idx_ccc = torch.gt(adj_bbb, 0)
        adj_onehop = adj_bbb.nonzero()
        nb_onehop = torch.sum(idx_ccc, dim=1)
        del adj_bbb, idx_ccc

        indices = dict()
        count_ = 0
        ct_ = 0
        while count_ < nb_onehop.shape[0]:
            tempee = []
            for _ in range(nb_onehop[count_].cpu().numpy()):
                if nb_onehop[count_] == 1:
                    ct_ += 1
                    continue
                else:
                    tempee.append(int(adj_onehop[ct_, 1].cpu().numpy()))
                    indices[int(idx_b[count_])] = tempee
                    ct_ += 1

            count_ += 1

        del count_,ct_
        nb_indices = len(indices)
        print('Nb of node b: ', nb_indices)
        idx_a, idx_c = torch.zeros((nb_indices), dtype=torch.int64), torch.zeros((nb_indices), dtype=torch.int64)
        idx_b = torch.zeros((nb_indices), dtype=torch.int64)
        for lll,i_b in enumerate(indices.keys()):
            i_a, i_c = torch.randperm(len(indices[i_b]))[:2]
            idx_a[lll]=indices[i_b][i_a]
            idx_c[lll]=indices[i_b][i_c]
            idx_b[lll]= i_b

        if torch.cuda.is_available():
            idx_a = idx_a.cuda()
            idx_b = idx_b.cuda()
            idx_c = idx_c.cuda()

        # 训练 adj 预测模型，输入一半相连，一半不相连节点下标
        idx_adj_b = idx_b
        idx_adj_c = torch.zeros_like((idx_adj_b), dtype=torch.int64)

        # adj_bulian = np.where((adj_g[idx_a[nb_indices//2:]].cpu()+torch.eye(adj_g.shape[0])).numpy() ==0)
        print('计算adj下标')
        for aaaaaaa,kkk in enumerate(idx_adj_b[nb_indices//2:]):
            for bbb in range(adj_g.shape[0]):
                if adj_g[kkk,bbb]==0 and kkk != bbb:
                    idx_adj_c[aaaaaaa+nb_indices//2]=torch.tensor(bbb)
                    if np.random.randint(50) == 1:
                        break
        print('adj下标计算完毕')
        idx_adj_c[:nb_indices//2]=idx_a[:nb_indices//2]
        ffff_index = [i for i in range(idx_adj_c.shape[0])]
        random.shuffle(ffff_index)
        idx_adj_c = idx_adj_c[ffff_index]
        idx_adj_b = idx_adj_b[ffff_index]

        total_loss = 0

        for epoch in range(epoches):
            # rand_index = random.randint(0,1)

            # rand_index = random.randint(1,2)
            rand_index = 2

            # if epoch > 1000:
            #     rand_index = 2
            # else:
            #     rand_index = 1

            if rand_index ==2:

                # print('22222222222222222222222222222222222222')
                trainer.model.train()

                # calculate pseudo label
                k = 10
                temp = torch.zeros([k, target_q.shape[0], target_q.shape[1]], dtype=target_q.dtype)
                temp = temp.cuda()
                for i in range(k):
                    temp[i, :, :] = trainer.predict_noisy(inputs_q, adj)
                target_predict = temp.mean(dim=0)
                target_predict = sharpen(target_predict, 0.1)
                target_q[idx_unlabeled] = target_predict[idx_unlabeled]
                del target_predict, temp

                # 拼接 features , labels
                # inputs_g = torch.cat((inputs_q, target_q), 1)
                inputs_g = inputs_q

                # 每50 epoch 更新一次 Mix 模型
                if epoch % 50 == 0:
                    g_optim.zero_grad()

                    for jjj in range(50):

                        emb = g_emb(inputs_g,aadj, idx_a, idx_c)

                        new_fea = g_fea(emb)
                        new_lab = g_lab(emb)
                        new_adj = g_adj(g_emb(inputs_g, idx_adj_b, idx_adj_c))

                        loss_f = g_mse_loss(new_fea, inputs_q[idx_b, :])
                        loss_l = g_mse_loss(new_lab, target_q[idx_b])
                        loss_a = g_mse_loss(new_adj, adj_g[idx_adj_b, idx_adj_c])

                        # total_loss = loss_f + loss_l + loss_a
                        total_loss = loss_a
                        total_loss.backward()
                        g_optim.step()

                        print(emb)
                        print("Mix | loss_f {:.5f} | loss_l {:.5f}  | loss_a {:.5f}".format(loss_f,loss_l,loss_a ))
                        print("Epoch | ",jjj)

                # 产生新的graph, idx_g_a 随机抽取原始节点生成新节点
                idx_g_a = idx_a
                idx_g_b = idx_b

                emb = g_emb(inputs_g, idx_g_a, idx_g_b)
                new_fea = g_fea(emb)
                new_lab = g_lab(emb)
                inputs_g_new = torch.cat((new_fea, new_lab), 1)

                idx_g_adj_a = torch.arange(0, inputs_g_new.shape[0], 1,dtype=torch.int64)
                new_adj=torch.zeros((inputs_g_new.shape[0], inputs_g_new.shape[0]),dtype=torch.float32)
                for id in range(inputs_g_new.shape[0]):
                    idx_g_adj_b = torch.cat((idx_g_adj_a[id:], idx_g_adj_a[:id]), 0)

                    new_adj[id]=g_adj(g_emb(inputs_g_new, idx_g_adj_a, idx_g_adj_b))
                new_adj = new_adj.cuda()

                # new_g_adj = fx(new_adj)

                # # 产生新的feature的adj, 同样只考虑一种情况
                # g_inputs = torch.cat((new_fea, new_lab), 1)
                # g_inputs = g_inputs.view(-1, 1, 2, g_inputs.shape[-1])
                #
                # index_i = np.linspace(0, new_fea.shape[0],new_fea.shape[0]//2+1)[:-1]
                # index_j = index_i + 1
                # # index_i, index_j = index_i.type(torch.long), index_j.type(torch.long)
                # f_index = torch.LongTensor([index_i, index_j])
                # f_index = f_index.cuda()
                #
                # new_emb = g_emb(g_inputs)
                # new_adj = g_adj(new_emb)
                # # new_gra = torch.zeros((new_fea.shape[0],new_fea.shape[0]))
                # # new_gra = new_gra.cuda()
                # # new_gra[index_i, index_j] = new_adj

                # new_gra = torch.sparse.FloatTensor(f_index, new_adj, (new_fea.shape[0],new_fea.shape[0]))

                # 计算loss
                trainer.optimizer.zero_grad()
                loss1 = trainer.update_soft(inputs_q, adj, target_q, idx_train)
                loss2 = trainer.update_soft(new_fea,new_adj, new_lab, idx_g_adj_a)

                total_loss = loss1 + 0.5*loss2
                # total_loss = loss2
                total_loss.backward()
                trainer.optimizer.step()




            if rand_index == 0:
                # print('0000000000000000000000000000000000000000000')
                trainer.model.train()
                trainer.optimizer.zero_grad()
                # predict k times to take the mean predict
                k = 10
                temp = torch.zeros([k, target_q.shape[0], target_q.shape[1]], dtype=target_q.dtype)
                temp = temp.cuda()
                for i in range(k):
                    temp[i,:,:] = trainer.predict_noisy(inputs_q)
                target_predict = temp.mean(dim = 0)

                target_predict = sharpen(target_predict,0.1)
                target_q[idx_unlabeled] = target_predict[idx_unlabeled]

                # rand mix
                temp = torch.randint(0, idx_unlabeled.shape[0], size=(idx_train.shape[0],))
                idx_unlabeled_subset = idx_unlabeled[temp]
                loss , loss_usup= trainer.update_soft_aux(inputs_q, target_q, target, idx_train, idx_unlabeled_subset, adj,  opt, mixup_layer =[1])
                mixup_consistency = get_current_consistency_weight(opt['mixup_consistency'], epoch)
                total_loss = loss + mixup_consistency*loss_usup
                total_loss.backward()
                trainer.optimizer.step()

            elif rand_index==1:
                # print('111111111111111111111111')
                trainer.model.train()
                trainer.optimizer.zero_grad()
                loss = trainer.update_soft(inputs_q,adj, target_q, idx_train)

                total_loss = loss
                total_loss.backward()
                trainer.optimizer.step()

            _, preds, accuracy_train = trainer.evaluate(inputs_q,adj, target, idx_train)
            _, preds, accuracy_dev = trainer.evaluate(inputs_q,adj, target, idx_dev)
            _, preds, accuracy_test = trainer.evaluate(inputs_q,adj, target, idx_test)
            _, preds, accuracy_test_ema = trainer_ema.evaluate(inputs_q,adj, target, idx_test)
            results += [(accuracy_dev, accuracy_test)]

            if epoch%50 == 0:
                if rand_index == 0:
                    print ('epoch :{:4d},loss:{:.10f}, train_acc:{:.3f}, dev_acc:{:.3f}, test_acc:{:.3f}'.format(epoch, total_loss.item(), accuracy_train, accuracy_dev, accuracy_test))
                else :
                     print ('epoch :{:4d},loss:{:.10f}, train_acc:{:.3f}, dev_acc:{:.3f}, test_acc:{:.3f}'.format(epoch, total_loss.item(), accuracy_train, accuracy_dev, accuracy_test))

            if accuracy_dev > best:
                best = accuracy_dev
                state = dict([('model', copy.deepcopy(trainer.model.state_dict())), ('optim', copy.deepcopy(trainer.optimizer.state_dict()))])

            update_ema_variables(gnn, gnn_ema, opt['ema_decay'], epoch)


        return results


    base_results = []
    base_results += train(opt['pre_epoch'])


    def get_accuracy(results):
        best_dev, acc_test = 0.0, 0.0
        for d, t in results:
            if d >= best_dev:
                best_dev, acc_test = d, t
        return acc_test

    acc_test = get_accuracy(base_results)

    print('Test acc{:.3f}'.format(acc_test * 100))

