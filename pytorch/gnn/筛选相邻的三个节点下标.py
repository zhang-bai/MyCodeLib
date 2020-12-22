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