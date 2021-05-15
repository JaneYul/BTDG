import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import pandas as pd 


class BTDG(torch.nn.Module):
    def __init__(self, data, e_d, r_d, coarse_grain, dropout):
        super(BTDG, self).__init__()
        self.entity_dim = e_d
        self.rel_dim = r_d
        
        self.fine_num = data.times_num           
        self.fine_times_id_dict = data.times_id_dict

        self.coarse = coarse_grain
        self.coarse_num = data.coarse_num_dict[self.coarse] 
        
        self.fine2coarse_list = data.times_coarse_list  

        self.S1 = torch.nn.Embedding(data.entity_num, e_d) 
        self.O1 = torch.nn.Embedding(data.entity_num, e_d)
        self.S2 = torch.nn.Embedding(data.entity_num, e_d)
        self.O2 = torch.nn.Embedding(data.entity_num, e_d)

        self.P1 = torch.nn.Embedding(data.rel_num * 2, r_d)
        self.P2 = torch.nn.Embedding(data.rel_num * 2, r_d)

        self.G1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (r_d, e_d, e_d)), dtype=torch.float, device='cuda', requires_grad=True))
        self.G2 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.coarse_num, r_d, e_d, e_d)), dtype=torch.float, device='cuda', requires_grad=True))

        self.T_S = torch.nn.Embedding(data.times_num, e_d)
        self.T_O = torch.nn.Embedding(data.times_num, e_d)

        self.bn11 = torch.nn.BatchNorm1d(e_d)
        self.bn12 = torch.nn.BatchNorm1d(e_d)
        self.bn21 = torch.nn.BatchNorm1d(e_d)
        self.bn22 = torch.nn.BatchNorm1d(e_d)

        self.dropout1 = torch.nn.Dropout(dropout[0])
        self.dropout2 = torch.nn.Dropout(dropout[1])

        self.loss = torch.nn.BCELoss()


    def get_coarse_num(self, coarse_num_dict, coarse):
        if coarse not in {'year', 'quarterly', 'month'}:
            coarse_int = int(coarse)
            return int(self.fine_num / coarse_int) + 1
        else:
            return coarse_num_dict[coarse]


    def initialize(self):
        torch.nn.init.xavier_normal_(self.S1.weight)
        torch.nn.init.xavier_normal_(self.S2.weight)
        torch.nn.init.xavier_normal_(self.P1.weight)
        torch.nn.init.xavier_normal_(self.P2.weight)
        torch.nn.init.xavier_normal_(self.O1.weight)
        torch.nn.init.xavier_normal_(self.O2.weight)
        torch.nn.init.xavier_normal_(self.T_S.weight)
        torch.nn.init.xavier_normal_(self.T_O.weight)
        self.train_fine2coarse_list = self.fine2coarse_list


    def batch_Tucker2d(self, m1, m2, t):
        tuple_num = m1.size(0)
        m1_size1 = m1.size(1)
        t_size0 = t.size(0)
        t_size1 = t.size(1)
        t_size2 = t.size(2)

        t_temp1 = t.reshape(t_size0, t_size1*t_size2)
        result_temp1 = torch.mm(m2, t_temp1)
        result_temp1 = result_temp1.view(tuple_num, t_size1, t_size2)
        result_temp2 = torch.bmm(m1.view(tuple_num, 1, m1_size1), result_temp1).view(tuple_num, t_size2)
        return result_temp2


    def time_smoothness_regularizer(self):
        loss = (torch.norm(self.T_S.weight[1:] - self.T_S.weight[:-1]) + torch.norm(self.T_O.weight[1:] - self.T_O.weight[:-1])) / self.fine_num
        return loss


    def forward(self, data_batch, vt_times_coarse_list=None):
        if data_batch.shape[1] == 4:
            return self.forward_point(data_batch)
        elif data_batch.shape[1] == 5:
            self.fine2coarse_list = self.train_fine2coarse_list
            if vt_times_coarse_list != None:
                self.fine2coarse_list = vt_times_coarse_list
            return self.forward_period(data_batch) 


    def forward_point(self, data_batch):   
        s = torch.Tensor(data_batch[:,0]).cuda()
        p = torch.Tensor(data_batch[:,1]).cuda()
        o = torch.Tensor(data_batch[:,2]).cuda()
        times = torch.Tensor(data_batch[:,3]).cuda()

        S1_embed = self.S1(s.long())
        P1_embed = self.P1(p.long())
        S1_embed = self.bn11(S1_embed)
        S1_embed = self.dropout1(S1_embed)
        pred1_embed = self.batch_Tucker2d(S1_embed, P1_embed, self.G1)
        pred1_embed = self.bn12(pred1_embed)
        pred1 = torch.mm(pred1_embed, self.O1.weight.transpose(1, 0))
        
        S2_embed = self.S2(s.long())
        P2_embed = self.P2(p.long())
        S2_embed = self.bn21(S2_embed)
        S2_embed = self.dropout2(S2_embed)
        
        TS_embed = self.T_S(times.long())
        TO_embed = self.T_O(times.long())

        times_list = times.detach().cpu().numpy().astype(int).tolist()
        count_time = Counter(times_list)

        cur_idx = 0                                        
        pred2_embed = torch.zeros(len(s), self.entity_dim).cuda()
        for (t, n) in sorted(count_time.items()):  
            S2_embed_temp = S2_embed[cur_idx:cur_idx+n]
            P2_embed_temp = P2_embed[cur_idx:cur_idx+n]
            TS_embed_temp = TS_embed[cur_idx:cur_idx+n]
            TO_embed_temp = TO_embed[cur_idx:cur_idx+n]
            
            time_coarse = self.fine2coarse_list[self.coarse][t]
            pred2_embed_temp = self.batch_Tucker2d(S2_embed_temp * TS_embed_temp, P2_embed_temp, self.G2[time_coarse]) * TO_embed_temp
            pred2_embed[cur_idx:cur_idx+n] = pred2_embed_temp
            cur_idx += n

        pred2_embed = self.bn22(pred2_embed)
        pred2 = torch.mm(pred2_embed, self.O2.weight.transpose(1, 0))

        pred = pred1 + pred2            
        pred = torch.sigmoid(pred)

        return pred


    def forward_period(self, data_batch):  

        s = torch.Tensor(data_batch[:,0]).cuda()
        p = torch.Tensor(data_batch[:,1]).cuda()
        o = torch.Tensor(data_batch[:,2]).cuda()
        start_times = data_batch[:,3].tolist()
        end_times = data_batch[:,4].tolist()

        S1_embed = self.S1(s.long())
        P1_embed = self.P1(p.long())
        S1_embed = self.bn11(S1_embed)
        S1_embed = self.dropout1(S1_embed)
        
        pred1_embed = self.batch_Tucker2d(S1_embed, P1_embed, self.G1)
        pred1_embed = self.bn12(pred1_embed)
        pred1 = torch.mm(pred1_embed, self.O1.weight.transpose(1, 0))
          
        S2_embed = self.S2(s.long())
        P2_embed = self.P2(p.long())
        S2_embed = self.bn21(S2_embed)
        S2_embed = self.dropout2(S2_embed)

        pred2_embed = torch.zeros(len(s), self.entity_dim).cuda()
        for i in range(len(s)): 
            S2_embed_temp = S2_embed[i].view(1, -1)
            P2_embed_temp = P2_embed[i].view(1, -1)

            start_times_id = self.fine_times_id_dict[start_times[i]][3] 
            end_times_id = self.fine_times_id_dict[end_times[i]][3]
            TS_embed_temp = torch.mean(self.T_S.weight[start_times_id:end_times_id+1], dim=0)   
            TO_embed_temp = torch.mean(self.T_O.weight[start_times_id:end_times_id+1], dim=0)
            
            times_coarse_list = self.fine2coarse_list[0][(start_times[i], end_times[i])][self.coarse]
            times_coarse_coeff = self.fine2coarse_list[1][(start_times[i], end_times[i])][self.coarse]

            G2_temp = self.G2[times_coarse_list]
            G2_temp = G2_temp.reshape(len(times_coarse_list), -1)
            G2_res_temp = G2_temp * torch.Tensor(times_coarse_coeff).cuda().reshape(-1, 1)
            G2_res_temp = G2_res_temp.reshape(len(times_coarse_list), self.rel_dim, self.entity_dim, self.entity_dim)
            G2_res = torch.sum(G2_res_temp, dim=0)

            pred2_embed_temp = self.batch_Tucker2d(S2_embed_temp * TS_embed_temp, P2_embed_temp, G2_res) * TO_embed_temp
            pred2_embed[i] = pred2_embed_temp

        pred2_embed = self.bn22(pred2_embed)
        pred2 = torch.mm(pred2_embed, self.O2.weight.transpose(1, 0)) 

        pred = pred1 + pred2  
        pred = torch.sigmoid(pred)

        return pred

