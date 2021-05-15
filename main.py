from load_data import *
from model import *
import torch
import numpy as np
import argparse
import time
import collections
import os
import sys
from tqdm import tqdm

torch.set_num_threads(2)

class Trainer:
    def __init__(self, data, num_epochs, batch_size, lr, dr, entity_dim, rel_dim, coarse_grain, alpha, label_smoothing, dropout):
        self.data = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dr = dr
        self.entity_dim = entity_dim
        self.rel_dim = rel_dim
        self.coarse_grain = coarse_grain

        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        
        self.cuda = torch.cuda.is_available()


    def get_targets(self, data):
        targets_vocab = collections.defaultdict(list)
        if len(data[0]) == 4:
            for quad in data:
                targets_vocab[(quad[0], quad[1], quad[3])].append(quad[2])
        elif len(data[0]) == 5:
            for quad in data:
                targets_vocab[(quad[0], quad[1], quad[3], quad[4])].append(quad[2])
        return targets_vocab


    def get_batch(self, data, batch_size, idx, targets_vocab_o): 
        if idx + batch_size < len(data):
            data_batch = data[idx:idx+batch_size]
        else:
            data_batch = data[idx:]
        data_batch = np.array(data_batch)
        
        index = np.argsort(data_batch, axis=0)[:,3]
        data_batch = data_batch[index,:]
        
        targets_o = np.zeros((len(data_batch), self.data.entity_num))
        if len(data[0]) == 4:
            for i, d in enumerate(data_batch):
                d = d.tolist()
                targets_o[i, targets_vocab_o[(d[0], d[1], d[3])]] = 1.0  
        elif len(data[0]) == 5:
            for i, d in enumerate(data_batch):
                d = d.tolist()
                targets_o[i, targets_vocab_o[(d[0], d[1], d[3], d[4])]] = 1.0
        if self.cuda:
            targets_o = torch.Tensor(targets_o).cuda()

        return data_batch, targets_o


    def evaluate(self, model, data, all_tuples_targets_vocab_o, vt_times_coarse_list): 
        start_time = time.time()

        hits_o = [[] for i in range(10)]
        ranks_o = []                               
        for idx in range(0, len(data), self.batch_size):
            data_batch, _ = self.get_batch(data, self.batch_size, idx, all_tuples_targets_vocab_o)
            
            predictions_o = model.forward(data_batch, vt_times_coarse_list) 
            o_idx = torch.Tensor(data_batch[:,2]) 
                                         
            for j in range(len(data_batch)):  
                corr_idx_o = int(o_idx[j].item())
                filt = all_tuples_targets_vocab_o[(data_batch[j][0], data_batch[j][1], data_batch[j][3])] if len(data[0]) == 4 \
                        else all_tuples_targets_vocab_o[(data_batch[j][0], data_batch[j][1], data_batch[j][3], data_batch[j][4])]
                target_value = predictions_o[j, corr_idx_o].item()
                predictions_o[j, filt] = 0.0
                predictions_o[j, corr_idx_o] = target_value

                sort_predictions_o, sort_idx_o = torch.sort(predictions_o[j], descending=True)
                sort_idx_o = sort_idx_o.cpu().numpy()

                rank_o = np.where(sort_idx_o == corr_idx_o)[0][0]   
                
                for k in range(10):
                    if rank_o <= k:
                        hits_o[k].append(1)
                    else:
                        hits_o[k].append(0)
                ranks_o.append(rank_o+1) 

        print('Time:', time.time() - start_time)
        print('Hits@1: ', np.mean(hits_o[0]))  
        print('Hits@3: ', np.mean(hits_o[2]))                  
        print('Hits@10: ', np.mean(hits_o[9]))
        print('Mean rank: ', np.mean(ranks_o))  
        print('Mean reciprocal rank: ', np.mean(1.0/np.array(ranks_o)))                


    def train(self):
        train_tuples_targets_vocab_o = self.get_targets(self.data.train_data)  

        model = BTDG(self.data, self.entity_dim, self.rel_dim, self.coarse_grain, self.dropout)
        if self.cuda:
            model.cuda()
        model.initialize()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        if self.dr:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.dr)

        print('---Start training BTDG---')
        for epoch in range(1, self.num_epochs+1):
            model.train()
            losses = []
            np.random.shuffle(self.data.train_data) 
            for idx in range(0, len(self.data.train_data), self.batch_size):
                data_batch, targets_sorted_batch_o = self.get_batch(self.data.train_data, self.batch_size, idx, train_tuples_targets_vocab_o) 
                optimizer.zero_grad()
                if self.label_smoothing:
                    targets_sorted_batch_o = ((1.0 - self.label_smoothing) * targets_sorted_batch_o) + (1.0 / targets_sorted_batch_o.size(1))
                predictions_o = model.forward(data_batch)
                loss = model.loss(predictions_o, targets_sorted_batch_o) + self.alpha * model.time_smoothness_regularizer()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if self.dr:
                scheduler.step()

            print('***epoch:%s***' % str(epoch))
            print('train loss:', np.mean(losses))

            # valid
            valid_tuples_targets_vocab_o = self.get_targets(self.data.train_data + self.data.valid_data)
            all_tuples_targets_vocab_o = self.get_targets(self.data.train_data + self.data.valid_data + self.data.test_data)
            model.eval()   
            with torch.no_grad():
                print('Validation:')
                self.evaluate(model, self.data.valid_data, valid_tuples_targets_vocab_o, self.data.valid_times_coarse_list)
                print('Test:')
                self.evaluate(model, self.data.test_data, all_tuples_targets_vocab_o, self.data.test_times_coarse_list)
            print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikidata12k", help="choose a dataset: icews14, icews05-15, wikidata12k")
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="size of batch")
    parser.add_argument("--lr", type=float, default=0.001, help="")
    parser.add_argument("--dr", type=float, default=1.0, help="")
    parser.add_argument("--entity_dim", type=int, default=100, help="")
    parser.add_argument("--rel_dim", type=int, default=50, help="")
    parser.add_argument("--coarse_grain", type=str, default="years5", help="choose coarse grain if icews: year, quarterly, month; \
                                                                               if wikidata12k: century, decade, years5")
    parser.add_argument("--alpha", type=float, default=0.01, help="coefficient of the time smoothness term")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="label smoothing")
    parser.add_argument("--dropout1", type=float, default=0.5)
    parser.add_argument("--dropout2", type=float, default=0.5)

    args = parser.parse_args()

    if ((args.dataset == 'icews14' or args.dataset == 'icews05-15') and args.coarse_grain not in {'year', 'quarterly', 'month'}) or \
       (args.dataset == 'wikidata12k' and args.coarse_grain not in {'century', 'decade', 'years5'}):
       print('the input parameter "coarse_grain" is wrong, check it')
       exit(1)

    torch.backends.cudnn.deterministic = True
    seed = 100
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    print(args)

    data = Data(data_dir=args.dataset)
    trainer = Trainer(data, num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr, dr=args.dr,
                      entity_dim=args.entity_dim, rel_dim=args.rel_dim, coarse_grain=args.coarse_grain,
                      alpha = args.alpha, label_smoothing=args.label_smoothing, 
                      dropout=[args.dropout1, args.dropout2]
                     )
    trainer.train()


