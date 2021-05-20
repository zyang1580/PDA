import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
import os
import collections
import heapq
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.sparse as sp


plt.switch_backend('agg')


def load(load_path, filename, **kwargs):

    filename = os.path.join(load_path, filename)
    print(filename)
    record = sp.load_npz(filename)
    return record

class Data():

    def load_ori_data(self, args):
        self.path = './data/{}/'.format(args.dataset)
        if args.model == 'mf' or args.model == 'biasmf':
            if args.source=="dice":
                self.path = "data/ml10m_dice/"
                train_record = sp.load_npz(self.path+"train_coo_record.npz").tolil(copy=True)
                train_skew_record = sp.load_npz(self.path+"train_skew_coo_record.npz").tolil(copy=True)
                valid_record = sp.load_npz(self.path+"val_coo_record.npz").tolil(copy=True)
                test_record = sp.load_npz(self.path+"test_coo_record.npz").tolil(copy=True)
                self.n_users, self.n_items = train_record.shape[0], train_record.shape[1]
                self.n_train = int(np.sum(train_record)+np.sum(train_skew_record))
                self.n_test = int(np.sum(test_record))
                self.n_valid = int(np.sum(valid_record))
                for i in range(self.n_users):
                    self.train_user_list[i] = train_record.rows[i] + train_skew_record.rows[i]
                    self.test_user_list[i] = test_record.rows[i]
                    self.valid_user_list[i] = valid_record.rows[i]
                self.users = list(range(self.n_users))
                self.items = list(range(self.n_items))
                self.test_users = set(self.test_user_list.keys())
                self.valid_users = set(self.valid_user_list.keys())
            else:
                train_file = self.path + 'train.txt'
                valid_file = self.path + 'valid.txt'
                test_file = self.path + 'test.txt'
                with open(train_file) as f:
                    for line in f.readlines():
                        line = line.strip('\n').split(' ')
                        if len(line) == 0:
                            continue
                        line = [int(i) for i in line]
                        user = line[0]
                        items = line[1:]
                        if (len(items)==0):
                            continue
                        self.train_user_list[user] = items
                        for item in items:
                            self.train_item_list[item].append(user)
                        self.n_users = max(self.n_users, user)
                        self.n_items = max(self.n_items, max(items))
                        self.n_train += len(items)

                with open(valid_file) as f:
                    for line in f.readlines():
                        line = line.strip('\n').split(' ')
                        if len(line) == 0:
                            continue
                        line = [int(i) for i in line]
                        user = line[0]
                        items = line[1:]
                        if len(items) == 0:
                            continue
                        self.valid_user_list[user] = items
                        self.valid_items.update(set(items))
                        for item in items:
                            self.valid_item_list[item].append(user)
                        self.n_users = max(self.n_users, user)
                        self.n_items = max(self.n_items, max(items))
                        self.n_valid += len(items)

                with open(test_file) as f:
                    for line in f.readlines():
                        line = line.strip('\n').split(' ')
                        if len(line) == 0:
                            continue
                        line = [int(i) for i in line]
                        user = line[0]
                        items = line[1:]
                        if len(items) == 0:
                            continue
                        self.test_user_list[user] = items
                        for item in items:
                            self.test_item_list[item].append(user)
                        self.n_users = max(self.n_users, user)
                        self.n_items = max(self.n_items, max(items))
                        self.n_test += len(items)
                print(self.n_train,self.n_valid,self.n_test)
                self.n_users = self.n_users + 1
                self.n_items = self.n_items + 1
                self.users = list(range(self.n_users))
                self.items = list(range(self.n_items))


                # for i in range(self.n_users):
                #     for item in self.train_user_list[i]:
                #         self.train_item_list[item].append(i)
                #     for item in self.test_user_list[i]:
                #         self.test_item_list[item].append(i)
                #     for item in self.valid_user_list[i]:
                #         self.valid_item_list[item].append(i)
                #     self.n_train += len(self.train_user_list[i])
                #     self.n_test += len(self.test_user_list[i])
                #     self.n_valid += len(self.valid_user_list[i])
                self.valid_users = set(self.valid_user_list.keys())
                self.test_users = set(self.test_user_list.keys())
        elif args.model == 'CausalE' or args.model == 'IPSmf':
            if args.dataset == 'movielens_ml_10m' or args.dataset == 'movielens_ml_1m' or args.dataset == 'lastfm' or args.dataset == 'addressa'\
                                                                    or args.dataset == 'kwai' or args.dataset == 'globe':
                if args.skew == 1:
                    train_file = self.path + 'skew_train.txt'
                else:
                    train_file = self.path + 'train.txt'
                test_file = self.path + 'test.txt'

                with open(train_file) as f:
                    for line in f.readlines():
                        line = line.strip('\n').split(' ')
                        if len(line) == 0:
                            continue
                        line = [int(i) for i in line]
                        user = line[0]
                        items = line[1:]
                        self.train_user_list[user] = items
                        self.users.add(user)
                        self.n_users = max(self.n_users, user)
                        self.n_items = max(self.n_items, max(items))
                        # self.n_train += len(items)

                with open(test_file) as f:
                    for line in f.readlines():
                        line = line.strip('\n').split(' ')
                        if len(line) == 0:
                            continue
                        line = [int(i) for i in line]
                        user = line[0]
                        items = line[1:]
                        self.test_user_list[user] = items
                        self.n_users = max(self.n_users, user)
                        self.n_items = max(self.n_items, max(items))
                        # self.n_test += len(items)
                self.n_users = self.n_users + 1
                if args.dataset == 'movielens_ml_10m':
                    self.n_items = 8790
                elif args.dataset == 'movielens_ml_1m':
                    self.n_items = 3125
                elif args.dataset == 'lastfm':
                    self.n_items = 2822
                elif args.dataset == 'addressa':
                    self.n_items = 744
                elif args.dataset == 'kwai':
                    self.n_items = 80524
                elif args.dataset == 'globe':
                    self.n_items = 12005
                # self.users = list(range(self.n_users))
                self.items = list(range(self.n_items))
                self.users = list(self.users)



                countTrainItem = {}
                countTestItem = {}
                countTrainInters = [0,0,0]
                countTestInters = [0,0,0]
                for i in range(self.n_users):
                    for item in self.train_user_list[i]:
                        self.train_item_list[item].append(i)
                        if item not in countTrainItem.keys():
                            countTrainItem[item] = 0
                        countTrainItem[item] += 1
                    for item in self.test_user_list[i]:
                        self.test_item_list[item].append(i)
                        if item not in countTestItem:
                            countTestItem[item] = 0
                        countTestItem[item] += 1
                    self.n_train += len(self.train_user_list[i])
                    self.n_test += len(self.test_user_list[i])
                topNum = [0.01, 0.05, 0.1]
                topNum1 = [int(i * len(self.train_item_list)) for i in topNum]
                topNum2 = [int(i * len(self.test_item_list)) for i in topNum]
                # print(topNum1, topNum2)
                countTrainItem = [heapq.nlargest(i, countTrainItem, key = countTrainItem.get) for i in topNum1]
                countTestItem = [heapq.nlargest(i, countTestItem, key = countTestItem.get) for i in topNum2]
                # print(countTrainItem, countTestItem)

                for user, items in self.train_user_list.items():
                    for item in items:
                        for i in range(len(topNum)):
                            if item in countTrainItem[i]:
                                countTrainInters[i] += 1
                for user, items in self.test_user_list.items():
                    for item in items:
                        for i in range(len(topNum)):
                            if item in countTestItem[i]:
                                countTestInters[i] += 1
                
                for i in range(3):
                    print(countTrainInters[i]/self.n_train, countTestInters[i]/self.n_test, len(countTrainItem[i]), len(countTestItem[i]), len(set(countTestItem[i])&set(countTrainItem[i])))

    def load_imb_data(self):
        if args.model == 'mf':
            if args.dataset == 'movielens_ml_1m' or args.dataset == 'movielens_ml_1m_sorted' or args.dataset == 'movielens_ml_10m' or \
                                                                    args.dataset == 'movielens_ml_10m_sorted' or args.dataset == 'lastfm' or args.dataset == 'addressa' or args.dataset == 'globe':
                user_file = self.path + 'user_list.json'
                with open(user_file) as f:
                    temp_list = json.loads(f.read())
                    for user in temp_list:
                        items = temp_list[user]
                        #items = [int(i) for i in items]   #item_id starts from 0。
                        self.n_items = max(self.n_items, max(items))
                        self.user_list.append(items)
                self.n_users = len(self.user_list)
                self.n_items = self.n_items + 1
                self.users = list(range(self.n_users))
                self.items = list(range(self.n_items))
                # print(self.n_users, self.n_items)

                ratio = args.devide_ratio

                
                for i in range(len(self.user_list)):
                    items = self.user_list[i]
                    cutPoint = int(len(items) * ratio)
                    self.train_user_list[i] = items[:cutPoint]
                    self.test_user_list[i] = items[cutPoint:]
                    # self.train_user_list[i] = rd.sample(list(self.users), cutPoint)
                    # for item in items:
                    #     if item in self.train_user_list[i]:
                    #         pass
                    #     self.test_user_list[i].append(item)


                # print(countTrainItem)
                # print(countTestItem)
                # print(set(countTestItem) & set(countTrainItem))

            elif args.dataset == 'gowalla':
                train_file = self.path + 'train.txt'
                test_file = self.path + 'test.txt'

                with open(train_file) as f:
                    for line in f.readlines():
                        line = line.strip('\n').split(' ')
                        if len(line) == 0:
                            continue
                        line = [int(i) for i in line]
                        user = line[0]
                        items = line[1:]
                        self.train_user_list[user] = items
                        self.n_users = max(self.n_users, user)
                        self.n_items = max(self.n_items, max(items))

                with open(test_file) as f:
                    for line in f.readlines():
                        line = line.strip('\n').split(' ')
                        if len(line) == 0:
                            continue
                        line = [int(i) for i in line]
                        user = line[0]
                        items = line[1:]
                        self.test_user_list[user] = items
                        self.n_users = max(self.n_users, user)
                        self.n_items = max(self.n_items, max(items))
                self.n_users = self.n_users + 1
                self.n_items = self.n_items + 1
                self.users = list(range(self.n_users))
                self.items = list(range(self.n_items))

        elif args.model == 'CausalE' or args.model == 'IPSmf':
            if args.dataset == 'movielens_ml_10m' or args.dataset == 'movielens_ml_1m' or args.dataset == 'lastfm' or args.dataset == 'addressa'\
                                                                    or args.dataset == 'kwai' or args.dataset == 'globe':
                if args.skew == 1:
                    train_file = self.path + 'skew_train.txt'
                else:
                    train_file = self.path + 'train.txt'
                test_file = self.path + 'test.txt'

                with open(train_file) as f:
                    for line in f.readlines():
                        line = line.strip('\n').split(' ')
                        if len(line) == 0:
                            continue
                        line = [int(i) for i in line]
                        user = line[0]
                        items = line[1:]
                        self.train_user_list[user] = items
                        self.users.add(user)
                        self.n_users = max(self.n_users, user)
                        #self.n_items = max(self.n_items, max(items))

                with open(test_file) as f:
                    for line in f.readlines():
                        line = line.strip('\n').split(' ')
                        if len(line) == 0:
                            continue
                        line = [int(i) for i in line]
                        user = line[0]
                        items = line[1:]
                        self.test_user_list[user] = items
                        self.n_users = max(self.n_users, user)
                        #self.n_items = max(self.n_items, max(items))
                self.n_users = self.n_users + 1
                if args.dataset == 'movielens_ml_10m':
                    self.n_items = 8790
                elif args.dataset == 'movielens_ml_1m':
                    self.n_items = 3125
                elif args.dataset == 'lastfm':
                    self.n_items = 3646
                elif args.dataset == 'addressa':
                    self.n_items = 744
                elif args.dataset == 'kwai':
                    self.n_items = 80524
                elif args.dataset == 'globe':
                    self.n_items = 12005
                # self.users = list(range(self.n_users))
                self.items = list(range(self.n_items))
                self.users = list(self.users)


        countTrainItem = {}
        countTestItem = {}
        countTrainInters = [0,0,0]
        countTestInters = [0,0,0]
        for i in range(self.n_users):
            for item in self.train_user_list[i]:
                self.train_item_list[item].append(i)
                if item not in countTrainItem.keys():
                    countTrainItem[item] = 0
                countTrainItem[item] += 1
            for item in self.test_user_list[i]:
                self.test_item_list[item].append(i)
                if item not in countTestItem:
                    countTestItem[item] = 0
                countTestItem[item] += 1
            self.n_train += len(self.train_user_list[i])
            self.n_test += len(self.test_user_list[i])

        # t = [len(x) for x in self.train_item_list.values()]
        # t.sort(reverse=True)
        # print(t[0])

        topNum = [0.01, 0.05, 0.1]
        topNum1 = [int(i * len(self.train_item_list)) for i in topNum]
        topNum2 = [int(i * len(self.test_item_list)) for i in topNum]
        # print(topNum1, topNum2)
        countTrainItem_topk = [heapq.nlargest(i, countTrainItem, key = countTrainItem.get) for i in topNum1]
        countTestItem_topk = [heapq.nlargest(i, countTestItem, key = countTestItem.get) for i in topNum2]
        # print(countTrainItem, countTestItem)

        for user, items in self.train_user_list.items():
            for item in items:
                for i in range(len(topNum)):
                    if item in countTrainItem_topk[i]:
                        countTrainInters[i] += 1
        for user, items in self.test_user_list.items():
            for item in items:
                for i in range(len(topNum)):
                    if item in countTestItem_topk[i]:
                        countTestInters[i] += 1
                
        for i in range(3):
            print(countTrainInters[i]/self.n_train, countTestInters[i]/self.n_test, len(countTrainItem_topk[i]), len(countTestItem_topk[i]), len(set(countTestItem_topk[i])&set(countTrainItem_topk[i])))
        
        idxs = list(range(self.n_items))
        for idx in idxs:
            if idx not in countTrainItem.keys():
                countTrainItem[idx] = 0
        idxs.sort(key = lambda x:-countTrainItem[x])
        # print(countTrainItem[idxs[0]])
        # print(idxs).



        # self.plot_fit_pic(args, idxs, countTrainItem)


        user_max = args.user_max
        user_num_per_cls = []
        cls_num = self.n_items
        imb_factor = (1.0*args.user_min/args.user_max)# **(1.0/args.lam)
        # print(imb_factor)
        lam = args.lam
        if args.imb_type == 'exp':
            for cls_idx in self.items:
                num = user_max * (imb_factor**(lam*cls_idx / (cls_num - 1.0)))
                user_num_per_cls.append(max(1,int(num)))
        elif args.imb_type == 'step':
            topN = int(cls_num * args.top_ratio)
            user_max = countTrainItem[idxs[topN]]
            for cls_idx in range(topN):
                user_num_per_cls.append(int(user_max))
            for cls_idx in range(cls_num - topN):
                user_num_per_cls.append(int(user_max * imb_factor))
        
        self.train_user_list = collections.defaultdict(list)
        for item, user_num in zip(idxs, user_num_per_cls):
            item_list = self.train_item_list[item]
            # rd.shuffle(item_list)
            # self.train_item_list[item] = item_list[:user_num]
            item_list_num = len(item_list)
            if item_list_num > user_num:
                self.train_item_list[item] = item_list[item_list_num-user_num:]
            else:
                self.train_item_list[item] = item_list
        
        for item, users in self.train_item_list.items():
            for user in users:
                self.train_user_list[user].append(item)
        self.users = list(set(self.train_user_list.keys()))
        # print(len(self.users))

    def plot_pics(self):
        y = []
        for item, users in self.train_item_list.items():
            y.append(len(users))
        y = np.asarray(y)

        points = [10, 50, 100, 200, 500]
        count = [0, 0, 0, 0, 0, 0]
        area_sum = [0, 0, 0, 0, 0, 0]
        rate = []
        belong = []
        sorted_id = np.argsort(y)
        y = y[sorted_id]
        p = 0
        for n, score in enumerate(y):
            while p!=5 and points[p] < score:
                p += 1
            count[p] += 1
            area_sum[p] += score
            belong.append(p)

        for i in range(len(count)):
            rate.append(1.0*count[i]/self.n_items)



        y = []
        for user, items in self.train_user_list.items():
            y.append(len(items))
        y = np.asarray(y)

        points = [5, 7, 10, 15, 20]
        count = [0, 0, 0, 0, 0, 0]
        area_sum = [0, 0, 0, 0, 0, 0]
        userrate = []
        userbelong = []
        usersorted_id = np.argsort(y)
        y = y[usersorted_id]
        p = 0
        for n, score in enumerate(y):
            while p!=5 and points[p] < score:
                p += 1
            count[p] += 1
            area_sum[p] += score
            userbelong.append(p)

        for i in range(len(count)):
            userrate.append(1.0*count[i]/self.n_users)
                


        return sorted_id, belong, rate, usersorted_id, userbelong, userrate
    
    def plot_fit_pic(self, args, idxs, countTrainItem):
        # print(123123131312312)
        N_max = countTrainItem[idxs[0]]
        N_min = max(countTrainItem[idxs[-1]], 1)
        print(N_max, N_min)
        # print(len(self.train_item_list[idxs[-1]]))
        a = N_min/N_max
        N = self.n_items
        x = np.linspace(0, 1, N)
        y = [countTrainItem[idxs[i]] for i in range(N)]
        # print(y)
        # x = x[:100]
        # y = y[:100]
        y = np.asarray(y)
        plt.plot(x, y)

        def func(x, lam):
            return N_max*((N_min/N_max)**(lam*x))
        popt, pcov = curve_fit(func, x, y)
        y_hat = [func(i, popt[0]) for i in x]
        plt.plot(x, y_hat, 'r--')
        plt.savefig('./figures/{}_lambda={}.png'.format(args.dataset, popt[0]))
        s = 0
        for i in range(N):
            s += (y_hat[i]-y[i])**2
        print(s, s/N)

    def __init__(self, args):
        self.path = args.data_path + args.dataset + '/'
        self.batch_size = args.batch_size

        self.n_users, self.n_items, self.n_valid = 0, 0, 0
        self.n_train, self.n_test = 0, 0
        self.user_list = []
        self.item_list = []
        self.valid_users = set()
        self.valid_items = set()
        self.train_user_list = collections.defaultdict(list)
        self.test_user_list = collections.defaultdict(list)
        self.train_item_list = collections.defaultdict(list)
        self.test_item_list = collections.defaultdict(list)
        self.valid_user_list = collections.defaultdict(list)
        self.valid_item_list = collections.defaultdict(list)
        self.users = set()
        self.items = set()
        
        #print(os.getcwd())
        
        if args.data_type == 'ori':
            self.load_ori_data(args)
        else:
            self.load_imb_data(args)
        
        self.valid_users = list(self.valid_users)
        self.valid_items = list(self.valid_items)
        # self.plot_pics(args)
        print('n_items:', self.n_items, 'n_users:', self.n_users)
        sum = 0
        for item, users in self.train_item_list.items():
            sum += len(users)
        print("sparsity:", 1.0*sum/self.n_items/self.n_users)

        
        
    def add_expo_popularity(self,popularity):
        self.expo_popularity = popularity

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.users, self.batch_size)
        else:
            users = [rd.choice(self.users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []

        for user in users:
            if self.train_user_list[user] == []:
                pos_items.append(0)
            else:
                pos_items.append(rd.choice(self.train_user_list[user]))
            while True:
                neg_item = rd.choice(self.items)
                if neg_item not in self.train_user_list[user]:
                    neg_items.append(neg_item)
                    break

        # for i in range(len(users)):
        #     '''
        #     founction??
        #     '''
        #     if pos_items[i] >= self.n_items:
        #         neg_items[i] += self.n_items

        return users, pos_items, neg_items

    def sample2(self):
        if self.batch_size <= len(self.valid_users):
            users = rd.sample(self.valid_users, self.batch_size)
        else:
            users = [rd.choice(self.valid_users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []

        for user in users:
            if self.valid_user_list[user] == []:
                pos_items.append(0)
            else:
                pos_items.append(rd.choice(self.valid_user_list[user]))
            while True:
                neg_item = rd.choice(self.items)
                if neg_item not in (self.valid_user_list[user]+self.train_user_list[user]):
                    neg_items.append(neg_item)
                    break

        return users, pos_items, neg_items

    def sample_test(self):
        if self.batch_size <= len(self.test_users):
            users = rd.sample(self.test_users, self.batch_size)
        else:
            users = [rd.choice(self.test_users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []

        for user in users:
            if self.test_user_list[user] == []:
                pos_items.append(0)
            else:
                pos_items.append(rd.choice(self.test_user_list[user]))
            while True:
                neg_item = rd.choice(self.items)
                if neg_item not in (self.test_user_list[user]+self.train_user_list[user]):
                    neg_items.append(neg_item)
                    break

        return users, pos_items, neg_items

import pandas as pd
class Data2():
    def load_ori_data(self, args):
        self.path = './data/{}/'.format(args.dataset)
        if args.model == 'mf' or args.model == 'biasmf':
            train_file = self.path + 'train_with_time.txt'
            valid_file = self.path + 'valid.txt'
            test_file = self.path + 'test.txt'

            # -------------------  load train data ------------------------
            train_data = pd.read_csv(train_file,header=None,sep=' ')
            train_data.columns = ['uid','iid','time','stars']
            train_data = train_data[['uid','iid','time']]
            unique_time = train_data['time'].unique()
            print("time slot unique in train:",unique_time)
            self.unique_times = list(unique_time)
            if train_data['time'].unique().shape[0] < 2:
                raise RuntimeWarning("there only one time slot for train...., this may cause our method not work")
            
            for col in train_data.columns:
                train_data[col] = train_data[col].astype(int)
            user_item_time = train_data.groupby('uid')[('iid','time')].agg(list)
            self.train_user_list = dict(zip(user_item_time.index,user_item_time['iid']))
            self.train_user_list_time = dict(zip(user_item_time.index,user_item_time['time']))
            item_user = train_data.groupby('iid')[('uid','time')].agg(list)
            print(item_user.head(2))
            self.train_item_list = dict(zip(item_user.index,item_user['uid']))

            self.n_users = max(self.n_users,train_data['uid'].max())
            self.n_items = max(self.n_items,train_data['iid'].max())
            self.n_train = train_data.shape[0]

            # ------------------ load  test and valuation
                
            # with open(train_file) as f:
            #     for line in f.readlines():
            #         line = line.strip('\n').split(' ')
            #         if len(line) == 0:
            #             continue
            #         line = [int(i) for i in line]
            #         user = line[0]
            #         items = line[1:]
            #         if (len(items)==0):
            #             continue
            #         self.train_user_list[user] = items
            #         for item in items:
            #             self.train_item_list[item].append(user)
            #         self.n_users = max(self.n_users, user)
            #         self.n_items = max(self.n_items, max(items))
            #         self.n_train += len(items)

            with open(valid_file) as f:
                for line in f.readlines():
                    line = line.strip('\n').split(' ')
                    if len(line) == 0:
                        continue
                    line = [int(i) for i in line]
                    user = line[0]
                    items = line[1:]
                    if len(items) == 0:
                        continue
                    self.valid_user_list[user] = items
                    self.valid_items.update(set(items))
                    for item in items:
                        self.valid_item_list[item].append(user)
                    self.n_users = max(self.n_users, user)
                    self.n_items = max(self.n_items, max(items))
                    self.n_valid += len(items)

            with open(test_file) as f:
                for line in f.readlines():
                    line = line.strip('\n').split(' ')
                    if len(line) == 0:
                        continue
                    line = [int(i) for i in line]
                    user = line[0]
                    items = line[1:]
                    if len(items) == 0:
                        continue
                    self.test_user_list[user] = items
                    for item in items:
                        self.test_item_list[item].append(user)
                    self.n_users = max(self.n_users, user)
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
            print(self.n_train,self.n_valid,self.n_test)
            self.n_users = self.n_users + 1
            self.n_items = self.n_items + 1
            self.users = list(range(self.n_users))
            self.items = list(range(self.n_items))

            self.valid_users = set(self.valid_user_list.keys())
            self.test_users = set(self.test_user_list.keys())
        else:
            raise NotImplementedError("only can sampling for mf-type model")
        



       

    def __init__(self, args):
        self.path = args.data_path + args.dataset + '/'
        self.batch_size = args.batch_size

        self.n_users, self.n_items, self.n_valid = 0, 0, 0
        self.n_train, self.n_test = 0, 0
        self.user_list = []
        self.item_list = []
        self.valid_users = set()
        self.valid_items = set()
        self.train_user_list = collections.defaultdict(list)
        self.train_user_list_time = collections.defaultdict(list)
        self.test_user_list = collections.defaultdict(list)
        self.train_item_list = collections.defaultdict(list)
        self.test_item_list = collections.defaultdict(list)
        self.valid_user_list = collections.defaultdict(list)
        self.valid_item_list = collections.defaultdict(list)
        self.users = set()
        self.items = set()
        
        #print(os.getcwd())
        
        if args.data_type == 'ori':
            self.load_ori_data(args)
        else:
            self.load_imb_data(args)
        
        self.valid_users = list(self.valid_users)
        self.valid_items = list(self.valid_items)
        # self.plot_pics(args)
        print('n_items:', self.n_items, 'n_users:', self.n_users)
        sum = 0
        for item, users in self.train_item_list.items():
            sum += len(users)
        print("sparsity:", 1.0*sum/self.n_items/self.n_users)

    def add_expo_popularity(self,popularity):
        self.expo_popularity = popularity
    def plot_pics(self):
        y = []
        for item, users in self.train_item_list.items():
            y.append(len(users))
        y = np.asarray(y)

        points = [10, 50, 100, 200, 500]
        count = [0, 0, 0, 0, 0, 0]
        area_sum = [0, 0, 0, 0, 0, 0]
        rate = []
        belong = []
        sorted_id = np.argsort(y)
        y = y[sorted_id]
        p = 0
        for n, score in enumerate(y):
            while p!=5 and points[p] < score:
                p += 1
            count[p] += 1
            area_sum[p] += score
            belong.append(p)

        for i in range(len(count)):
            rate.append(1.0*count[i]/self.n_items)



        y = []
        for user, items in self.train_user_list.items():
            y.append(len(items))
        y = np.asarray(y)

        points = [5, 7, 10, 15, 20]
        count = [0, 0, 0, 0, 0, 0]
        area_sum = [0, 0, 0, 0, 0, 0]
        userrate = []
        userbelong = []
        usersorted_id = np.argsort(y)
        y = y[usersorted_id]
        p = 0
        for n, score in enumerate(y):
            while p!=5 and points[p] < score:
                p += 1
            count[p] += 1
            area_sum[p] += score
            userbelong.append(p)

        for i in range(len(count)):
            userrate.append(1.0*count[i]/self.n_users)
                


        return sorted_id, belong, rate, usersorted_id, userbelong, userrate  


    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.users, self.batch_size)
        else:
            users = [rd.choice(self.users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []

        for user in users:
            if self.train_user_list[user] == []:
                pos_items.append(0)
            else:
                pos_items.append(rd.choice(self.train_user_list[user]))
            while True:
                neg_item = rd.choice(self.items)
                if neg_item not in self.train_user_list[user]:
                    neg_items.append(neg_item)
                    break

        # for i in range(len(users)):
        #     '''
        #     founction??
        #     '''
        #     if pos_items[i] >= self.n_items:
        #         neg_items[i] += self.n_items

        return users, pos_items, neg_items

    def sample2(self):
        if self.batch_size <= len(self.valid_users):
            users = rd.sample(self.valid_users, self.batch_size)
        else:
            users = [rd.choice(self.valid_users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []

        for user in users:
            if self.valid_user_list[user] == []:
                pos_items.append(0)
            else:
                pos_items.append(rd.choice(self.valid_user_list[user]))
            while True:
                neg_item = rd.choice(self.items)
                if neg_item not in (self.valid_user_list[user]+self.train_user_list[user]):
                    neg_items.append(neg_item)
                    break

        return users, pos_items, neg_items

    def sample_test(self):
        if self.batch_size <= len(self.test_users):
            users = rd.sample(self.test_users, self.batch_size)
        else:
            users = [rd.choice(self.test_users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []

        for user in users:
            if self.test_user_list[user] == []:
                pos_items.append(0)
            else:
                pos_items.append(rd.choice(self.test_user_list[user]))
            while True:
                neg_item = rd.choice(self.items)
                if neg_item not in (self.test_user_list[user]+self.train_user_list[user]):
                    neg_items.append(neg_item)
                    break

        return users, pos_items, neg_items