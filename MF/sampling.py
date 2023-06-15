from NeuRec.MF.load_data import Data
import numpy as np
from prefetch_generator import background

@background(max_prefetch=3)
def multi_sampling():
    worker = 10
    pool = multiprocessing.Pool(worker)
    all_users = data.train_user_list.keys()
    sampled_data  = pool.map(sampling_one_user,all_users)
    users = []
    pos_items = []
    neg_items = []
    for re in sampled_data:
        users.extend(re['user'])
        pos_items.extend(re['pos'])
        neg_items.append(re['neg'])
    return users,pos_items,neg_items

def sampling_one_user(u):
    pos_items = data.train_user_list[u]
    N_ps = len(pos_items)
    neg_items = []
    n_items = data.n_items
    for i in range(N_ps):
        one_neg = np.random.randint(n_items)
        while one_neg in pos_items:
            one_neg = np.random.randint(n_items)
        neg_items.append(one_neg)
    users = [u] * N_ps
    return {'user':users,'pos':pos_items,'neg':neg_items}


def _batch_sampling(itr, pos_dict, neg_dict, tot_neg, batch_epoch, p_thre, item_max, neg_pro_dict):
    '''
    subprocess
    :param itr:
    :param pos_dict:
    :param neg_dict:
    :param tot_neg:
    :return:
    '''
    neg_items = np.zeros([itr.shape[0], tot_neg])
    expo_flag = np.zeros([itr.shape[0], tot_neg])
    p = np.random.rand(itr.shape[0], tot_neg)
    k = 0
    for x in itr:
        u = x[0]
        try:
            idx1 = np.where(p[k] <= p_thre)[0]
            idx2 = np.where(p[k] > p_thre)[0]
            neg_items[k, idx1] = np.random.choice(neg_dict[u], size=idx1.shape[0])
            expo_flag[k, idx1] += 1
            for idx2_i in idx2:
                temp = np.random.randint(item_max)
                while temp in pos_dict[u]:
                    temp = np.random.randint(item_max)
                neg_items[k, idx2_i] = temp

        except:
            idx2 = np.arange(tot_neg)
            for idx2_i in idx2:
                temp = np.random.randint(item_max)
                while temp in pos_dict[u]:
                    temp = np.random.randint(item_max)
                neg_items[k, idx2_i] = temp
        k += 1
    expo_flag = expo_flag.reshape(itr.shape[0], batch_epoch, -1)
    neg_items = neg_items.reshape(itr.shape[0], batch_epoch, -1)
    pos_flag = np.ones([expo_flag.shape[0], expo_flag.shape[1], 1])
    expo_flag = np.concatenate([pos_flag, expo_flag], axis=-1)
    return [itr,np.concatenate([neg_items,expo_flag],axis=-1)]

def _batch_sampling2(itr,pos_dict,neg_dict,tot_neg,batch_epoch,p_thre,item_max,neg_pro_dict):
    '''
    subprocess
    :param itr:
    :param pos_dict:
    :param neg_dict:
    :param tot_neg:
    :return:
    '''
    neg_items = np.zeros([itr.shape[0], tot_neg])
    expo_flag = np.zeros([itr.shape[0], tot_neg])
    p = np.random.rand(itr.shape[0], tot_neg)
    k = 0
    for x in itr:
        u = x[0]
        pos_item = pos_dict[u]
        try:
            neg_item_u = neg_dict[u]
            idx1 = np.where(p[k] <= p_thre)[0]
            idx2 = np.where(p[k] > p_thre)[0]
        except:
            idx1 = None
            idx2 = np.arange(tot_neg)
        if idx1 is not None and idx1.shape[0]>0:
            neg_items[k, idx1] = np.random.choice(neg_item_u, size=idx1.shape[0])
            expo_flag[k, idx1] += 1
        if idx2.shape[0] > 0:
            l2 = idx2.shape[0]
            tmp = np.random.randint(item_max, size=l2 * 5) # sampling 5 times items
            tmp = np.setdiff1d(tmp, pos_item,True)
            if tmp.shape[0] >= l2:  # sampling enough
                neg_items[k, idx2] = tmp[:l2]
            else:                   # not enough
                tmp = np.random.randint(item_max, size=l2 * 10) #sampling more
                tmp = np.setdiff1d(tmp, pos_item,assume_unique=True)
                l_t = min(tmp.shape[0], l2)
                idx2_t = idx2[:l_t]
                neg_items[k, idx2_t] = tmp[:l_t]            # saving not in pos
                for idx2_i in idx2[l_t:]:                   # sampling others
                    temp = np.random.randint(item_max)
                    while temp in pos_item:
                        temp = np.random.randint(item_max)
                    neg_items[k, idx2_i] = temp
        k += 1
    expo_flag = expo_flag.reshape(itr.shape[0], batch_epoch, -1)
    neg_items = neg_items.reshape(itr.shape[0], batch_epoch, -1)
    pos_flag = np.ones([expo_flag.shape[0], expo_flag.shape[1], 1])
    expo_flag = np.concatenate([pos_flag, expo_flag], axis=-1)
    return [itr,np.concatenate([neg_items,expo_flag],axis=-1)]

def _batch_sampling3(itr,pos_dict,neg_dict,tot_neg,batch_epoch,p_thre,item_max,neg_pro_dict):
    '''
    subprocess, this process in random sampling stage, we will make sure that the sampled items not from neg interactions.
    :param itr:
    :param pos_dict:
    :param neg_dict:
    :param tot_neg:
    :param neg_pro_dict: probability of sampling for items in the neg_dict
    :return:
    '''
    neg_items = np.zeros([itr.shape[0], tot_neg])
    expo_flag = np.zeros([itr.shape[0], tot_neg])
    p = np.random.rand(itr.shape[0], tot_neg)
    k = 0
    for x in itr:
        u = x[0]
        pos_item = pos_dict[u]
        try:
            neg_item_u = neg_dict[u]
            if neg_pro_dict is not None:
                neg_item_p = neg_pro_dict[u]
            else:
                neg_item_p = None
            idx1 = np.where(p[k] <= p_thre)[0]
            idx2 = np.where(p[k] > p_thre)[0]
        except:
            idx1 = None
            neg_item_u = None
            idx2 = np.arange(tot_neg)
        if idx1 is not None and idx1.shape[0]>0:
            neg_items[k, idx1] = np.random.choice(neg_item_u, size=idx1.shape[0],p=neg_item_p)
            expo_flag[k, idx1] += 1
        if idx2.shape[0] > 0:
            l2 = idx2.shape[0]
            tmp = np.random.randint(0, item_max, size=l2 * 10) # sampling 5 times items
            if neg_item_u is not None:
                itr_items = np.concatenate([pos_item, neg_item_u], axis=0)
            else:
                itr_items = pos_item
            tmp = np.setdiff1d(tmp, itr_items,assume_unique=True)
            if tmp.shape[0] >= l2:  # sampling enough
                neg_items[k, idx2] = tmp[:l2]
            else:                   # not enough
                tmp = np.random.randint(0, item_max, size=l2 * 20) #sampling more
                tmp = np.setdiff1d(tmp, itr_items,assume_unique=True)
                l_t = min(tmp.shape[0], l2)
                idx2_t = idx2[:l_t]
                neg_items[k, idx2_t] = tmp[:l_t]            # saving not in pos
                for idx2_i in idx2[l_t:]:                   # sampling others
                    temp = np.random.randint(item_max)
                    while temp in itr_items:
                        temp = np.random.randint(item_max)
                    neg_items[k, idx2_i] = temp
        k += 1
    expo_flag = expo_flag.reshape(itr.shape[0], batch_epoch, -1)
    neg_items = neg_items.reshape(itr.shape[0], batch_epoch, -1)
    pos_flag = np.ones([expo_flag.shape[0], expo_flag.shape[1], 1])
    expo_flag = np.concatenate([pos_flag, expo_flag], axis=-1)
    return [itr,np.concatenate([neg_items,expo_flag],axis=-1)]
