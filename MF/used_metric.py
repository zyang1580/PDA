import numpy as np
import multiprocessing

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    try:
        r = r[:k]
    except:
        print(r)
        raise ImportError('error r')
    return np.mean(r)


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, maxlen, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    tp = 1. / np.log2(np.arange(2, k + 2))
    dcg_max = (tp[:min(maxlen, k)]).sum()
    if not dcg_max:
        return 0.
    r_k = r[:k]
    dcg_at_k_ = (r_k * tp).sum() 
    return dcg_at_k_ / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = r[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = r[:k]
    return min(1.,np.sum(r))
    

def get_r(user_pos_test, r):
    r_new = np.isin(r,user_pos_test).astype(np.float)
    return r_new

def get_performance(user_pos_test, r, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []
    r = get_r(user_pos_test,r)
    for K in Ks:
        precision.append(precision_at_k(r, K))#P = TP/ (TP+FP)
        recall.append(recall_at_k(r, K, len(user_pos_test)))#R = TP/ (TP+FN)
        ndcg.append(ndcg_at_k(r, K, len(user_pos_test)))
        hit_ratio.append(hit_at_k(r, K))#HR = SIGMA(TP) / SIGMA(test_set)
    # print(hit_ratio)

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}

def test_one_user(u):
    # user u's ratings for user u
    try:
        user_pos_test = test_user_list[u]
    except:
        user_pos_test = []
    r = rec_user_list[u]
    #print(len(r))
    return get_performance(user_pos_test, r, Ks)

cores=15
def test():
    pool = multiprocessing.Pool(cores)
    test_user = list(rec_user_list.keys())
    print('test_user number:',len(test_user))
    bat_result = pool.map(test_one_user, test_user)
    return bat_result
