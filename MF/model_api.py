from numpy.core.fromnumeric import shape
import tensorflow as tf
import numpy as np
import os
import sys
import random
import collections
from parse import parse_args
import scipy.sparse as sp
import heapq
import math
import time

class ConditionalBPRMF:
    '''
    PD/PDA
    PDG/PDG-A
    '''
    def __init__(self, args, data_config, use_dataset_api=False,users_api=None,pos_items_api=None,neg_items_api=None,pos_pop_api=None,neg_pop_api=None):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.decay = args.regs
        self.emb_dim = args.embed_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.c = args.c
        self.alpha = args.alpha
        self.beta = args.beta

        if use_dataset_api:
            self.users = users_api
            self.pos_items = pos_items_api
            self.neg_items = neg_items_api
            self.pos_pop = pos_pop_api
            self.neg_pop = neg_pop_api
        else:
            #placeholders
            self.users = tf.placeholder(tf.int32, shape = (None,))
            self.pos_items = tf.placeholder(tf.int32, shape = (None,))
            self.neg_items = tf.placeholder(tf.int32, shape = (None,))

            self.pos_pop = tf.placeholder(tf.float32,shape = (None,))
            self.neg_pop = tf.placeholder(tf.float32,shape = (None,))

        #initiative weights
        self.weights = self.init_weights()

        #neting
        user_embedding = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
        # user_rand_embedding = tf.nn.embedding_lookup(self.weights['user_rand_embedding'], self.users)
        # item_rand_embedding = tf.nn.embedding_lookup(self.weights['item_rand_embedding'], self.pos_items)
        

        self.const_embedding = self.weights['c']
        self.user_c = tf.nn.embedding_lookup(self.weights['user_c'], self.users)

        '''PD/PDG result without injecting popularity'''
        self.batch_ratings = tf.matmul(user_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)    #prediction, shape(user_embedding) != shape(pos_item_embedding)
        # self.user_const_ratings = self.batch_ratings - tf.matmul(self.const_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)   #auto tile
        # self.item_const_ratings = self.batch_ratings - tf.matmul(user_embedding, self.const_embedding, transpose_a=False, transpose_b = True)       #auto tile
        # self.user_rand_ratings = self.batch_ratings - tf.matmul(user_rand_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)
        # self.item_rand_ratings = self.batch_ratings - tf.matmul(user_embedding, item_rand_embedding, transpose_a=False, transpose_b = True)


        self.mf_loss, self.reg_loss = self.create_bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss = self.mf_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        # trainable_v1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'parameter')
        # self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list = trainable_v1)
        # two branch
        # self.w = tf.Variable(self.initializer([self.emb_dim,1]), name = 'item_branch')
        # self.w_user = tf.Variable(self.initializer([self.emb_dim,1]), name = 'user_branch')
        # self.sigmoid_yu = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['user_embedding'], self.w_user)))
        # self.sigmoid_yi = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['item_embedding'], self.w)))
        # two branch bpr
        self.mf_loss_pop_global, self.reg_loss_pop_global = self.create_bpr_loss_with_pop_global(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_pop_global = self.mf_loss_pop_global + self.reg_loss_pop_global
        self.opt_pop_global = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_pop_global)
        self._statistics_params()

    def init_weights(self):
        weights = dict()
        self.initializer = tf.contrib.layers.xavier_initializer()
        initializer = self.initializer
        with tf.variable_scope('parameter'):
            weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_embedding')
            weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_embedding')
            weights['user_rand_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_rand_embedding', trainable = False)
            weights['item_rand_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_rand_embedding', trainable = False)
        with tf.variable_scope('const_embedding'):
            self.rubi_c = tf.Variable(tf.zeros([1]), name = 'rubi_c')
            weights['c'] = tf.Variable(tf.zeros([1, self.emb_dim]), name = 'c')
        weights['user_c'] = tf.Variable(tf.zeros([self.n_users, 1]), name = 'user_c_v')
        return weights
    

    def create_bpr_loss_with_pop_global(self, users, pos_items, neg_items): # this global does not refer to global popularity, just a name
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item stop

        pos_scores = tf.nn.elu(pos_scores) + 1
        neg_scores = tf.nn.elu(neg_scores) + 1
        pos_scores_with_pop = tf.multiply(pos_scores,self.pos_pop)
        neg_scores_with_pop = tf.multiply(neg_scores,self.neg_pop)

        maxi = tf.log(tf.nn.sigmoid(pos_scores_with_pop - neg_scores_with_pop)+1e-10)
        self.condition_ratings = (tf.nn.elu(self.batch_ratings) + 1 ) * tf.squeeze(self.pos_pop)
        self.mf_loss_ori = tf.negative(tf.reduce_mean(maxi))
        mf_loss = self.mf_loss_ori
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores)+1e-10)

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # first branch
        # fusion
        mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    # def create_bpr_loss2(self, users, const_embedding, pos_items, neg_items):
    #     pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) - tf.matmul(const_embedding, pos_items, transpose_a=False, transpose_b = True)
    #     neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) - tf.matmul(const_embedding, neg_items, transpose_a=False, transpose_b = True)

    #     regularizer = tf.nn.l2_loss(const_embedding)
    #     regularizer = regularizer/self.batch_size

    #     maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

    #     mf_loss = tf.negative(tf.reduce_mean(maxi))
    #     reg_loss = self.decay * regularizer
    #     return mf_loss, reg_loss
    
    # def create_bce_loss2(self, users, const_embedding, pos_items, neg_items):
    #     pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) - tf.matmul(const_embedding, pos_items, transpose_a=False, transpose_b = True)
    #     neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) - tf.matmul(const_embedding, neg_items, transpose_a=False, transpose_b = True)

    #     regularizer = tf.nn.l2_loss(const_embedding)
    #     regularizer = regularizer/self.batch_size

    #     mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
    #     reg_loss = self.decay * regularizer
    #     return mf_loss, reg_loss

    def update_c(self, sess, c):
        sess.run(tf.assign(self.rubi_c, c*tf.ones([1])))

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

# class U_ConditionalBPRMF:
#     '''
#     Not used
#     '''
#     def __init__(self, args, data_config, use_dataset_api=False,users_api=None,pos_items_api=None,neg_items_api=None,pos_pop_api=None,neg_pop_api=None):
#         self.n_users = data_config['n_users']
#         self.n_items = data_config['n_items']
#
#         self.decay = args.regs
#         self.emb_dim = args.embed_size
#         self.lr = args.lr
#         self.batch_size = args.batch_size
#         self.verbose = args.verbose
#         self.exp_init_values = args.exp_init_values
#         print("init exp:",self.exp_init_values)
#
#         if use_dataset_api:
#             self.users = users_api
#             self.pos_items = pos_items_api
#             self.neg_items = neg_items_api
#             self.pos_pop = pos_pop_api
#             self.neg_pop = neg_pop_api
#         else:
#             #placeholders
#             self.users = tf.placeholder(tf.int32, shape = (None,))
#             self.pos_items = tf.placeholder(tf.int32, shape = (None,))
#             self.neg_items = tf.placeholder(tf.int32, shape = (None,))
#
#             self.pos_pop = tf.placeholder(tf.float32,shape = (None,))
#             self.neg_pop = tf.placeholder(tf.float32,shape = (None,))
#
#         #initiative weights
#         self.weights = self.init_weights()
#
#         #neting
#         user_embedding = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
#         pos_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
#         neg_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
#         users_exp = tf.nn.embedding_lookup(self.weights['user_pop_exp'], self.users)
#
#
#
#
#         self.batch_ratings = tf.matmul(user_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)    #prediction, shape(user_embedding) != shape(pos_item_embedding)
#
#         item_pop = tf.reshape(self.pos_pop,(1,-1))
#         users_exp_ = tf.reshape(users_exp,(-1,1))
#         self.item_pop_exp = tf.pow(item_pop, users_exp_, name='power')
#         self.condition_ratings = tf.multiply( (tf.nn.elu(self.batch_ratings) + 1 ), self.item_pop_exp)
#
#
#
#
#         self.mf_loss_pop_global, self.reg_loss_pop_global = self.create_bpr_loss_with_pop_global(user_embedding, pos_item_embedding, neg_item_embedding, users_exp)
#         self.loss_pop_global = self.mf_loss_pop_global + self.reg_loss_pop_global
#         self.opt_pop_global = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_pop_global,var_list=[self.weights['user_embedding'],self.weights['item_embedding']])
#         # self.opt_pop_exp = tf.train.GradientDescentOptimizer(learning_rate=self.lr*10).minimize(self.loss_pop_global,var_list=[self.weights['user_pop_exp']])
#         self.opt_pop_exp = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_pop_global,var_list=[self.weights['user_pop_exp']])
#         self.opt = tf.group(self.opt_pop_global, self.opt_pop_exp)
#         self.clip_op = tf.assign(self.weights['user_pop_exp'], tf.clip_by_value(self.weights['user_pop_exp'], 0.01, 0.5))
#         self._statistics_params()
#
#     def init_weights(self):
#         weights = dict()
#         self.initializer = tf.contrib.layers.xavier_initializer()
#         initializer = self.initializer
#         self.initializer_exp = tf.constant_initializer(self.exp_init_values)
#         initializer_exp = self.initializer_exp
#         with tf.variable_scope('parameter'):
#             weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_embedding')
#             weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_embedding')
#             weights['user_pop_exp'] = tf.Variable(initializer_exp([self.n_users, 1]), name = 'pop_exp') #, trainable=False
#         return weights
#
#
#     def create_bpr_loss_with_pop_global(self, users, pos_items, neg_items, users_exp):
#         pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
#         neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
#         # item stop
#
#         pos_scores = tf.nn.elu(pos_scores) + 1
#         neg_scores = tf.nn.elu(neg_scores) + 1
#
#         pos_pop = tf.pow(self.pos_pop, tf.squeeze(users_exp))
#         neg_pop = tf.pow(self.neg_pop, tf.squeeze(users_exp))
#
#         pos_scores_with_pop = tf.multiply(pos_scores,pos_pop)
#         neg_scores_with_pop = tf.multiply(neg_scores,neg_pop)
#
#         maxi = tf.log(tf.nn.sigmoid(pos_scores_with_pop - neg_scores_with_pop)+1e-10)
#         self.mf_loss_ori = tf.negative(tf.reduce_mean(maxi))
#         mf_loss = self.mf_loss_ori
#         # regular
#         regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
#         regularizer = regularizer/self.batch_size
#
#         reg_loss = self.decay * regularizer
#         return mf_loss, reg_loss
#
#     def _statistics_params(self):
#         # number of params
#         total_parameters = 0
#         for variable in self.weights.values():
#             shape = variable.get_shape()  # shape is an array of tf.Dimension
#             variable_parameters = 1
#             for dim in shape:
#                 variable_parameters *= dim.value
#             total_parameters += variable_parameters
#         if self.verbose > 0:
#             print("#params: %d" % total_parameters)



class BPRMFTempPop:
    '''
    BPRMF(t)-pop
    '''
    def __init__(self, args, data_config, use_dataset_api=False, users_api=None, pos_items_api=None, neg_items_api=None,
                 temp_api=None, raw_api=None):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.temp_num = data_config['temp_num']

        self.decay = args.regs
        self.emb_dim = args.embed_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        

        if use_dataset_api:
            self.users = users_api
            self.pos_items = pos_items_api
            self.neg_items = neg_items_api
            self.temp = temp_api
            self.raw = raw_api
        else:
            # placeholders
            self.users = tf.placeholder(tf.int32, shape=(None,))
            self.pos_items = tf.placeholder(tf.int32, shape=(None,))
            self.neg_items = tf.placeholder(tf.int32, shape=(None,))

            self.temp = tf.placeholder(tf.int32, shape=(None,))
            self.row = tf.placeholder(tf.int32, shape=(None,))

        # initiative weights
        self.weights = self.init_weights()

        # neting
        user_embedding = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
        user_temp_bias_all = tf.nn.embedding_lookup(self.weights['user_temp_bias'],self.users)
        pos_item_temp_init_bias_all = tf.nn.embedding_lookup(self.weights['item_temp_init_bias'],self.pos_items)
        neg_item_temp_init_bias_all = tf.nn.embedding_lookup(self.weights['item_temp_init_bias'],self.neg_items)
        index = tf.concat((tf.reshape(self.raw,(-1,1)),tf.reshape(self.temp,(-1,1))),axis=-1)
        user_temp_bias = tf.gather_nd(user_temp_bias_all,index) # temp bias
        pos_item_init_bias = tf.gather(pos_item_temp_init_bias_all,self.temp_num,axis=-1) # init bias
        pos_item_temp_bias = tf.gather_nd(pos_item_temp_init_bias_all,index)  # temp bias
        neg_item_init_bias = tf.gather(neg_item_temp_init_bias_all,self.temp_num,axis=-1) # init bias
        neg_item_temp_bias = tf.gather_nd(neg_item_temp_init_bias_all,index)  # temp bias

        # preference
        pos_preference = tf.reduce_sum(tf.multiply(user_embedding, pos_item_embedding),axis=1)
        neg_preference = tf.reduce_sum(tf.multiply(user_embedding, neg_item_embedding),axis=1)
        
        # temproal bias
        user_bias = tf.add(user_temp_bias,tf.constant(1.0))
        pos_item_bias = tf.add(pos_item_init_bias, pos_item_temp_bias )
        neg_item_bias = tf.add(neg_item_init_bias,neg_item_temp_bias)
        pos_bias =  tf.multiply(user_bias, pos_item_bias)
        neg_bias =  tf.multiply(user_bias, neg_item_bias)

        # scores and bpr loss
        pos_scores = tf.add(pos_bias,pos_preference)
        neg_scores = tf.add(neg_bias,neg_preference)
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores) + 1e-10)
        self.mf_loss = tf.negative(tf.reduce_mean(maxi))

        regularizer = tf.nn.l2_loss(user_embedding) + tf.nn.l2_loss(pos_item_embedding) + tf.nn.l2_loss(neg_item_embedding)
        # regularizer_bias = tf.nn.l2_loss(user_temp_bias)+ tf.nn.l2_loss()
        regularizer = regularizer / self.batch_size
        self.reg_loss = self.decay * regularizer
        self.loss = self.mf_loss + self.reg_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # inference  -- prference
        self.batch_preference = tf.matmul(user_embedding, pos_item_embedding, transpose_a=False, transpose_b=True)
        # inference  -- user bias
        batch_user_bias = tf.add(user_temp_bias,tf.constant(1.0))
        batch_user_bias = tf.reshape(batch_user_bias,(-1,1))
        # inference -- item bias
        item_temp_init_bias_all = pos_item_temp_init_bias_all
        item_temp_bias = tf.gather(item_temp_init_bias_all, self.temp_num-1, axis=-1) # most recent temp bias
        item_init_bias = tf.gather(item_temp_init_bias_all, self.temp_num, axis=-1) # init bias
        tot_item_bias = tf.add(item_temp_bias, item_init_bias)
        tot_item_bias = tf.reshape(tot_item_bias,(1,-1))
        # inference -- tot bias
        self.batch_tot_bias = tf.matmul(batch_user_bias,tot_item_bias,transpose_a=False,transpose_b=False)
        # inference -- scores
        self.batch_ratings = tf.add(self.batch_preference, self.batch_tot_bias)

        self._statistics_params()

    def init_weights(self):
        weights = dict()
        self.initializer = tf.contrib.layers.xavier_initializer()
        initializer = self.initializer
        with tf.variable_scope('parameter'):
            weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            weights['user_temp_bias'] = tf.Variable(initializer([self.n_users, 1]), name='user_temp_bias')
            #weights['item_init_bias'] = tf.Variable(initializer([self.n_items, 1]),name='item_init_bias')
            weights['item_temp_init_bias'] = tf.Variable(initializer([self.n_items, self.temp_num+1]),name='item_temp_bias') # temp_bias + one init bias
        return weights

    # def creat_bpr_with_temp_bias(self,users,pos_item,neg_items,user_temp,):
    #     pas

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)


class BPRMF:
    '''
    BPRMF
    '''
    def __init__(self, args, data_config,use_dataset_api=False,users_api=None,pos_items_api=None,neg_items_api=None):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.decay = args.regs
        self.emb_dim = args.embed_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.c = args.c
        self.alpha = args.alpha
        self.beta = args.beta
        #placeholders
        if use_dataset_api:  # using dataset api
            self.users = users_api
            self.pos_items = pos_items_api
            self.neg_items = neg_items_api 
        else:                # feed_dict 
            self.users = tf.placeholder(tf.int32, shape = (None,))
            self.pos_items = tf.placeholder(tf.int32, shape = (None,))
            self.neg_items = tf.placeholder(tf.int32, shape = (None,))

        #initiative weights
        self.weights = self.init_weights()

        #neting
        user_embedding = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
        user_rand_embedding = tf.nn.embedding_lookup(self.weights['user_rand_embedding'], self.users)
        item_rand_embedding = tf.nn.embedding_lookup(self.weights['item_rand_embedding'], self.pos_items)
        

        self.const_embedding = self.weights['c']
        self.user_c = tf.nn.embedding_lookup(self.weights['user_c'], self.users)

        self.batch_ratings = tf.matmul(user_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)    #prediction, shape(user_embedding) != shape(pos_item_embedding)
        self.user_const_ratings = self.batch_ratings - tf.matmul(self.const_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)   #auto tile
        self.item_const_ratings = self.batch_ratings - tf.matmul(user_embedding, self.const_embedding, transpose_a=False, transpose_b = True)       #auto tile
        self.user_rand_ratings = self.batch_ratings - tf.matmul(user_rand_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)
        self.item_rand_ratings = self.batch_ratings - tf.matmul(user_embedding, item_rand_embedding, transpose_a=False, transpose_b = True)


        self.mf_loss, self.reg_loss = self.create_bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss = self.mf_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        trainable_v1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'parameter')
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list = trainable_v1)
        # two branch
        self.w = tf.Variable(self.initializer([self.emb_dim,1]), name = 'item_branch')
        self.w_user = tf.Variable(self.initializer([self.emb_dim,1]), name = 'user_branch')
        self.sigmoid_yu = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['user_embedding'], self.w_user)))
        self.sigmoid_yi = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['item_embedding'], self.w)))
        # two branch bpr
        self.mf_loss_two, self.reg_loss_two = self.create_bpr_loss_two_brach(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_two = self.mf_loss_two + self.reg_loss_two
        self.opt_two = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two)
        # two branch bce
        self.mf_loss_two_bce, self.reg_loss_two_bce = self.create_bce_loss_two_brach(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_two_bce = self.mf_loss_two_bce + self.reg_loss_two_bce
        self.opt_two_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce)
        # two branch bce user&item
        self.mf_loss_two_bce_both, self.reg_loss_two_bce_both = self.create_bce_loss_two_brach_both(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_two_bce_both = self.mf_loss_two_bce_both + self.reg_loss_two_bce_both
        self.opt_two_bce_both = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce_both)
        # 2-stage training
        self.mf_loss2, self.reg_loss2 = self.create_bpr_loss2(user_embedding, self.const_embedding, pos_item_embedding, neg_item_embedding)
        self.loss2 = self.mf_loss2 + self.reg_loss2
        trainable_v2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'const_embedding')
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2, var_list = trainable_v2)


        self.mf_loss2_bce, self.reg_loss2_bce = self.create_bce_loss2(user_embedding, self.const_embedding, pos_item_embedding, neg_item_embedding)
        self.loss2_bce = self.mf_loss2_bce + self.reg_loss2_bce
        self.opt2_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2_bce, var_list = trainable_v2)
        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        
        
        self.opt3 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2, var_list = [self.weights['user_embedding'],self.weights['item_embedding']])
        self.opt3_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2_bce, var_list = [self.weights['user_embedding'],self.weights['item_embedding']])

        self._statistics_params()

        self.mf_loss_bce, self.reg_loss_bce = self.create_bce_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_bce = self.mf_loss_bce + self.reg_loss_bce
        self.opt_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_bce)


        # user wise two branch mf
        self.mf_loss_userc_bce, self.reg_loss_userc_bce = self.create_bce_loss_userc(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_userc_bce = self.mf_loss_userc_bce + self.reg_loss_userc_bce
        self.opt_userc_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_userc_bce, var_list = [self.weights['user_c']])
        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)




    def init_weights(self):
        weights = dict()
        self.initializer = tf.contrib.layers.xavier_initializer()
        initializer = self.initializer
        with tf.variable_scope('parameter'):
            weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_embedding')
            weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_embedding')
            weights['user_rand_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_rand_embedding', trainable = False)
            weights['item_rand_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_rand_embedding', trainable = False)
        with tf.variable_scope('const_embedding'):
            self.rubi_c = tf.Variable(tf.zeros([1]), name = 'rubi_c')
            weights['c'] = tf.Variable(tf.zeros([1, self.emb_dim]), name = 'c')
        
        weights['user_c'] = tf.Variable(tf.zeros([self.n_users, 1]), name = 'user_c_v')

        return weights

    def create_bpr_loss_two_brach(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item stop

        pos_scores = tf.nn.elu(pos_scores) + 1
        neg_scores = tf.nn.elu(neg_scores) + 1




        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items

        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        ps_sigmoid = tf.nn.sigmoid(self.pos_item_scores)
        ns_sigmoid = tf.nn.sigmoid(self.neg_item_scores)
        # first branch
        pos_scores = pos_scores* ps_sigmoid
        neg_scores = neg_scores* ns_sigmoid

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores)+1e-10)
        self.rubi_ratings = (tf.nn.elu(self.batch_ratings) + 1  - self.rubi_c) * tf.squeeze(ps_sigmoid)
        
        # self.shape1 = tf.shape(self.batch_ratings)
        # self.shape2 = tf.shape(tf.squeeze(ps_sigmoid))
        # self.rubi_ratings = (self.batch_ratings-self.rubi_c) * tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        self.direct_minus_ratings = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        

        self.mf_loss_ori_bce = tf.negative(tf.reduce_mean(maxi))
        # second branch
        # maxi_item = tf.log(tf.nn.sigmoid(self.pos_item_scores - self.neg_item_scores))
        # self.mf_loss_item_bce = tf.negative(tf.reduce_mean(maxi_item))
        self.mf_loss_item_bce = tf.reduce_mean(tf.negative(tf.log(ps_sigmoid + 1e-10))+tf.negative(tf.log(1-ns_sigmoid+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori_bce + self.alpha*self.mf_loss_item_bce
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss_two_brach(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        # self.rubi_ratings = (self.batch_ratings-self.rubi_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # self.direct_minus_ratings = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # first branch
        # fusion
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)
        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori + self.alpha*self.mf_loss_item
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss_two_brach_both(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        users_stop = users
        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        self.user_scores = tf.matmul(users_stop, self.w_user)
        # self.rubi_ratings_both = (self.batch_ratings-self.rubi_c)*(tf.transpose(tf.nn.sigmoid(self.pos_item_scores))+tf.nn.sigmoid(self.user_scores))
        # self.direct_minus_ratings_both = self.batch_ratings-self.rubi_c*(tf.transpose(tf.nn.sigmoid(self.pos_item_scores))+tf.nn.sigmoid(self.user_scores))
        self.rubi_ratings_both_nonc = self.batch_ratings * tf.transpose(tf.nn.sigmoid(self.pos_item_scores))*tf.nn.sigmoid(self.user_scores)  # add by zyang
        self.rubi_ratings_both = (self.batch_ratings-self.rubi_c)*tf.transpose(tf.nn.sigmoid(self.pos_item_scores))*tf.nn.sigmoid(self.user_scores)
        self.rubi_ratings_both_poptest = self.batch_ratings*tf.nn.sigmoid(self.user_scores)
        self.direct_minus_ratings_both = self.batch_ratings-self.rubi_c*tf.transpose(tf.nn.sigmoid(self.pos_item_scores))*tf.nn.sigmoid(self.user_scores)
        # first branch
        # fusion
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)*tf.nn.sigmoid(self.user_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)*tf.nn.sigmoid(self.user_scores)

        # pos_scores = pos_scores*(tf.nn.sigmoid(self.pos_item_scores)+tf.nn.sigmoid(self.user_scores))
        # neg_scores = neg_scores*(tf.nn.sigmoid(self.neg_item_scores)+tf.nn.sigmoid(self.user_scores))


        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # third branch
        self.mf_loss_user = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.user_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.user_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori + self.alpha*self.mf_loss_item + self.beta*self.mf_loss_user
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss
    
    
    
    def create_bce_loss_userc(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        self.pos_item_scores = tf.matmul(pos_items_stop,self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop,self.w)
        self.rubi_ratings_userc = (self.batch_ratings-self.user_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        self.direct_minus_ratings_userc = self.batch_ratings-self.user_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # first branch
        # fusion
        pos_scores = (pos_scores-self.user_c)*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = (pos_scores-self.user_c)*tf.nn.sigmoid(self.neg_item_scores)
        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori #+ self.alpha*self.mf_loss_item
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss
        
        # self.rubi_ratings_userc = (self.batch_ratings-self.user_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # self.direct_minus_ratings_userc = self.batch_ratings-self.user_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # pos_scores = (tf.reduce_sum(tf.multiply(users, pos_items), axis=1)-self.user_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # neg_scores = (tf.reduce_sum(tf.multiply(users, neg_items), axis=1)-self.user_c)*tf.squeeze(tf.nn.sigmoid(self.neg_item_scores))
        # # first branch
        # # fusion
        # mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
        # # regular
        # regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        # regularizer = regularizer/self.batch_size
        # reg_loss = self.decay * regularizer
        # return mf_loss, reg_loss

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores)+1e-10)

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bce_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # first branch
        # fusion
        mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def create_bpr_loss2(self, users, const_embedding, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) - tf.matmul(const_embedding, pos_items, transpose_a=False, transpose_b = True)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) - tf.matmul(const_embedding, neg_items, transpose_a=False, transpose_b = True)

        regularizer = tf.nn.l2_loss(const_embedding)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss
    
    def create_bce_loss2(self, users, const_embedding, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1) - tf.matmul(const_embedding, pos_items, transpose_a=False, transpose_b = True)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1) - tf.matmul(const_embedding, neg_items, transpose_a=False, transpose_b = True)

        regularizer = tf.nn.l2_loss(const_embedding)
        regularizer = regularizer/self.batch_size

        mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def update_c(self, sess, c):
        sess.run(tf.assign(self.rubi_c, c*tf.ones([1])))

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)




class ConditionalGBPRMF:
    '''
    not used
    '''
    def __init__(self, args, data_config):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_pop = data_config['n_pop']

        self.decay = args.regs
        self.emb_dim = args.embed_size
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.verbose = args.verbose
        self.c = args.c
        self.alpha = args.alpha
        self.beta = args.beta
        
        #placeholders
        self.users = tf.placeholder(tf.int32, shape = (None,))
        self.pos_items = tf.placeholder(tf.int32, shape = (None,))
        self.neg_items = tf.placeholder(tf.int32, shape = (None,))

        self.pos_pop = tf.placeholder(tf.int32,shape = (None,))
        self.neg_pop = tf.placeholder(tf.int32,shape = (None,))
        self.raw = tf.placeholder(tf.int32,shape=(None,))

        #initiative weights
        self.weights = self.init_weights()

        #neting
        user_embedding = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_item_embedding = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
        
        

        self.const_embedding = self.weights['c']
        self.user_c = tf.nn.embedding_lookup(self.weights['user_c'], self.users)

        self.batch_ratings = tf.matmul(user_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)    #prediction, shape(user_embedding) != shape(pos_item_embedding)
        # self.user_const_ratings = self.batch_ratings - tf.matmul(self.const_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)   #auto tile
        # self.item_const_ratings = self.batch_ratings - tf.matmul(user_embedding, self.const_embedding, transpose_a=False, transpose_b = True)       #auto tile
        # self.user_rand_ratings = self.batch_ratings - tf.matmul(user_rand_embedding, pos_item_embedding, transpose_a=False, transpose_b = True)
        # self.item_rand_ratings = self.batch_ratings - tf.matmul(user_embedding, item_rand_embedding, transpose_a=False, transpose_b = True)


        # self.mf_loss, self.reg_loss = self.create_bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding)
        # self.loss = self.mf_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(self.loss)
        # trainable_v1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'parameter')
        # self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, var_list = trainable_v1)
        # two branch
        self.w = tf.Variable(self.initializer([self.emb_dim,1]), name = 'item_branch')
        self.w_user = tf.Variable(self.initializer([self.emb_dim,1]), name = 'user_branch')
        self.sigmoid_yu = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['user_embedding'], self.w_user)))
        self.sigmoid_yi = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['item_embedding'], self.w)))
        # two branch bpr
        self.mf_loss_pop_global, self.reg_loss_pop_global = self.create_bpr_loss_with_pop_global(user_embedding, pos_item_embedding, neg_item_embedding)
        self.loss_pop_global = self.mf_loss_pop_global + self.reg_loss_pop_global
        self.opt_pop_global = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_pop_global)
        self._statistics_params()

    def init_weights(self):
        weights = dict()
        self.initializer = tf.contrib.layers.xavier_initializer()
        initializer = self.initializer
        with tf.variable_scope('parameter'):
            weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name = 'user_embedding')
            weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name = 'item_embedding')
            weights['pop_embedding'] = tf.Variable(initializer([self.n_pop,self.emb_dim]),name = 'pop_embedding' )
        with tf.variable_scope('const_embedding'):
            self.rubi_c = tf.Variable(tf.zeros([1]), name = 'rubi_c')
            weights['c'] = tf.Variable(tf.zeros([1, self.emb_dim]), name = 'c')
        weights['user_c'] = tf.Variable(tf.zeros([self.n_users, 1]), name = 'user_c_v')
        return weights
    

    def create_bpr_loss_with_pop_global(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)   

        shape1  = tf.shape(pos_scores)

        u_pop_scores = tf.matmul(users,self.weights['pop_embedding'],transpose_a=False, transpose_b = True)
        u_pop_scores = tf.nn.softmax(u_pop_scores,axis=-1)

        shape2 = tf.shape(u_pop_scores)

        N = self.pos_pop.shape.as_list()[0]
        # print(N)
        # row = tf.constant([[i] for i in range(N)],dtype=tf.int32)
        
        pos_pop_index = tf.concat([tf.reshape(self.raw,[-1,1]), tf.reshape(self.pos_pop,[-1,1])], axis=-1)
        neg_pop_index = tf.concat([tf.reshape(self.raw,[-1,1]), tf.reshape(self.neg_pop,[-1,1])], axis=-1)


        shape3 = tf.shape(pos_pop_index)

        self.shapes1 = [shape1,shape2,shape3]
        pos_pop = tf.gather_nd(u_pop_scores, pos_pop_index)
        neg_pop = tf.gather_nd(u_pop_scores, neg_pop_index)
        pos_pop_inference = tf.gather(u_pop_scores,self.pos_pop,axis=1)

        shape4 = tf.shape(pos_pop)
        self.shapes = [shape1,shape2,shape3,shape4]
        # item stop

        pos_scores = tf.nn.elu(pos_scores) + 1
        neg_scores = tf.nn.elu(neg_scores) + 1
        pos_scores_with_pop = tf.multiply(pos_scores,pos_pop)
        neg_scores_with_pop = tf.multiply(neg_scores,neg_pop)

        maxi = tf.log(tf.nn.sigmoid(pos_scores_with_pop - neg_scores_with_pop)+1e-10)

        UI_interaction = tf.nn.elu(self.batch_ratings) + 1    # batch_size * N_item
        self.condition_ratings = tf.multiply(UI_interaction,pos_pop_inference) # * tf.squeeze(pos_pop)  # only one pop

        all_pop_scores = tf.expand_dims(u_pop_scores,-1)      # batch_size * N_pop *1 
        self.intervention_rating =  tf.reduce_sum(tf.multiply(all_pop_scores, tf.expand_dims(UI_interaction,1)), axis=1)       #  E_(pop|u) (Pre)

        self.mf_loss_ori = tf.negative(tf.reduce_mean(maxi))
        mf_loss = self.mf_loss_ori
        # regular
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    # def create_bpr_loss(self, users, pos_items, neg_items):
    #     pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
    #     neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

    #     regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
    #     regularizer = regularizer/self.batch_size

    #     maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores)+1e-10)

    #     mf_loss = tf.negative(tf.reduce_mean(maxi))
    #     reg_loss = self.decay * regularizer
    #     return mf_loss, reg_loss

    # def create_bce_loss(self, users, pos_items, neg_items):
    #     pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
    #     neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
    #     # first branch
    #     # fusion
    #     mf_loss = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-9))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-9)))
    #     # regular
    #     regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
    #     regularizer = regularizer/self.batch_size
    #     reg_loss = self.decay * regularizer
    #     return mf_loss, reg_loss

    def update_c(self, sess, c):
        sess.run(tf.assign(self.rubi_c, c*tf.ones([1])))

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)
