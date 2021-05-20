import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run pop_bias.")
    parser.add_argument('--data_path', nargs='?', default='./data/',  # change by zyang
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='kwai',
                        help='Choose a dataset from {movielens_ml_1m, movielens_ml_10m, gowalla}')
    parser.add_argument('--source', nargs='?', default='normal',
    help='...') # not used
    parser.add_argument('--train', nargs='?', default='normal',
    help='normal(MF) | s_condition (PD/PDA)| temp (BPRMF(t)-pop)')
    parser.add_argument('--test', nargs='?', default='normal',
    help='normal(MF) | s_condition (PD/PDA)| temp (BPRMF(t)-pop)')
    parser.add_argument('--valid_set', nargs='?', default='test',
    help='test | valid')
    parser.add_argument('--save_dir',nargs='?',default="/data/zyang/save_model/",
    help='save path')

    parser.add_argument('--alpha', type=float, default=1e-3,  # not used
                        help='alpha')
    parser.add_argument('--beta', type=float, default=1e-3, # not used
                        help='beta')

    parser.add_argument('--pc_alpha', type=float, default=0.1, # not used
                        help='alpha')
    parser.add_argument('--pc_beta', type=float, default=0.1, # not used
                        help='beta')

    parser.add_argument('--exp_init_values',type=float,default=0.1,help='power coff initial value')
    parser.add_argument('--pop_exp', type=float, default=0.1,
                        help='popularity power coff')          # gamma in paper
    parser.add_argument('--early_stop', type=int, default=1,
                        help='alpha')
    parser.add_argument('--need_save', type=int, default=1,
                        help='0: do not save model, 1:saving')
    parser.add_argument('--cores', type=int, default=1,
                        help='cores for  prefetch')

    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')
    parser.add_argument('--load_epoch', type=int, default=400,
                        help='Epoch which to load, for pretraining.')  # not used
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--Ks', nargs='?', default='[20]',
                        help='Evaluate on Ks optimal items.')
    parser.add_argument('--epochs', nargs='?', default='[]',
                        help='Test c on these epochs.')
    parser.add_argument('--regs', type=float, default=1e-5,
                        help='Regularizations.')
    parser.add_argument('--fregs', type=float, default=1e-5,
                        help='fine-tune Regularizations.')    # not used
    parser.add_argument('--c', type=float, default=10.0,
                        help='Constant c.')               # not used
    parser.add_argument('--train_c', type=str, default="val",
                        help='val | test')         # not used
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='Weight decay of optimizer.')  # not used
    parser.add_argument('--model', nargs='?', default='mf',
                        help='Specify model type, choose from {mf, CausalE}')
    parser.add_argument('--skew', type=int, default=0,
                        help='Use not skewed dataset.')  # not used
    parser.add_argument('--model_type', nargs='?', default='o',
                        help='Specify model type, choose from {o, c, ic, rc, irc}')  # not used
    parser.add_argument('--devide_ratio', type=float, default=0.8,
                        help='Train/Test.')            # not used
    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')
    
    parser.add_argument('--pop_used', type=int, default=-2,
                        help='pop_rate used in test')         # not used

    parser.add_argument('--cuda', type=str, default='1',
                        help='Avaiable GPU ID')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: no pretrain, 1: load pretrain model')  # not used
    parser.add_argument('--check_c', type=int, default=1,
                        help='0: no checking, 1: check a range of cs')  # not used
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log\'s interval epoch while training')
    parser.add_argument('--pop_wd', type=float, default=0.,
                        help='weight decay of popularity')   # not used
    parser.add_argument('--base', type=float, default=-1.,
                        help='check range base.')         # not used
    parser.add_argument('--cf_pen', type=float, default=1.0,
                        help='Imbalance loss.')          # not used
    parser.add_argument('--saveID', nargs='?', default='',
                        help='Specify model save path.')
    parser.add_argument('--user_min', type=int, default=1,
                        help='user_min.')             # not used
    parser.add_argument('--user_max', type=int, default=1000,
                        help='user max per cls.')       # not used
    parser.add_argument('--data_type', nargs='?', default='ori',
                        help='load imbalanced data or not.')
    parser.add_argument('--imb_type', nargs='?', default='exp',
                        help='imbalance type.')        # not used
    parser.add_argument('--top_ratio', type=float, default=0.1,
                        help='imbalance top ratio.') # not used
    parser.add_argument('--lam', type=float, default=1.,
                        help='lambda.')    # not used
    parser.add_argument('--check_epoch', nargs='?', default='all',
                        help='check all epochs or select some or search in range.')   # not used
    parser.add_argument('--start', type=float, default=-1.,
                        help='check c start.')  # not used
    parser.add_argument('--end', type=float, default=1.,
                        help='check c end.')  # not used
    parser.add_argument('--step', type=int, default=20,
                        help='check c step.')      # not used
    parser.add_argument('--out', type=int, default=0)   # not used
    return parser.parse_args()
