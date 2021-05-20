from parse import parse_args
from load_data import Data,Data2
import multiprocessing
import heapq

args = parse_args()

if args.train == 's_condition' or args.train == 'sg_condition' or args.train == 'temp_pop' or args.train == 'us_condition':
    #PD/PDA/PDG/BPRMF(t)-pop
    data = Data2(args)
else: #BPRMF
    data = Data(args)


#sorted_id, belong, rate, usersorted_id, userbelong, userrate = data.plot_pics()
Ks = eval(args.Ks)
BATCH_SIZE = args.batch_size
ITEM_NUM = data.n_items
USER_NUM = data.n_users

points = [10, 50, 100, 200, 500]