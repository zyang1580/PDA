import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Run pop_bias.")
parser.add_argument('--path', nargs='?', default="data/ml_10m/",  # change by zyang
                    help='Input data path.')
parser.add_argument('--slot_count', type=int, default=13,  # change by zyang
                    help='Input data path.')
args = parser.parse_args()

root = args.path #'./data/ml_10m/'
slot_count = args.slot_count
item_list = []
for i in range(slot_count):
    path = root+'t_{}.txt'.format(i)
    with open(path) as f:
        for line in f:
            item_list.append(int(line.split()[0]))
# print("item_list:",item_list)
n_item = len(set(item_list))
pop_item = []
for i in range(slot_count):
    path = root+'t_{}.txt'.format(i)
    total = 0
    item_pop_list_t=[]
    with open(path) as f:
        for line in f:
            line = line.strip().split()
            item, pop = int(line[0]), len(line[1:])
            item_pop_list_t.append((item,pop))
            total+=pop
    pop_item.append([1/(total+n_item) for _ in range(n_item)])
    # pop_item.append([0/(total) for _ in range(n_item)])
    for item,pop in item_pop_list_t:
        # print(item,n_item)
        pop_item[i][item] = (pop+1.0)/(total+n_item)
        # pop_item[i][item] = 1e6*(pop)/(total)
pop_item = np.array(pop_item)
# 0-1
# pop_item = (pop_item-np.min(pop_item))/(np.max(pop_item)-np.min(pop_item))

for k in range(pop_item.shape[0]):
    pop_item[k] = (pop_item[k] - pop_item[k].min()) / (pop_item[k].max() - pop_item[k].min())

print("tot information:\nmean:",pop_item.mean(axis=1))
print("max:",pop_item.max(axis=1))
print("min:",pop_item.min(axis=1))

with open(root+"item_pop_seq_ori2.txt","w") as f:
    for i in range(n_item):
        pop_seq_i = pop_item[:, i]
        write_str = ""
        write_str += str(i) + ' '
        for pop in pop_seq_i:
            write_str += str(pop) + ' '
        write_str = write_str.strip(' ')
        write_str += '\n'
        f.write(write_str)
