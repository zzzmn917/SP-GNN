import pickle

import argparse
# 4.23 10：10
# 改：无向图有邻居数量的要求，但不是3个以内都是，只要2个以内的（和论文8一样）  

# 实现了无向图的所有内容，得到了所需要的项目的邻居权重向量，还有邻居列表
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Tmall/Nowplaying')
parser.add_argument('--sample_num', type=int, default=12)
opt = parser.parse_args()
 
dataset = opt.dataset
sample_num = opt.sample_num

seq = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))

if dataset == 'diginetica':
    num = 43098
elif dataset == "Tmall":
    num = 40728
elif dataset == "Nowplaying":
    num = 60417
else:
    num = 3

relation = []
neighbor = [] * num

all_test = set()  

adj1 = [dict() for _ in range(num)]  
adj = [[] for _ in range(num)]   


for i in range(len(seq)):   
    data = seq[i]     
    for k in range(1, 3):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])


for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1
# 生成权重列表
weight = [[] for _ in range(num)]
# 对于每个节点 t，将其邻接关系按权重从大到小排序，并分别保存邻接节点和对应的权重
for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]
# 截取前 sample_num 个邻接节点和对应的权重。sample_num用于控制每个节点在邻接列表中保留的邻居数量，论文中设置的最大邻居数为12
for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('datasets/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))

