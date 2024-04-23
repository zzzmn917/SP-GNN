import numpy as np
import torch
from torch.utils.data import Dataset
# 4.23 10：10
# 改：但是这部分和我的创新点内容不一致，代码最后得到四个关系矩阵，而我要得到出入度归一化矩阵A——已改完！可以测试一下  

# 这里代码是关于会话有向图的实现内容，他最终得到了项目与项目之间的关系向量矩阵。


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set  
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)  
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]  
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]  
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]  
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]  

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y) 

# 补全session的长度。nowData指的是某一个会话序列，inputData指的是全部会话序列。train_len指定会话的长度
def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]  
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len  
    
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len


# 这段保证了每个结点的邻居节点列表都是统一的长度；不同节点的邻居权重列表数量也是统一的。
# adj_dict：表示图中每个实体的邻居、 n_entity：图中实体的总数、 sample_num：每个实体的最大邻居数12、 num_dict：图中每个实体邻居的权重
def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)  # 矩阵用来存储每个实体的邻居实体，行数为 n_entity，列数为 sample_num
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)  # 矩阵用来存储每个实体的邻居实体权重，行数为 n_entity，列数为 sample_num
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)  # 无放回抽样
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)   # 有放回抽样
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity  # 返回更新后的邻居实体矩阵 adj_entity 和对应的权重矩阵 num_entity


class Data(Dataset):
    def __init__(self, data, train_len=None):
        inputs, mask, max_len = handle_data(data[0], train_len)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len


    def __getitem__(self, index):
        u_input, mask, targets = self.inputs[index], self.mask[index], self.targets[index] 
        max_n_node = self.max_len
        node = np.unique(u_input)
        items = node.tolist() + (max_n_node - len(node)) * [0]  # 保证所有的会话序列最后生成的项目列表长度是一致的，少的用0补齐
        adj1 = np.zeros((max_n_node, max_n_node))  
        for i in np.arange(len(u_input) - 1):
            # 邻接矩阵的对角线元素设置为 1
            u = np.where(node == u_input[i])[0][0]
            adj1[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            adj1[v][v] = 1
            adj1[u][v] = 1
        adj_sum_in = np.sum(adj1, 0)     
        adj_sum_in[np.where(adj_sum_in == 0)] = 1  
        adj_in = np.divide(adj1, adj_sum_in) 
        adj_sum_out = np.sum(adj1, 1)   
        adj_sum_out[np.where(adj_sum_out == 0)] = 1  
        adj_out = np.divide(adj1.transpose(), adj_sum_out)  
        adj1 = np.concatenate([adj_in, adj_out]).transpose()
        # self.adj.append(adj1)  # adj：归一化后的出入度邻接矩阵  
        
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]

        return [torch.tensor(alias_inputs), torch.tensor(adj1), torch.tensor(items),
                torch.tensor(mask), torch.tensor(targets), torch.tensor(u_input)]

    def __len__(self):
        return self.length