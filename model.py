import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator, TargetAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F

# 改完！第一版本 4月23日 16:04   （明天colab运行一下，gpu，记得model中的129注释掉，128取消注释）

# 差的第二步为： S2：在这边基于DHCN代码加入基于生成的全局嵌入和局部嵌入做一个损失函数conloss；另外会话嵌入部分要将公式12的内容换为自己创新点中的St时间嵌入表示或者兴趣嵌入表示。
# 最终这边要有两个损失函数，借鉴DHCN代码，看看怎么修改。做个对比学习。


# 生成最终的项目嵌入、会话嵌入还有预测分数。


class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt
        self.beta = opt.beta
        self.batch_size = opt.batch_size
        self.num_node = num_node  # 节点数量
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local    # 局部dropout比例
        self.dropout_global = opt.dropout_global   # 全局dropout比例
        self.dropout_target = opt.dropout_target  # 兴趣dropout比例
        self.hop = opt.n_iter   # 图卷积层数
        self.sample_num = opt.n_sample    # 采样节点数
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()   # 邻居矩阵
        self.num = trans_to_cuda(torch.Tensor(num)).float()   # 邻居节点的数量
        
        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0, step=opt.step)
        self.target_agg = TargetAggregator(self.dim, self.opt.alpha, opt.dropout_target)
        self.global_agg = []
        # 判断激活函数是否为 ReLU，如果是，则创建一个使用 ReLU 激活函数的全局聚合器实例；否则，创建一个使用 tanh 激活函数的实例
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # 项目嵌入 & 位置嵌入
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))  # 论文公式11的W3
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))    # 论文公式13的q
        self.glu1 = nn.Linear(self.dim, self.dim)        # 论文公式13的W4
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)   #论文公式13的W5
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()   # 定义了交叉熵损失函数
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)  # 定义了优化器，用于训练模型
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)   # 定义了学习率调度器，用于训练模型

        self.reset_parameters()   # 初始化模型的参数

    # 对模型的参数进行随机初始化
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    # 返回目标节点的邻接节点信息和邻接节点数量信息
    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]
    
    # hidden即为公式10所得的项目的最终嵌入表示——隐藏状态
    def compute_scores(self, hidden, targets,inputs, mask):
        mask = mask.float().unsqueeze(-1)
        
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        inputs = self.embedding(inputs)
        targets =self.embedding(targets)
        pos_emb = self.pos_embedding.weight[:len]    # 从位置嵌入（pos_embedding）中获取与序列长度相匹配的嵌入
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)   #  将位置嵌入扩展为与隐藏状态张量相同的形状

        # 替换原来的加权平均值为兴趣嵌入表示的计算
        hs = self.target_agg(inputs, targets, mask)
        # 池化操作，将 hs 的尺寸调整为 (batch_size, dim)
        hs= torch.mean(hs, dim=1)  # 可以根据具体需求选择其他池化操作
        
        # hs = hs.unsqueeze(-2).repeat(1, len, 1)   # 形状和隐藏状态向量一致
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)    # 论文公式11的结果
        hs_expanded = hs.unsqueeze(1).expand(-1, nh.size(1), -1)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs_expanded))  # 论文公式13除去权重矩阵q的结果
        beta = torch.matmul(nh, self.w_2)    # 论文公式13的结果，得到注意力权重
        beta = beta * mask   # 将注意力权重与掩码相乘，以过滤掉不相关的部分
        select = torch.sum(beta * hidden, 1)  # 论文公式14的结果

        b = self.embedding.weight[1:]  # n_nodes x latent_size  获取初始嵌入权重，去除第一个嵌入，因为通常第一个嵌入被用作填充标记
        scores = torch.matmul(select, b.transpose(1, 0))    # 公式15  b.transpose就是初始项目嵌入，由初始项目权重b转置得来
        return scores

    def SSL(self, h_local, h_global):
        # 对输入的嵌入表示进行随机打乱
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        # 对行和列都随机打乱
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        # 计算两张量的相似度分数
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)
        # 计算了正样本的相似度分数；sess_emb_hgnn是超图最后生成的会话嵌入，另一个是线图生成的会话嵌入
        pos = score(h_local, h_global)
        # 计算了负样本的相似度分数
        neg1 = score(h_global, row_column_shuffle(h_local))
        one = torch.cuda.FloatTensor(neg1.shape).fill_(1)
        # one = torch.FloatTensor(neg1.shape).fill_(1)
        # 自监督学习的损失函数，公式9
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss



    def forward(self, inputs, adj, mask_item, item, targets):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)  # 获得初始项目嵌入h
        targets1 =self.embedding(targets) # 获取每个样本的最后一个项目的嵌入表示
        # local、targets
        h_local = self.local_agg(h, adj, mask_item)
        h_target = self.target_agg(h, targets1, mask_item)
        # global
        item_neighbors = [inputs]  # 初始化项目邻居
        weight_neighbors = []   # 初始化邻居列表
        support_size = seqs_len
        
        # 对于每个跳数进行循环迭代，进行采样操作，并更新项目邻居和权重邻居列表
        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]   # 获取邻居实体向量
        weight_vectors = weight_neighbors   # 获取邻居权重向量

        session_info = []   # 存储每一轮迭代中的会话信息
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)  # 可以将不在会话中的项目的嵌入表示置零
        
        # mean 
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)  # 计算项嵌入的平均值
        
        # sum
        # sum_item_emb = torch.sum(item_emb, 1)
        
        sum_item_emb = sum_item_emb.unsqueeze(-2)   # 调整形状

        # 这是一个循环，它遍历了每一层的迭代次数。self.hop 表示模型中使用的全局聚合器的层数
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))
        
        # 这一个大的for循环是将这几个参数传给aggregator.py的全局聚合器的forward函数中。也是公式6的结果
        # 这是一个外部循环，它遍历了每一层的迭代次数
        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            # 这是一个内部循环，它遍历了每一轮迭代的次数
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]   # self.global_agg 列表中选择第 n_hop 层的全局聚合器
                vector = aggregator(self_vectors=entity_vectors[hop],  # 当前层的实体向量
                                    neighbor_vector=entity_vectors[hop+1].view(shape),  # 下一层的邻居向量 
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),  # 邻居权重
                                    extra_vector=session_info[hop])  # 当前会话的信息 
                entity_vectors_next_iter.append(vector)  # 将计算得到的下一层实体向量 vector 添加到 entity_vectors_next_iter 列表中
            entity_vectors = entity_vectors_next_iter    # 更新当前层的实体向量为下一轮迭代计算得到的值

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)  # 将得到的全局实体向量中的第一层作为全局表示 h_global

        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)   # aggregator.py文件中局部聚合器的输入，即有向项目嵌入，论文公式9的结果
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)  # aggregator.py文件中全局聚合器的输入，即全局项目嵌入，论文公式5的结果
        h_target = F.dropout(h_target, self.dropout_target, training=self.training)
        output = h_local + h_global   # 论文公式10，最终的项目嵌入，也就是本文件中hidden的表示内容
        output = torch.cat((h_target, output), dim=1)
        con_loss = self.SSL(h_local, h_global)
        return output, self.beta*con_loss


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    targets = trans_to_cuda(targets).long()
    

    hidden , con_loss = model(items, adj, mask, inputs, targets)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    # hs, con_loss = model(inputs, adj, mask, items)
    return targets, model.compute_scores(seq_hidden, targets, inputs, mask), con_loss


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()  # 将模型设为训练模式
    total_loss = 0.0
    # num_workers 参数指定用于数据加载的线程数，batch_size 参数指定批量大小，shuffle 参数表示是否在每个 epoch 开始时打乱数据，pin_memory 参数表示是否将数据存储在 CUDA 固定内存中
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    # 对训练数据加载器进行迭代，每次迭代加载一个批量的数据
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores, con_loss = forward(model,  data)  # 对数据进行前向传播，获取目标和预测分数
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss = loss + con_loss
        loss.backward()  # 反向传播，计算梯度
        model.optimizer.step()
        total_loss += loss  # 累加损失值
    print('\ttotal_Loss:\t%.3f' % total_loss)   # 打印当前累计损失
    model.scheduler.step()   # 调度器进行一步更新，可能调整学习率
    
    

    print('start predicting: ', datetime.datetime.now())   # 打印当前时间，表示开始测试
    model.eval()   # 将模型设为评估模式
    # 创建测试数据加载器，用于迭代测试数据集
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []    # 初始化结果列表
    hit, mrr = [], []
    for data in test_loader:
        targets, scores, con_loss = forward(model, data) 
        sub_scores = scores.topk(20)[1]  # 从预测分数中取出排名前20的项目
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()   # 将排名结果转移到 CPU 设备，并转换为 NumPy 数组
        # targets = targets.numpy()   # 将目标转换为 NumPy 数组（自己改）
        targets = targets.cpu().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
               mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)

    return result
