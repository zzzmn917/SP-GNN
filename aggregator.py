import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy

# 4.23 10：10
# 还差两步   S1：在这边加入时间嵌入代码或者兴趣嵌入代码。二选一吧，不然太麻烦，或者自己想想，有什么可以将两者结合的方法。——暂定写兴趣嵌入 


# 通过全局聚合器中得到的权重来得到全局项目嵌入（这部分可以不变）；通过局部聚合器中得到局部项目嵌入（这里要变，使用GNN或者MGU看看哪个效果好！可以先试着用GNN）——已改完
# 应该要在这部分加入生成时间嵌入的代码（combine temporal）；兴趣嵌入代码（论文MGU）——已加了兴趣嵌入表示，这部分没有问题


# Aggregator 类是一个基类，它定义了一些共享的属性和方法
class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout   # 防止过拟合的概率
        self.act = act   # 激活函数
        self.batch_size = batch_size
        self.dim = dim   # 向量的维度 

    def forward(self):
        pass


# 兴趣聚合器实现的是兴趣嵌入的功能——按照原局部聚合器内容进行更改
class TargetAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(TargetAggregator, self).__init__()
        self.dim = dim  # dim 表示向量的维度
        self.dropout = dropout
        
        # 这里只要参数矩阵W
        self.w_0 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, target, mask_item=None):
        h = hidden   # 节点的初始项目嵌入
        ht = target  # 会话的标签
        batch_size = h.shape[0]  # 获取批量大小
        N = h.shape[1]    # 获取节点数量
        
        a_input = torch.matmul(h, self.w_0.t().unsqueeze(0))

        e_0 = torch.matmul(a_input, ht.unsqueeze(-1)).squeeze(-1)
        
        # e_0 = self.leakyrelu(e_0).unsqueeze(-1).view(batch_size, N, N)
        
        
        # 使用极小值对不相连的节点的注意力系数进行屏蔽
        # mask = -9e15 * torch.ones_like(e_0)
        
        # alpha = torch.where(h > 0, e_0, mask)
        alpha = torch.softmax(e_0, dim=-1)   # 对得到的 alpha 进行 softmax 归一化，使得所有权重的总和为 1
        
        # 得到最终的兴趣嵌入表示
        output = torch.matmul(alpha, h)
        return output




# 局部聚合器通过GNN实现局部项目嵌入，更改代码了——照着SR-GNN。改完了，但是没验证。
class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0.,  step=1, name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim  # dim 表示向量的维度
        self.dropout = dropout
        self.step = step
        
        self.w_ih = nn.Parameter(torch.Tensor(3 *self.dim, 2 *self.dim))
        self.w_hh = nn.Parameter(torch.Tensor(3 *self.dim, self.dim))
        self.b_ih = nn.Parameter(torch.Tensor(3 *self.dim))
        self.b_hh = nn.Parameter(torch.Tensor(3 *self.dim))
        self.b_oah = nn.Parameter(torch.Tensor(self.dim))
        self.b_iah = nn.Parameter(torch.Tensor(self.dim))

        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.linear_edge_in = nn.Linear(self.dim, self.dim, bias=True)
        self.linear_edge_out = nn.Linear(self.dim, self.dim, bias=True)
        self.linear_edge_f = nn.Linear(self.dim, self.dim, bias=True)

    def GNNCell(self, hidden, adj, mask_item=None):
        h = hidden   # 节点的隐藏状态向量（初始嵌入，由model.py中97行代码获得，并且通过100行代码进行传入的）
        # a = adj.shape[1]
        # batch_size = h.shape[0]  # 获取批量大小
        # N = h.shape[1]    # 获取节点数量
        # 首先对输入信息进行线性变换。A是邻接矩阵，h是当前时刻的隐藏状态
        input_in = torch.matmul(adj[:, :, :adj.shape[1]], self.linear_edge_in(h)) + self.b_iah
        input_out = torch.matmul(adj[:, :, adj.shape[1]: 2 * adj.shape[1]], self.linear_edge_out(h)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        # gi：当前节点的输入信息    gh：上一个节点的隐藏状态
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(h, self.w_hh, self.b_hh)
        # 使用chunk函数将gi和gh分成三个部分，分别表示重置门、更新门和候选隐状态
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        # 公式3 重置门
        resetgate = torch.sigmoid(i_r + h_r)
        # 公式2 更新门
        inputgate = torch.sigmoid(i_i + h_i)
        # 公式4
        newgate = torch.tanh(i_n + resetgate * h_n)
        # 公式5 ？？？没看懂？？？
        hy = newgate + inputgate * (h - newgate)
        return hy

    def forward(self, hidden, adj, mask_item=None):
        for i in range(self.step):
            hidden = self.GNNCell( hidden, adj, mask_item=None)
        return hidden    


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act   # 激活函数，默认为ReLU
        self.dim = dim  # 维度

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))  # 是一个 (dim + 1) × dim 的张量,用于注意力权重的线性变换，论文中公式2的W1
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))        # 是一个 1 × dim 的张量,用于注意力权重的线性变换，论文中公式2的q1
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))  # 用于信息聚合的线性变换，论文中公式5的W2
        self.bias = nn.Parameter(torch.Tensor(self.dim))   # 用于信息聚合的偏置，论文中的？（好像没提到）

    # self_vectors: 当前节点的表示；neighbor_vector: 邻居节点的表示；neighbor_weight: 邻居节点的权重；extra_vector=None: 额外的向量（可选）
    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            # 论文中的s*hvj，实际上是将 extra_vector 中的每个元素与 neighbor_vector 中对应位置的元素相乘，也就是公式3的代码放在这里面了。即公式2的大括号内的部分。
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            # 公式2去掉q1参数的内容
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)  # 公式2的结果（matmul是进行矩阵乘法操作）
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)   # 公式4
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)  # 公式1，得到最终的项目的邻居嵌入表示
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        # 沿着最后一个维度拼接起来，得到一个维度为 (batch_size, dim_self + dim_neighbor) 的张量
        output = torch.cat([self_vectors, neighbor_vector], -1)
        # 对拼接后的表示进行 dropout 操作，以减少过拟合
        output = F.dropout(output, self.dropout, training=self.training)
        # 将 dropout 后的表示与模型参数 self.w_3 进行矩阵相乘，得到输出表示。这里的w_3为论文中公式5的W2
        output = torch.matmul(output, self.w_3)
        #  这一步操作将张量的最后一个维度压缩，将其移除
        output = output.view(batch_size, -1, self.dim)
        # 应用了激活函数 self.act，在代码中默认为 ReLU 激活函数。公式5
        output = self.act(output)
        return output

