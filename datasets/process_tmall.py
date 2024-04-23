import argparse
import time
import csv
import pickle
import operator
import datetime
import os

# tmall数据集的数据处理方式和nowplaying的处理方式一致。并且两个和SR-GNN的preprocess.py逻辑都一样

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

with open('tmall_data.csv', 'w') as tmall_data:
    with open('tmall/dataset15.csv', 'r') as tmall_file:
        header = tmall_file.readline()
        tmall_data.write(header)
        for line in tmall_file:
            data = line[:-1].split('\t')
            if int(data[2]) > 120000:
                break
            tmall_data.write(line)

print("-- Starting @ %ss" % datetime.datetime.now())
with open('tmall_data.csv', "r") as f:
    reader = csv.DictReader(f, delimiter='\t')  # 使用 \t 作为分隔符
    sess_clicks = {}  # 记录每个session点击过的物品
    sess_date = {}   # 点击的时间
    ctr = 0       # 计数器
    curid = -1      # 当前会话id
    curdate = None     # 当前日期
    for data in reader:   # 迭代处理 CSV 文件中的每一行数据
        sessid = int(data['SessionId'])
        if curdate and not curid == sessid:
            date = curdate
            sess_date[curid] = date
        curid = sessid    
        item = int(data['ItemId'])   # 获取项目ID
        curdate = float(data['Time'])  # 获取时间戳

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = float(data['Time'])
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())   # 获取当前时间

# sess_clicks是全部会话序列；sess_click[s]代表的是一个单独会话；s则代表sess_clicks的副本，防止sess_clicks出现变动

# 过滤掉会话长度为1的会话
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# 统计每个物品出现的次数，按照从高到低排序，方便查看哪些物品更受欢迎
iid_counts = {}    # 记录每个物品出现的次数，遍历全部会话得到每个物品的点击次数
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))   # 将物品点击总数按序排列

# 过滤掉点击物品数小于2的物品
length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))    # 遍历每个会话，只保留在全部会话中点击次数大于等于5的物品 的会话
    if len(filseq) < 2 or len(filseq) > 40:    # 每个会话保留大于等于5的物品，并且会话序列长度必须要大于等于2或者小于等于40，否则删除
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())  # 列表 dates，其中每个元素都是一个元组 (session_id, date)
maxdate = dates[0][1]     # 初始化maxdate为列表的第一个会话的日期

# maxdate 将包含 dates 列表中的最大日期值（多用于按时间线划分训练集和测试集）
for _, date in dates:
    if maxdate < date:
        maxdate = date

# the last of 100 seconds for test，划分测试集
splitdate = maxdate - 100

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)   # 时间线之前是训练集
tes_sess = filter(lambda x: x[1] > splitdate, dates)   # 时间线之后是测试集

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]，升序排序
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]，升序
print(len(tra_sess))    # 186670    # 7966257
print(len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])   # 这两行代码用来输出排序后的训练集和测试集的前三个会话，用来检查结果的准确性
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())


#将训练集的会话转为序列，并且重新编号
# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []   # session对应的物品序列
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:   # s 是session的id
        seq = sess_clicks[s]
        outseq = []    # 初始化一个空列表，用于存储重新编号后的物品序列
        for i in seq:  #重新编号物品，使其从1开始
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur  若重新编号的物品序列长度小于2，则跳过当前会话
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print('item_ctr')
    print(item_ctr)     # 重新编号后的物品总数
    return train_ids, train_dates, train_seqs

# 测试集中只保留那些在训练集中出现过的物品
# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:   # item_dict 是用于存储训练集物品（items）的字典
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs

tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()

# 处理序列（去掉每个会话的最后一个点击的项目）
def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]  # labs 中存放的是原序列中最后一个物品
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids   # 最后输出 新序列（去掉labs）、日期、标签（原序列的最后一个物品）、商品id

tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
# ！！！这里如果自己的创新点是对的话，那麽这里数据集中应该还要存放一个时间戳。 tra = (tr_seqs, tr_labs, tr-dates)    tes = (te_seqs, te_labs, te_dates)
tra = (tr_seqs, tr_labs)   # 最终需要的训练集（去掉最后一个物品的新序列 + 最后一个目标物品）
tes = (te_seqs, te_labs)   # 同上
print('train_test')
print(len(tr_seqs))
print(len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])   # 这两行代码用来输出排序后的训练集和测试集的前三个会话，用来检查结果的准确性
all = 0

# tra_seqs 是原训练集序列、 tr_seqs是新训练集序列
for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all * 1.0/(len(tra_seqs) + len(tes_seqs)))  # 求平均序列长度
# 保存训练集和测试集的处理结果以及所有训练集的原始序列
if not os.path.exists('tmall'):
    os.makedirs('tmall')
pickle.dump(tra, open('tmall/train.txt', 'wb'))
pickle.dump(tes, open('tmall/test.txt', 'wb'))
pickle.dump(tra_seqs, open('tmall/all_train_seq.txt', 'wb'))

# Namespace(dataset='Tmall')
# Splitting train set and test set
# item_ctr
# 40728
# train_test
# 351268
# 25898
# avg length:  6.687663052493478
