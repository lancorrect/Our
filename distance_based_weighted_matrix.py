import pdb
import numpy as np
from copy import deepcopy
import argparse


def simple_watch(index_dict, stoi, default_key=None):
    if default_key is not None:
        child = index_dict[default_key]
        for key in child.keys():
            print("{'%s':%d}" % (stoi[key], child[key]), end=' ')
    else:
        for child in index_dict:
            for key in child.keys():
                print("{'%s':%d}" % (stoi[key], child[key]), end=' ')
            print('', end='   ')
    print()


def aspect_oriented_tree(opt, token, head, as_start, as_end):

    '''
    generate distance based weighted matrix
    :param opt: 命令行参数
    :param token: 单词列表，即一句话被tokenize后的列表
    :param head: 依赖关系
    :param as_start: aspect起始位置
    :param as_end: aspect终止位置
    :return: 该条数据的distance based weighted matrix
    '''

    stoi = {}  # 索引对单词的词典
    for i, t in enumerate(token):
        stoi[i] = t
    # print(stoi)
    children = []  # 存储每个单词与其相连的单词的索引
    for _ in range(len(token)):
        children += [{}]

    # pdb.set_trace()

    # 查找连接的单词
    for i in range(len(token)):
        for j in range(len(head)):
            # 如果第j个单词的头部是第i个单词，且第j个单词不在第i个单词所属的连接单词字典里，还要不是根节点
            if head[j] - 1 == i and j not in children[i].keys() and head[j] != 0:
                children[i][j] = 1
                children[j][i] = 1  # 在第j个单词的字典中添加与第i个单词的距离，意思是对称
        # 如果第i个单词的头部所指的单词不在字典当中，且第i个单词的头部不是根节点
        if head[i] - 1 not in children[i].keys() and head[i] != 0:
            children[i][head[i] - 1] = 1
            children[head[i] - 1][i] = 1  # 在第i个单词头部所指的单词的字典中添加与第i个单词的距离，表达对称
    # simple_watch(children, stoi)

    # 计算每个单词与aspect的距离
    children_asp_all = []
    for asp_idx in range(as_start, as_end):
        children_asp = deepcopy(children)
        head_idx = list(children_asp[asp_idx].keys())  # 与aspect相连的单词集合
        head_stack = deepcopy(head_idx)  # 栈
        # print(head_idx)
        # 所有的单词还没有遍历完，并且栈中还依然有单词的话，才会接着遍历。
        # 防止出现aspect不是与所有单词相连的，栈中已经没有单词了，但是还在循环
        while (len(head_idx) < len(token)) and (len(head_stack) > 0):
            idx_in_sent = head_stack.pop(0)  # 栈顶单词
            ids = list(children_asp[idx_in_sent].keys())  # 栈顶单词的相连单词
            for idx in ids:
                # 如果索引为idx的单词不在存储与aspect相连单词的距离的字典中，且该单词不在aspect的范围里
                # head_idx还有一个作用就是判断是否被遍历过，这里没有考虑单词是否在aspect范围中，因为这样会影响接下来的计算
                if idx not in head_idx and idx != asp_idx:
                    # 把存储aspect与各个单词距离的字典更新，添加上与索引为idx的单词的距离
                    children_asp[asp_idx][idx] = children_asp[idx_in_sent][idx] + children_asp[asp_idx][idx_in_sent]
                    head_stack = [idx] + head_stack  # 更新栈，将新的单词索引放到栈顶
                    head_idx += [idx]  # 添加遍历过的单词索引到head_idx中
        # simple_watch(children_asp, stoi, asp_idx)
        children_asp_all.append(children_asp)

    # distance based weighted matrix，分为两种模式
    '''
    第一种模式：将多个aspect一起考虑。对于某个单词，查找并存储该单词与每个aspect的距离，取最小的距离为该单词到所有aspect的距离。
              在矩阵中表示为每个aspect与该单词的距离都是相同的，且是最短距离。不同aspect之间的距离设置为1，对同一个aspect距离同样是1(self-loop)

    第二种模式：分别考虑aspect。对于某个单词，查找并存储该单词与每个aspect的距离，矩阵对应位置上存储每个aspect和该单词的距离。
              不同aspect之间的距离不做特殊要求，该是啥是啥，对同一个aspect距离是1(self-loop)
    '''
    dm = np.ones((opt.max_length, opt.max_length)) * (-np.inf)  # 初始化distance based weighted matrix，每个元素是无穷大，方便后面比较大小
    if opt.fusion is True:
        # 第一种模式
        aspect_indices = list(range(as_start, as_end))  # aspect的索引列表
        # 第一层循环含义是遍历除单个aspect以外单词的索引
        for word_id in range(len(token)):
            distances = [np.inf]  # 存储在该单词位置上的距离，先有一个无穷大是因为dm中初始值为无穷大
            # 遍历aspect的索引，同时还需要索引的索引，因为children_asp_all的维度是aspect的数量，跟aspect的索引不匹配
            for child_id, asp_id in enumerate(aspect_indices):
                asp_child = children_asp_all[child_id][asp_id]  # 找到该aspect下跟其他所有单词的距离词典
                # 在这里有个try的原因是担心有的aspect跟某个单词不连接但是其他aspect是连接的，如果不连接的话，还是把dm中的值(无穷大)放过去
                try:
                    distances.append(asp_child[word_id])
                except:
                    distances.append(np.inf)
            real_distance = min(distances)  # 取最小值
            for asp_id in aspect_indices:
                dm[asp_id][word_id] = real_distance  # 在对应位置上赋值
                dm[word_id][asp_id] = deepcopy(dm[asp_id][word_id])  # 对称
        for asp_id in aspect_indices:
            for asp_mutual in aspect_indices:
                dm[asp_id][asp_mutual] = 1  # 不光跟其他aspect距离是1，跟自己也是1

    else:
        # 第二种模式
        aspect_indices = list(range(as_start, as_end))
        for child_id, asp_id in enumerate(aspect_indices):
            asp_child = children_asp_all[child_id][asp_id]
            word_indices = list(asp_child.keys())
            # 在这里就不需要比较大小，直接赋值即可
            for word_id in word_indices:
                dm[asp_id][word_id] = asp_child[word_id]
                dm[word_id][asp_id] = deepcopy(dm[asp_id][word_id])
        for asp_id in aspect_indices:
            dm[asp_id][asp_id] = 1  # self-loop，自己跟自己的距离才是1

    # self-loop
    for i in range(len(dm)):
        dm[i][i] = 1

    return dm


def aspect_oriented_tree_debug(opt, dataset, sent_id):

    '''
    debug模式是为了查看单个数据转成的matrix
    :param opt: 命令行参数
    :param dataset: 一条数据
    :param sent_id: 在数据集中的位置
    :return: 该条数据所对应的矩阵
    '''

    d = dataset[sent_id]
    token = d['token']
    head = d['head']
    stoi = {}  # 索引对单词的词典
    for i, t in enumerate(token):
        stoi[i] = t
    children = []  # 存储每个单词与其相连的单词的索引
    for _ in range(len(token)):
        children += [{}]

    dm_sent = []

    for aspect in d['aspects']:
        as_start = aspect['from']
        as_end = aspect['to']

        # pdb.set_trace()

        # 查找连接的单词
        for i in range(len(token)):
            for j in range(len(head)):
                # 如果第j个单词的头部是第i个单词，且第j个单词不在第i个单词所属的连接单词字典里，还要不是根节点
                if head[j] - 1 == i and j not in children[i].keys() and head[j] != 0:
                    children[i][j] = 1
                    children[j][i] = 1  # 在第j个单词的字典中添加与第i个单词的距离，意思是对称
            # 如果第i个单词的头部所指的单词不在字典当中，且第i个单词的头部不是根节点
            if head[i] - 1 not in children[i].keys() and head[i] != 0:
                children[i][head[i] - 1] = 1
                children[head[i] - 1][i] = 1  # 在第i个单词头部所指的单词的字典中添加与第i个单词的距离，表达对称
        # simple_watch(children, stoi)

        # 计算每个单词与aspect的距离
        children_asp_all = []
        for asp_idx in range(as_start, as_end):
            children_asp = deepcopy(children)
            head_idx = list(children_asp[asp_idx].keys())  # 与aspect相连的单词集合
            head_stack = deepcopy(head_idx)  # 栈
            # print(head_idx)
            # 所有的单词还没有遍历完，并且栈中还依然有单词的话，才会接着遍历。
            # 防止出现aspect不是与所有单词相连的，栈中已经没有单词了，但是还在循环
            while (len(head_idx) < len(token)) and (len(head_stack) > 0):
                idx_in_sent = head_stack.pop(0)  # 栈顶单词
                ids = list(children_asp[idx_in_sent].keys())  # 栈顶单词的相连单词
                for idx in ids:
                    # 如果索引为idx的单词不在存储与aspect相连单词的距离的字典中，且该单词不在aspect的范围里
                    # head_idx还有一个作用就是判断是否被遍历过，这里没有考虑单词是否在aspect范围中，因为这样会影响接下来的计算
                    if idx not in head_idx and idx != asp_idx:
                        # 把存储aspect与各个单词距离的字典更新，添加上与索引为idx的单词的距离
                        children_asp[asp_idx][idx] = children_asp[idx_in_sent][idx] + children_asp[asp_idx][idx_in_sent]
                        head_stack = [idx] + head_stack  # 更新栈，将新的单词索引放到栈顶
                        head_idx += [idx]  # 添加遍历过的单词索引到head_idx中
            # simple_watch(children_asp, stoi, asp_idx)
            children_asp_all.append(children_asp)

        # distance based weighted matrix
        if opt.fusion is True:
            # 第一种模式
            dm = np.ones((len(token), len(token))) * np.inf  # 初始化distance based weighted matrix，每个元素是无穷大，方便后面比较大小
            aspect_indices = list(range(as_start, as_end))  # aspect的索引列表
            # 第一层循环含义是遍历除单个aspect以外单词的索引
            for word_id in range(len(token)):
                distances = [np.inf]  # 存储在该单词位置上的距离，先有一个无穷大是因为dm中初始值为无穷大
                # 遍历aspect的索引，同时还需要索引的索引，因为children_asp_all的维度是aspect的数量，跟aspect的索引不匹配
                for child_id, asp_id in enumerate(aspect_indices):
                    asp_child = children_asp_all[child_id][asp_id]  # 找到该aspect下跟其他所有单词的距离词典
                    # 在这里有个try的原因是担心有的aspect跟某个单词不连接但是其他aspect是连接的，如果不连接的话，还是把dm中的值(无穷大)放过去
                    try:
                        distances.append(asp_child[word_id])
                    except:
                        distances.append(np.inf)
                real_distance = min(distances)  # 取最小值
                for asp_id in aspect_indices:
                    dm[asp_id][word_id] = real_distance  # 在对应位置上赋值
                    dm[word_id][asp_id] = deepcopy(dm[asp_id][word_id])  # 对称
            for asp_id in aspect_indices:
                for asp_mutual in aspect_indices:
                    dm[asp_id][asp_mutual] = 1  # 不光跟其他aspect距离是1，跟自己也是1

        else:
            # 第二种模式
            dm = np.ones((len(token), len(token))) * np.inf
            aspect_indices = list(range(as_start, as_end))
            for child_id, asp_id in enumerate(aspect_indices):
                asp_child = children_asp_all[child_id][asp_id]
                word_indices = list(asp_child.keys())
                # 在这里就不需要比较大小，直接赋值即可
                for word_id in word_indices:
                    dm[asp_id][word_id] = asp_child[word_id]
                    dm[word_id][asp_id] = deepcopy(dm[asp_id][word_id])
            for asp_id in aspect_indices:
                dm[asp_id][asp_id] = 1  # self-loop，自己跟自己的距离才是1

        for i in range(len(dm)):
            dm[i][i] = 1

        dm_sent.append(dm)

    return np.array(dm_sent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion', default=True, type=bool,
                        help='fuse distance based weighted matrices belonging to different aspects')
    parser.add_argument('--alpha', default=-1, type=int,
                        help='the weight of distance')
    parser.add_argument('--distance_matrix_debug', default=False, type=bool, help='debug mode')
    opt = parser.parse_args()

    '''with open('./dataset/Laptops_corenlp/train.json') as f:
            dataset = json.load(f)
            f.close()
    if opt.debug is False:
        dm_all = aspect_oriented_tree(opt, dataset)
        print(dm_all)
    else:
        dm_sent = aspect_oriented_tree_debug(opt, dataset, 1)
        print(dm_sent)
        print(len(dm_sent))'''


if __name__ == '__main__':
    main()
