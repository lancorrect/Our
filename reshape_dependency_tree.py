from copy import deepcopy
import numpy as np


def reshape_dependency_tree(aspect_post, head, maxlen, tokens=None, multi_hop=False, max_hop=3):
    '''
    Adding multi hops
    This function is at the core of our algo, it reshape the dependency tree and center on the aspect.
    In open-sourced edition, I choose not to take energy(the soft prediction of dependency from parser)
    into consideration. For it requires tweaking allennlp's source code, and the energy is space-consuming.
    And there are no significant difference in performance between the soft and the hard(with non-connect) version.

    '''

    as_start = aspect_post[0]
    as_end = aspect_post[1]

    head_idx = []  # 跟方面词有关系的词的索引，不管是方面词作为头还是尾部
    adj_reshape = np.zeros((maxlen, maxlen), dtype=np.float32)
    # 1 hop  数据集中标出来的位置都是日常生活中的位置索引，真正在列表中还需要每个索引值减一

    for i in range(as_start, as_end):  # 这里的as_start, as_end是方面词开始位置和结束位置(绝对位置，左闭右开，即从as_start开始算)，循环的目的是找到与循环变量有依赖关系的词
        for j in range(len(head)):
            if i == head[j] - 1:  # 如果方面词作为依赖关系中的head
                # not root, not aspect，第一个子条件保证不是aspect,把aspect的位置过掉，第二个子条件保证不是root(这里head中不包括root，所以不用写)
                # 第三个子条件保证之前没有遍历过
                if (j < as_start or j >= as_end) and j not in head_idx:
                    adj_reshape[i][j] = 1
                    adj_reshape[j][i] = 1
                    head_idx.append(j)  # 存储遍历过的单词的索引
        # 把方面词作为尾部的连接线的头部不是方面词，并且也不能是root，不考虑有没有遍历过是因为跟方面词连接的每个单词在我看来都挺重要的，所以我就都带上
        if (head[i]-1 < as_start or head[i]-1 >= as_end) and head[i] != 0:
            adj_reshape[i][head[i]-1] = 1
            adj_reshape[head[i]-1][i] = 1
            head_idx.append(head[i]-1)  # 这里是保证之后广度搜索时可以探索方面词头部单词所连接的词是什么

    if multi_hop:  # 广度优先搜索
        current_hop = 2  # 现在的跳数
        added = True
        # 第一层遍历是确定是否已经遍历完所有节点
        # 循环条件是首先现在的跳数小于等于最大跳数，第二是dep_idx中索引的数量小于单词数量，保证遍历不会超出句子长度范围
        while current_hop <= max_hop and len(head_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(head_idx)  # deepcopy的含义是深复制，即将被复制对象再复制一遍成为一个全新的个体存在
            for i in dep_idx_temp:  # 第二层循环遍历目前在队列中的节点
                for j in range(len(head)):  # 第三层循环遍历跟目前节点有关系的节点，看他们跟方面词又没有关系，若有，是几跳
                    if i == head[j] - 1:  # i索引的单词是j索引单词的头部
                        if (j < as_start or j >= as_end) and j not in head_idx:
                            adj_reshape[i][j] = 1
                            adj_reshape[j][i] = 1
                            head_idx.append(j)
                            added = True  # 说明添加了新的单词进来，再次循环看看又没有新的符合要求的依赖关系
                # 查找i索引单词的头部，保证不能是root和aspect
                if (head[i]-1 < as_start or head[i]-1 >= as_end) and head[i] != 0 and head[i]- 1 not in head_idx:
                    adj_reshape[i][head[i]-1] = 1
                    adj_reshape[head[i]-1][i] = 1
                    head_idx.append(head[i]-1)
                    added = True
            current_hop += 1

    for i in range(maxlen):
        adj_reshape[i][i] = 1

    return adj_reshape
