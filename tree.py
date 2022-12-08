"""
Basic operations on trees.
"""

import numpy as np
from collections import defaultdict

import copy

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None  # 初始化父节点
        self.num_children = 0  # 孩子节点的数量
        self.children = list()  # 孩子节点集合

    def add_child(self, child):
        child.parent = self  # 孩子节点的父节点是本身
        self.num_children += 1  # 孩子节点数量+1
        self.children.append(child)  # 添加节点

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head, tokens, len_):
    """
    Convert a sequence of head indexes into a tree object.
    输入：
    head: 一句话的头部集合，里面是依赖头部的逻辑索引
    tokens: 单词集合
    len_: 该句话的长度
    输出：
    root: 树的根结点
    """
    if isinstance(head, list) == False:
        tokens = tokens[:len_].tolist()
        head = head[:len_].tolist()
    root = None  # 初始化根结点

    nodes = [Tree() for _ in head]  # 初始化树节点，有几个head就有几个节点

    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].dist = -1 # just a filler
        if h == 0:
            root = nodes[i]  # 如果该单词的头为0，表明它在依赖关系中是根结点
        else:
            try:
                nodes[h-1].add_child(nodes[i])  # 如果不是0，说明该单词依赖另一个单词，且索引为h-1，把该单词添加到头部单词的孩子节点中
            except:
                print(len_)
                exit()

    assert root is not None
    return root

def tree_to_adj(sent_len, tree, directed=False, self_loop=True):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    输入：
        sent_len: 该batch当中最长的语句长度
        tree: 依赖树
        directed: 有向图还是无向图，默认False是无向图
        self_loop: GCN卷积时是否考虑节点自身信息的操作
    输出：
        ret:
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)  # 构建邻接矩阵，大小是根据该batch当中最长的语句长度决定的

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]  # 队列操作，提取并删除队列中的第一个元素，类似于列表中的pop操作

        # t.idx代表的是该节点在句子中的绝对位置，用于之后邻接矩阵上相应位置标为1，表示两个单词之间有依赖关系
        # 至于为什么还用列表存起来，这是为了之后的self-loop
        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1  # 邻接矩阵相应位置标为1，表示两者之间有依赖关系
        queue += t.children  # 将它的孩子节点送入到队列中，达到遍历整棵树的目的

    if not directed:
        ret = ret + ret.T  # 如果是无向图，则需要再加一下邻接矩阵自身的转置

    if self_loop:
        for i in idx:
            ret[i, i] = 1  # 对角线上有单词的地方标为1，因为邻接矩阵是按照该batch中最长句子的长度初始化的
            # 所以对角线上不是每个位置都有单词的
    return ret

