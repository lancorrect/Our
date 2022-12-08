import os
import sys
# sys.path.append(r'./LAL-Parser/src_joint')
import re
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset
from reshape_dependency_tree import reshape_dependency_tree
from distance_based_weighted_matrix import aspect_oriented_tree


def ParseData(data_path):
    '''
    处理数据集，将数据集转变成我们想要的形式
    :param data_path: 数据集的位置，格式为json文件
    :return: 所有数据的列表，每个列表是一个字典，字典中的元素有：text(文本), aspect(方面词), pos(词性), post(位置),
             head(依赖关系中的头部),deprel(依赖关系), length(该句话的总长度), label(情感极性), mask(), aspect_post(方面词的位置)
             text_list(将句子分割成单个单词的列表)
    '''
    with open(data_path) as infile:
        all_data = []  # 初始化所有数据的列表
        data = json.load(infile)  # 读入数据
        for d in data:
            # 每个aspect都有其对应的句子，也就是说一个句子可能出现多次
            for aspect in d['aspects']:
                text_list = list(d['token'])  # 将tokens列表化，里面都是单个的单词或者标点符号
                tok = list(d['token'])       # word token
                length = len(tok)            # 一句话的总长度
                # if args.lower == True:
                tok = [t.lower() for t in tok]  # 小写
                tok = ' '.join(tok)  # 合并成文本
                asp = list(aspect['term'])   # 获取aspect
                asp = [a.lower() for a in asp]  # aspect全部小写
                asp = ' '.join(asp)  # 若是多个单词组成的aspect，则将它们连接起来
                label = aspect['polarity']   # 情感极性
                pos = list(d['pos'])         # 词性标签
                head = list(d['head'])       # 依赖关系头部
                deprel = list(d['deprel'])   # 依赖关系
                # position
                aspect_post = [aspect['from'], aspect['to']]  # 方面词的位置，从开始到结束，左闭右开，开头是0
                post = [abs(i-aspect['from']) for i in range(aspect['from'])] \
                       +[0 for _ in range(aspect['from'], aspect['to'])] \
                       +[i-aspect['to']+1 for i in range(aspect['to'], length)]  # 相对位置
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]    # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]  # 小于或大于aspect位置的填0，aspect位置上填1
                
                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,\
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list}
                all_data.append(sample)  # 实际上是每个aspect都形成了一条数据，包含原本句子中的其他信息

    return all_data


def build_tokenizer(fnames, max_length, data_file):
    parse = ParseData
    if os.path.exists(data_file):
        print('loading tokenizer:', data_file)
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        tokenizer = Tokenizer.from_files(fnames=fnames, max_length=max_length, parse=parse)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer


class Vocab(object):
    '''
    vocabulary of dataset
    该类的作用主要是作为一个字典，并且提供单词和索引相互转换的函数，是带有一定功能的字典
     '''
    def __init__(self, vocab_list, add_pad, add_unk):
        self._vocab_dict = dict()  # 单词对索引
        self._reverse_vocab_dict = dict()  # 索引对单词
        self._length = 0
        # 如果添加pad
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = self._length  # pad在词典中的索引时词典的长度，此时长度为0
            self._length += 1  # 更新词典长度，此时为1
            self._vocab_dict[self.pad_word] = self.pad_id  # 将pad的键值对更新到字典中
        if add_unk:
            self.unk_word = '<unk>'  # 添加词典中没有的单词的标识符
            self.unk_id = self._length  # unk在字典中的长度，此时的长度为1
            self._length += 1  # 此时的长度为2
            self._vocab_dict[self.unk_word] = self.unk_id  # 更新字典
        for w in vocab_list:
            self._vocab_dict[w] = self._length  # 建立字典，每个单词对应一个索引
            self._length += 1
        for w, i in self._vocab_dict.items():   
            self._reverse_vocab_dict[i] = w  # 建立索引对单词的字典
    
    def word_to_id(self, word):
        ''' 该函数的作用是找到单词对应的索引 '''
        if hasattr(self, 'unk_id'):
            # 如果字典中没有word这个单词，则返回self.unk_id
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]
    
    def id_to_word(self, id_):
        ''' 该函数的作用是找到索引对应的单词 '''
        if hasattr(self, 'unk_word'):  # hasattr判断self中有没有unk_word这个属性
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]
    
    def has_word(self, word):
        ''' 判断某词在不在字典中'''
        return word in self._vocab_dict
    
    def __len__(self):
        return self._length
    
    @staticmethod
    def load_vocab(vocab_path: str):
        '''载入词典'''
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        '''存储词典'''
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class Tokenizer(object):
    '''
    transform text to indices
    该类虽然只是简单的实现了tokenizer的功能——将单词分隔开这么简单，但是它还兼顾形成字典(调用Vocab类)和将单词转换为索引序列的功能
    '''
    def __init__(self, vocab, max_length, lower, pos_char_to_int, pos_int_to_char):
        self.vocab = vocab
        self.max_length = max_length
        self.lower = lower

        self.pos_char_to_int = pos_char_to_int
        self.pos_int_to_char = pos_int_to_char
    
    @classmethod
    # classmethod可以在类实例化之前调用这个函数，cls指代的是这个类本身，可以再次调用构造函数
    def from_files(cls, fnames, max_length, parse, lower=True):
        corpus = set()  # 初始化语料库集合
        pos_char_to_int, pos_int_to_char = {}, {}
        # 循环遍历训练集和测试集
        for fname in fnames:
            for obj in parse(fname):
                text_raw = obj['text']  # 每条数据的文本
                if lower:
                    text_raw = text_raw.lower()  # 小写
                # split_text是将完整的句子分割开，形成单一的单词或者标点，类似于tokenize的效果
                # 这里相当于构建了训练集和测试集的词典
                corpus.update(Tokenizer.split_text(text_raw))
        return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower, pos_char_to_int=pos_char_to_int, pos_int_to_char=pos_int_to_char)
    
    @staticmethod
    # staticmethod可以不经过实例化直接使用
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)  # 形成长度为最大长度，每个元素都是pad_id的向量
        if truncating == 'pre':
            trunc = sequence[-maxlen:]  # 截掉前面的部分
        else:
            trunc = sequence[:maxlen]  # 截掉后面的部分
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc  # x前面部分改为trunc,相当于在trunc后padding
        else:
            x[-len(trunc):] = trunc  # 在trunc前面padding
        return x
    
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        '''将文本中每个单词分隔开，并且将单词转换为索引'''
        if self.lower:
            text = text.lower()
        words = Tokenizer.split_text(text)
        sequence = [self.vocab.word_to_id(w) for w in words]  # self.vocab里已经有了训练集和测试集里面的单词
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse()  
        return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.max_length, 
                                      padding=padding, truncating=truncating)
    
    @staticmethod
    def split_text(text):
        # for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
        #     text = text.replace(ch, " "+ch+" ")
        return text.strip().split()


class SentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, tokenizer, opt, vocab_help):

        parse = ParseData
        post_vocab, pos_vocab, dep_vocab, pol_vocab = vocab_help  # 因为tokenizer中已经有了训练集和测试集的词，所以不再需要他们的帮助词典
        data = list()
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            tokens = tokenizer.text_to_sequence(obj['text'])  # 将文本中的单词转为索引并且填充
            aspect = tokenizer.text_to_sequence(obj['aspect'])  # max_length=10
            post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in obj['post']]  # 位置，用索引来替代位置
            post = tokenizer.pad_sequence(post, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']]  # 词性标签
            pos = tokenizer.pad_sequence(pos, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]  # 依赖关系
            deprel = tokenizer.pad_sequence(deprel, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            mask = tokenizer.pad_sequence(obj['mask'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')

            '''adj = np.ones(opt.max_length) * opt.pad_id
            if opt.parseadj:
                from absa_parser import headparser
                # * adj
                headp, syntree = headparser.parse_heads(obj['text'])
                adj = softmax(headp[0])
                adj = np.delete(adj, 0, axis=0)
                adj = np.delete(adj, 0, axis=1)
                adj -= np.diag(np.diag(adj))
                if not opt.direct:
                    adj = adj + adj.T  # 如果是无向图，需要再加上自己的转置矩阵
                adj = adj + np.eye(adj.shape[0])  # GCN中的self-loop
                adj = np.pad(adj, (0, opt.max_length - adj.shape[0]), 'constant')
            
            if opt.parsehead:
                from absa_parser import headparser
                headp, syntree = headparser.parse_heads(obj['text'])
                syntree2head = [[leaf.father for leaf in tree.leaves()] for tree in syntree]
                head = tokenizer.pad_sequence(syntree2head[0], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            else:
                head = tokenizer.pad_sequence(obj['head'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
                '''
            head = tokenizer.pad_sequence(obj['head'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64',
                                          padding='post', truncating='post')
            length = obj['length']
            polarity = polarity_dict[obj['label']]
            adj_distance = aspect_oriented_tree(opt, token=obj['text_list'], head=obj['head'],
                                                as_start=obj['aspect_post'][0], as_end=obj['aspect_post'][0])
            '''adj_reshape = reshape_dependency_tree(aspect_post=obj['aspect_post'], head=obj['head'],
                                                maxlen=opt.max_length, tokens=obj['text'],
                                                multi_hop=opt.multi_hop, max_hop=opt.max_hop)'''
            data.append({
                'text': tokens,
                'aspect': aspect, 
                'post': post,
                'pos': pos,
                'deprel': deprel,
                'head': head,
                'mask': mask,
                'length': length,
                'polarity': polarity,
                'adj_reshape': adj_distance,
            })

        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

def _load_wordvec(data_path, embed_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        if embed_dim == 200:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>' or tokens[0] == '<unk>': # avoid them
                    continue
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        elif embed_dim == 300:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>': # avoid them
                    continue
                elif tokens[0] == '<unk>':
                    word_vec['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
                word = ''.join((tokens[:-300]))  # glove文件中，每行后300个元素是词嵌入，前面的都是单词，本行代码只是为了取得单词
                # 如果词典是空的或者词典中有glove所拥有的单词，则把这个单词的词嵌入放入到嵌入矩阵中
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[word] = np.asarray(tokens[-300:], dtype='float32')
        else:
            print("embed_dim error!!!")
            exit()
            
        return word_vec

def build_embedding_matrix(vocab, embed_dim, data_file):
    if os.path.exists(data_file):
        print('loading embedding matrix:', data_file)
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(vocab), embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = _load_wordvec(fname, embed_dim, vocab)  # 载入glove函数，并生成词嵌入矩阵(字典的数据结构)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))  # 在词典中找到索引所对应的单词，然后再在嵌入矩阵中找到次嵌入向量
            # 如果glove中有词典中的单词则赋值给次嵌入矩阵，如果没有相当于该单词的词嵌入是0向量
            # pad就是0向量，因为载入的时候pad的词嵌入没有要
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix


def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class ABSAGCNData(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.data = []
        parse = ParseData
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end: ]

            from absa_parser import headparser
            headp, syntree = headparser.parse_heads(text)
            ori_adj = softmax(headp[0])
            ori_adj = np.delete(ori_adj, 0, axis=0)
            ori_adj = np.delete(ori_adj, 0, axis=1)
            ori_adj -= np.diag(np.diag(ori_adj))
            if not opt.direct:
                ori_adj = ori_adj + ori_adj.T
            ori_adj = ori_adj + np.eye(ori_adj.shape[0])
            assert len(text_list) == ori_adj.shape[0] == ori_adj.shape[1], '{}-{}-{}'.format(len(text_list), text_list, ori_adj.shape)

            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                # 如例子所示，从每个单词分割下来的内容在left_tok2ori_map中都用一个索引所替代
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)                   # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)          # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)  # asp_start是aspect的起始位置
            offset = len(left)  # 单词个数，之后要用它形成索引
            for ori_i, w in enumerate(term):        
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term) 
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i+offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len-2*len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()
                    
            bert_tokens = left_tokens + term_tokens + right_tokens
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
            truncate_tok_len = len(bert_tokens)
            tok_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')
            # 相当于是遍历一个矩阵，保证每个单词跟别的单词都有关系
            for i in range(truncate_tok_len):
                for j in range(truncate_tok_len):
                    # 由于一个单词分出来了很多片段，tok2ori表示每个片段所属单词的索引，
                    # 保证同属于一个单词的片段在邻接矩阵上的值是相等的
                    tok_adj[i][j] = ori_adj[tok2ori_map[i]][tok2ori_map[j]]

            # CLS+context+SEP+aspect+SEP
            context_asp_ids = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(
                bert_tokens)+[tokenizer.sep_token_id]+tokenizer.convert_tokens_to_ids(term_tokens)+[tokenizer.sep_token_id]
            context_asp_len = len(context_asp_ids)  # bert处理句式的长度
            # 此处的padding只是添加完bert所需的cls和sep后还需要添加的长度，后面很多变量还是基于原始句子的长度所以不用它
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)  # 原始句子的长度
            # 前面两个1一个是代表CLS，另一个是SEP，最后一个1是SEP
            context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            # 减一是因为前面有一个1了，如果还用opt.max_length-context_len就会超出范围
            # 前面有个[0]的原因是模型中的输出结果第一位有一个cls分类隐藏向量，剩下的长度跟原始句子长度相同，
            # 之后需要src_mask运用到求注意力矩阵的过程中
            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)  # 原始句子的mask
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            context_asp_attention_mask = [1] * context_asp_len + paddings  # 注意力mask
            context_asp_ids += paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            # pad adj
            context_asp_adj_matrix = np.zeros(
                (tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('float32')
            context_asp_adj_matrix[1:context_len + 1, 1:context_len + 1] = tok_adj
            data = {
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'adj_matrix': context_asp_adj_matrix,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
            }
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
