"""
Prepare vocabulary and initial word vectors.
"""
import json
import tqdm
import pickle
import argparse
import numpy as np
from collections import Counter

class VocabHelp(object):
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)  # 初始化字典的单词列表，并把pad和unk放入进去
        for tok in specials:
            del counter[tok]
        
        # sort by frequency, then alphabetically。先按照频率排序，再按照字母表顺序排序
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)    # words_and_frequencies is a tuple

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos，构建单词对索引字典
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}


    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        '''扩展词典'''
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

def parse_args():
    '''定义好需要制作词典的数据集路径(data_dir)，和输出的字典路径(vocab_dir)'''
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', help='TACRED directory.')
    parser.add_argument('--vocab_dir', help='Output vocab directory.')
    parser.add_argument('--lower', default=True, help='If specified, lowercase all words.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # input files
    train_file = args.data_dir + '/train.json'
    test_file = args.data_dir + '/test.json'

    # output files
    # token
    vocab_tok_file = args.vocab_dir + '/vocab_tok.vocab'
    # position
    vocab_post_file = args.vocab_dir + '/vocab_post.vocab'
    # pos_tag
    vocab_pos_file = args.vocab_dir + '/vocab_pos.vocab'
    # dep_rel
    vocab_dep_file = args.vocab_dir + '/vocab_dep.vocab'
    # polarity
    vocab_pol_file = args.vocab_dir + '/vocab_pol.vocab'

    # load files，读取文件，把数据集中每个部分都读出来
    print("loading files...")
    # 返回的是所有数据的单词列表，所有数据的词性，所有数据的依赖关系和训练集的最大长度
    train_tokens, train_pos, train_dep, train_max_len = load_tokens(train_file)
    test_tokens, test_pos, test_dep, test_max_len = load_tokens(test_file)

    # lower tokens
    if args.lower:
        train_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in\
                (train_tokens, test_tokens)]

    # counters，计数相当于是把重复的单词只保留一个，并且之后还可以按照频率排序，一举两得
    token_counter = Counter(train_tokens+test_tokens)  # tokens计数
    pos_counter = Counter(train_pos+test_pos)  # 词性计数
    dep_counter = Counter(train_dep+test_dep)  # 依赖关系计数
    max_len = max(train_max_len, test_max_len)  # 训练集和测试集最大长度中取最大的
    post_counter = Counter(list(range(-max_len, max_len)))  # 从-max_len到max_len总共有这么多数，表示有这么多相对位置，对位置计数
    pol_counter = Counter(['positive', 'negative', 'neutral'])

    # build vocab
    print("building vocab...")
    token_vocab = VocabHelp(token_counter, specials=['<pad>', '<unk>'])
    pos_vocab = VocabHelp(pos_counter, specials=['<pad>', '<unk>'])
    dep_vocab = VocabHelp(dep_counter, specials=['<pad>', '<unk>'])
    post_vocab = VocabHelp(post_counter, specials=['<pad>', '<unk>'])
    pol_vocab = VocabHelp(pol_counter, specials=[])
    print("token_vocab: {}, pos_vocab: {}, dep_vocab: {}, post_vocab: {}, pol_vocab: {}".format(len(token_vocab), len(pos_vocab), len(dep_vocab), len(post_vocab), len(pol_vocab)))

    print("dumping to files...")  # 保存词典
    token_vocab.save_vocab(vocab_tok_file)
    pos_vocab.save_vocab(vocab_pos_file)
    dep_vocab.save_vocab(vocab_dep_file)
    post_vocab.save_vocab(vocab_post_file)
    pol_vocab.save_vocab(vocab_pol_file)
    print("all done.")

def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        pos = []
        dep = []
        max_len = 0
        for d in data:
            tokens.extend(d['token'])
            pos.extend(d['pos'])
            dep.extend(d['deprel'])
            max_len = max(len(d['token']), max_len)
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens, pos, dep, max_len

if __name__ == '__main__':
    main()