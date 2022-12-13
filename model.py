import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree import Tree, head_to_tree, tree_to_adj
from reshape_dependency_tree import reshape_dependency_tree
import math
import copy

class GCNClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        in_dim = opt.hidden_dim * 2
        self.opt = opt
        self.gcn_model = GCNAbsaModel(opt, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(in_dim, opt.polarities_dim)

    def forward(self, inputs):
        outputs1, outputs2, kl_loss = self.gcn_model(inputs)
        # final_output = outputs2
        final_output = torch.cat((outputs1, outputs2), dim=-1)
        logits = self.classifier(final_output)
        return logits, kl_loss

class GCNAbsaModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        '''self.emb = nn.Embedding(opt.tok_size, opt.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix.cuda(), requires_grad=False)'''

        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix, dtype=torch.float), freeze=True)
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None        # POS emb
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0) if opt.post_dim > 0 else None    # position emb
        embeddings = (self.emb, self.pos_emb, self.post_emb)

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt.hidden_dim, opt.num_layers)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, adj_reshape = inputs           # unpack inputs, l为长度
        maxlen = max(l.data)
        # adj_spacy = adj_spacy[:, :maxlen, :maxlen]
        mask = mask[:, :maxlen]  # 经过BiLSTM以后，第二维的维度变化，所以mask的维度也要做到相应变化，之前是max_length大小

        def inputs_to_tree_reps(head, words, l):
            # l中保存了同一个batch里所有语句的长度，遍历l的个数就是在遍历语句的个数，在此就是要生成多少树
            trees = [head_to_tree(head[i], words[i], l[i]) for i in range(len(l))]  # 返回依赖树集合
            adj = [tree_to_adj(maxlen, tree, directed=self.opt.direct, self_loop=self.opt.loop).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda()

        def outputs_reshape_tree(asp_start, asp_end, head, words, l):
            # l中保存了同一个batch里所有语句的长度，遍历l的个数就是在遍历语句的个数，在此就是要生成多少树
            asp_posts = [[asp_start[i], asp_end[i]] for i in range(len(l))]
            adj_reshape = [reshape_dependency_tree(aspect_post=asp_posts[i], head=head[i], tokens=words[i],
                                                   maxlen=maxlen).reshape(1, maxlen, maxlen) for i in range(len(l))]
            adj = np.concatenate(adj_reshape, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda()

        adj = inputs_to_tree_reps(head.data, tok.data, l.data)  # if self.opt.spacy is False else adj_spacy
        # adj_reshape = outputs_reshape_tree(asp_start.data, asp_end.data, head.data, tok.data, l.data)
        # adj_reshape = adj_reshape[:, :maxlen, :maxlen]
        h1, h2, kl_loss = self.gcn(inputs)
        
        # avg pooling asp feature, h:(16,28,50)
        asp_wn = mask.sum(dim=1).unsqueeze(-1)                        # aspect words num, mask:(16,85), asp_wn:(16,1)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.hidden_dim)    # mask for h, mask:(16,85,50)
        outputs1 = (h1 * mask).sum(dim=1) / asp_wn                        # mask h1
        outputs2 = (h2 * mask).sum(dim=1) / asp_wn                        # mask h2

        return outputs1, outputs2, kl_loss

class GCN(nn.Module):
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = opt.embed_dim+opt.pos_dim+opt.post_dim
        self.emb, self.pos_emb, self.post_emb = embeddings

        # rnn layer
        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, opt.rnn_hidden, opt.rnn_layers, batch_first=True,
                           dropout=opt.rnn_dropout, bidirectional=opt.bidirect)

        if opt.bidirect:
            self.in_dim = opt.rnn_hidden * 2
        else:
            self.in_dim = opt.rnn_hidden

        # drop out
        self.rnn_drop = nn.Dropout(opt.rnn_dropout)
        self.in_drop = nn.Dropout(opt.input_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # gcn layer
        self.wa = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.wa.append(nn.Linear(input_dim, self.mem_dim))

        self.ws = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.in_dim if j == 0 else self.mem_dim
            self.ws.append(nn.Linear(input_dim, self.mem_dim))

        # aspect-aware attention
        self.attention_heads = opt.attention_heads
        self.attn = MultiHeadAttention(self.opt, self.attention_heads, self.mem_dim * 2)

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size, emb_class='pos'):
        h0, c0 = rnn_zero_state(batch_size, self.opt.rnn_hidden, self.opt.rnn_layers, self.opt.bidirect)
        # pack_padded_sequence是一个对rnn的输入数据进行一个压紧的操作，由于序列是变长的，在经过padding后产生了冗余，用这个函数可以压紧
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        # pad_packed_sequence与pack_padded_sequence相反，将压紧的序列填充回来，第二个维度的大小取决于输入序列中最长的序列长度
        # 所以在之后的维度计算过程中就会出现mask与隐藏向量不一致的情况，故需要对mask进行操作，截断道跟最长序列长度一样的长度
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, adj_reshape = inputs           # unpack inputs
        maxlen = max(l.data)
        src_mask = (tok != 0).unsqueeze(-2)  # 查看序列当中哪些位置是有单词的，如果该位置上是有单词的，则值不等于0.故src_mask中的值为True或False
        # [batch_size, maxlen, 1]，mask_是用来对最终的自注意力矩阵操作，将有单词的位置保留值，没有单词的位置值变为0
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs = embs + [self.pos_emb(pos)]
        if self.opt.post_dim > 0:
            embs = embs + [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # 为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous
        # chunk(连续的块)。
        self.rnn.flatten_parameters()

        # rnn layer, tok.size->(batch_size, max_length), l是一个batch里所有句子的真实长度，是一个列表
        # rnn的输出维度：(batch_size, max_len, hidden_size*2) max_len代表的是该batch里所有的句子最大的长度，最后一维乘以2是因为双向LSTM
        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, l.to('cpu'), tok.size()[0], 'pos'))

        # aspect-aware attention
        # 原始的mask维度为[batch_size, max_length]，只在aspect的位置上记录值为1
        aspect_ids = aspect_indices(mask)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.hidden_dim * 2)  # [batch_size, max_length, hidden_dim*2]
        mask = mask[:, :maxlen, :]  # [batch_size, max_len, hidden_dim*2]
        aspect_outs = (gcn_inputs * mask)

        aspect_scores, s_attn = self.attn(gcn_inputs, gcn_inputs, src_mask, aspect_outs, aspect_ids)
        aspect_score_list = [attn_adj.squeeze(1) for attn_adj in torch.split(aspect_scores, 1, dim=1)]
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(s_attn, 1, dim=1)]

        aspect_score_avg = None
        adj_s = None

        # Average Aspect-aware Attention scores
        for i in range(self.attention_heads):
            if aspect_score_avg is None:
                aspect_score_avg = aspect_score_list[i]
            else:
                aspect_score_avg += aspect_score_list[i]
        aspect_score_avg = aspect_score_avg / self.attention_heads

        # * Average Multi-head Attention matrices
        for i in range(self.attention_heads):
            if adj_s is None:
                adj_s = attn_adj_list[i]
            else:
                adj_s += attn_adj_list[i]
        adj_s = adj_s / self.attention_heads

        for j in range(adj_s.size(0)):
            # torch.diag(adj_ag[j])维度为[maxlen]， torch.diag(torch.diag(adj_ag[j]))维度为[maxlen, maxlen]
            # 去掉单词本身对自己的注意力，第一个diag是提取出来对角线并形成向量，外面再套一个diag是生成一个矩阵，对角线元素为之前的向量
            adj_s[j] -= torch.diag(torch.diag(adj_s[j]))
            adj_s[j] += torch.eye(adj_s[j].size(0)).cuda()  # self-loop
        adj_s = mask_ * adj_s  # 把多余的对角线元素遮盖掉

        # distance based weighted matrix 把指数弄上去
        adj_reshape = adj_reshape[:, :maxlen, :maxlen]
        adj_reshape = torch.exp(self.opt.alpha*adj_reshape)

        # aspect-aware attention * distance based weighted matrix
        distance_mask = (aspect_score_avg > torch.ones_like(aspect_score_avg)*self.opt.beta)  # 找到大于阈值的位置，如果大于阈值，值为1，否则为0
        # 在distance矩阵中将值为True的位置上替换成1，说明该单词足够重要
        adj_reshape = adj_reshape.masked_fill(distance_mask, 1).cuda()
        # 注意力矩阵和距离矩阵相乘，得到aspect oriented gcn的邻接矩阵，它们相乘后数据类型会变成float64，之后会报错，所以先改变一下
        adj_ag = (adj_reshape * aspect_score_avg).type(torch.float32)

        # KL divergence
        kl_loss = F.kl_div(adj_ag.softmax(-1).log(), adj_s.softmax(-1), reduction='sum')
        kl_loss = torch.exp((-1.0)*kl_loss*self.opt.gama)
        
        # gcn layer
        denom_s = adj_s.sum(2).unsqueeze(2) + 1    # norm，加上1是防止出现sum后为0然后把0作为分母的情况
        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_s = gcn_inputs
        outputs_ag = gcn_inputs

        for l in range(self.layers):
            # ************Aspect oriented gcn*************
            # Ax_pos:(batch_size, max_len, rnn_hidden*2), adj:(batch_size,max_len,max_len)
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.wa[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            # ************self-attention gcn*************
            # Ax_post:(batch_size, max_len, rnn_hidden*2), adj:(batch_size,max_len,max_len)
            Ax_s = adj_s.bmm(outputs_s)
            AxW_s = self.ws[l](Ax_s)
            AxW_s = AxW_s / denom_s
            gAxW_s = F.relu(AxW_s)

            # * mutual Biaffine module
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine1), torch.transpose(gAxW_s, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_s, self.affine2), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            gAxW_ag, gAxW_s = torch.bmm(A1, gAxW_s), torch.bmm(A2, gAxW_ag)
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag
            outputs_s = self.gcn_drop(gAxW_s) if l < self.layers - 1 else gAxW_s

        '''# ************Aspect+POS*************
        for l in range(self.layers):
            # Ax_pos:(batch_size, max_len, rnn_hidden*2), adj:(batch_size,max_len,max_len)
            Ax_pos = adj_reshape.bmm(outputs_pos)
            AxW_pos = self.W_pos[l](Ax_pos)
            AxW_pos = AxW_pos / denom_reshape
            gAxW_pos = F.relu(AxW_pos)

        # ************Aspect+Position*************
        for l in range(self.layers+1):
            # Ax_post:(batch_size, max_len, rnn_hidden*2), adj:(batch_size,max_len,max_len)
            Ax_post = adj.bmm(outputs_post)
            AxW_post = self.W_post[l](Ax_post)
            AxW_post = AxW_post / denom
            outputs_post = F.relu(AxW_post)
        gAxW_post = outputs_post

        A1 = F.softmax(torch.bmm(torch.matmul(gAxW_pos, self.affine1), torch.transpose(gAxW_post, 1, 2)), dim=-1)
        A2 = F.softmax(torch.bmm(torch.matmul(gAxW_post, self.affine2), torch.transpose(gAxW_pos, 1, 2)), dim=-1)
        gAxW_pos, gAxW_ag = torch.bmm(A1, gAxW_post), torch.bmm(A2, gAxW_pos)
        outputs_pos = self.pos_drop(gAxW_pos)
        outputs_post = self.post_drop(gAxW_post)'''

        return outputs_ag, outputs_s, kl_loss

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def aspect_indices(mask):
    '''
    返回每条数据下，属于aspect的单词的索引
    '''
    aspect_id_mask = copy.deepcopy(mask).cpu()  # 复制aspect的mask并转到cpu
    aspect_id_mask = torch.nonzero(aspect_id_mask).numpy()  # 找到矩阵中非零元素，返回的结果是一个两维矩阵
    aspect_id_dict = {}  # 建立字典，键是为了标识属于哪一条数据，值是该条数据下，属于aspect的单词索引，可能有多个或一个
    for elem in aspect_id_mask:
        key = elem[0]
        value = elem[1]
        if key in aspect_id_dict.keys():
            aspect_id_dict[key].append(value)
        else:
            aspect_id_dict[key] = [value]
    # 把键去掉，只留下值，成为一个二维矩阵，每一行的索引代表的就是某一条数据，每一行的值代表的就是在该条数据下，属于aspect的单词的索引
    aspect_ids = list(aspect_id_dict.values())
    return aspect_ids


def attention(opt, query, key, aspect, aspect_ids, weight_m, bias_m, mask, dropout):
    d_k = query.size(-1)
    maxlen = query.size(-2)
    batch = query.size()[0]
    attn_heads = weight_m.size(0)
    weight_m = weight_m.unsqueeze(0).expand(batch, attn_heads, d_k, d_k)
    # aspect-aware attention
    aspect_scores = None
    if opt.fusion is True:
        asps = torch.tanh(
            torch.add(torch.matmul(torch.matmul(aspect, weight_m), key.transpose(-2, -1)), bias_m))
        aspects_avg = asps.sum(dim=-2) / maxlen  # [batch_size, h, maxlen]，对同一条数据中所有属于aspect单词的向量进行平均池化
        aspect_scores = torch.zeros_like(asps)  # 存储所有结果
        for asp_id in range(len(aspect_ids)):
            for h in range(attn_heads):
                asp_vec = aspects_avg[asp_id, h, :]  # 挑选出每条数据对应的aspect向量
                aspect_scores[asp_id, h, :] = asps[asp_id, h, :]
                # aspect_ids存储的是每条数据中，属于aspect的单词的索引，他可能有多个或者一个。在矩阵对应的行和列处赋值上之前挑出的向量
                # 这里在行和列分别赋值的原因是保证图是一个无向图，在邻接矩阵上的表示为对称矩阵，同时也表示所有属于aspect的单词所在的行和列
                # 上值都是相同的，符合融合的想法
                for idx in aspect_ids[asp_id]:
                    aspect_scores[asp_id][h][idx, :] = asp_vec
                    aspect_scores[asp_id][h][:, idx] = asp_vec

    else:
        aspect_scores = torch.tanh(
            torch.add(torch.matmul(torch.matmul(aspect, weight_m), key.transpose(-2, -1)), bias_m))
    # self-attention
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 自注意力机制中需要转置
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 这里的mask指的是句子中有单词的位置，如果值为0，则替换成-1e9

    s_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        s_attn = dropout(s_attn)

    return aspect_scores.cuda(), s_attn.cuda()


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, opt, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.opt = opt
        self.d_k = d_model // h  # 输入数据的维度除以注意力头数得到的每个注意力头的维度是多少
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 克隆两个线性层
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model  # 输入数据的维度
        self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k))
        self.bias_m = nn.Parameter(torch.Tensor(1))
        self.dense = nn.Linear(d_model, self.d_k)

    def forward(self, query, key, mask, aspect, aspect_ids):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # batch的大小
        # 这里首先将query和key输入到了liner层中含义是分别与权重矩阵相乘
        # transpose将第二维和第三维转置，就是将头数和句子长度交换一下，则该矩阵某一行的意思是在一个batch下，某个注意力头下某个单词的向量
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        aspect_scores, attn = None, None
        batch, aspect_dim = aspect.size()[0], aspect.size()[-1]
        aspect = self.dense(aspect)  # 将aspect向量的维度改变，从原来的维度大小改为每个注意力头中向量的维度大小
        aspect = aspect.unsqueeze(1).expand(batch, self.h, query.size()[2], self.d_k)  # 将aspect向量复制到每个注意力头上去
        aspect_scores, self_attn = attention(self.opt, query, key, aspect, aspect_ids, self.weight_m, self.bias_m, mask, self.dropout)

        return aspect_scores, self_attn

