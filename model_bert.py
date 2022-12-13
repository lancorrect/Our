
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(opt.bert_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        outputs1, outputs2, kl_loss,  pooled_output= self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2, pooled_output), dim=-1)
        logits = self.classifier(final_outputs)

        return logits, kl_loss


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_reshape, src_mask, aspect_mask = inputs
        h1, h2, kl_loss, pooled_output = self.gcn(inputs)
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)  
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2)
        outputs1 = (h1 * aspect_mask).sum(dim=1) / asp_wn  # 相当于只保留了aspect的隐藏向量并进行了归一化
        outputs2 = (h2 * aspect_mask).sum(dim=1) / asp_wn
        return outputs1, outputs2, kl_loss, pooled_output


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        '''self.attdim = 100
        self.W = nn.Linear(self.attdim,self.attdim)
        self.Wx= nn.Linear(self.attention_heads+self.attdim*2, self.attention_heads)
        self.Wxx = nn.Linear(self.bert_dim, self.attdim)
        self.Wi = nn.Linear(self.attdim,50)
        self.aggregate_W = nn.Linear(self.attdim*(opt.num_layers + 1), self.attdim)'''
        # gcn layer
        self.W = nn.ModuleList()  # 存储各种module，跟list类似，用append和extend来添加各类层
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim  # 第一层维数是bert的维数，后面是自定义的
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(self.opt, opt.attention_heads, self.bert_dim)
        self.wa = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.wa.append(nn.Linear(input_dim, self.mem_dim))

        self.ws = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.ws.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_reshape, src_mask, aspect_mask = inputs  # aspect_mask[batch_size, max_length]
        src_mask = src_mask.unsqueeze(-2) 
        batch = src_mask.size(0)
        len = src_mask.size()[2]
        
        sequence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids).values()  # 必须添加values，否则输出的结果是key，是一行字符串
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        # gcn_inputs = self.Wxx(gcn_inputs)
        aspect_ids = aspect_indices(aspect_mask)
        # asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)  # batch中每条数据的aspect所拥有的的单词数
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.bert_dim)
        aspect_outs = gcn_inputs*aspect_mask

        aspect_scores, s_attn = self.attn(gcn_inputs, gcn_inputs, src_mask, aspect_outs, aspect_ids)
        aspect_score_list = [attn_adj.squeeze(1) for attn_adj in torch.split(aspect_scores, 1, dim=1)]
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(s_attn, 1, dim=1)]
        multi_head_list = []
        outputs_dep = None
        adj_ag = None

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
        adj_s = src_mask.transpose(1, 2) * adj_s  # 把多余的对角线元素遮盖掉

        # distance based weighted matrix 把指数弄上去
        # adj_reshape = adj_reshape[:, :maxlen, :maxlen]
        adj_reshape = torch.exp(self.opt.alpha * adj_reshape)

        # aspect-aware attention * distance based weighted matrix
        distance_mask = (
                    aspect_score_avg > torch.ones_like(aspect_score_avg) * self.opt.beta)  # 找到大于阈值的位置，如果大于阈值，值为1，否则为0
        # 在distance矩阵中将值为True的位置上替换成1，说明该单词足够重要
        adj_reshape = adj_reshape.masked_fill(distance_mask, 1).cuda()
        # 注意力矩阵和距离矩阵相乘，得到aspect oriented gcn的邻接矩阵，它们相乘后数据类型会变成float64，之后会报错，所以先改变一下
        adj_ag = (adj_reshape * aspect_score_avg).type(torch.float32)

        # KL divergence
        kl_loss = F.kl_div(adj_ag.softmax(-1).log(), adj_s.softmax(-1), reduction='sum')
        kl_loss = torch.exp((-1.0) * kl_loss * self.opt.gama)

        # gcn layer
        denom_s = adj_s.sum(2).unsqueeze(2) + 1  # norm，加上1是防止出现sum后为0然后把0作为分母的情况
        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_s = gcn_inputs
        outputs_ag = gcn_inputs

        '''weight_adj=attn_tensor   
        gcn_outputs=gcn_inputs   
        layer_list = [gcn_inputs]'''
     

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

        return outputs_ag, outputs_s, kl_loss, pooled_output


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

    return aspect_scores, s_attn

 
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, opt, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()  
        self.opt = opt
        self.d_k = d_model // h  
        self.h = h    
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k)) 
        self.bias_m = nn.Parameter(torch.Tensor(1))
        self.dense = nn.Linear(d_model, self.d_k)

    def forward(self, query, key, mask, aspect, aspect_ids):
        mask = mask[:, :, :query.size(1)]  
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        aspect_scores, attn = None, None
        batch, aspect_dim = aspect.size()[0], aspect.size()[-1]
        aspect = self.dense(aspect)  # 将aspect向量的维度改变，从原来的维度大小改为每个注意力头中向量的维度大小
        aspect = aspect.unsqueeze(1).expand(batch, self.h, query.size()[2], self.d_k)  # 将aspect向量复制到每个注意力头上去
        aspect_scores, self_attn = attention(self.opt, query, key, aspect, aspect_ids, self.weight_m, self.bias_m, mask,
                                             self.dropout)

        return aspect_scores, self_attn