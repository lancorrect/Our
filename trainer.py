import os
import sys
import copy
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from transformers import BertModel, AdamW
from time import strftime, localtime


class Trainer:
    def __init__(self, opt, model, train_dataloader, test_dataloader, logger):
        self.opt = opt
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger


    def _print_args(self):
        '''打印参数'''
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))  # prod的作用是返回张量中所有元素的乘积
            if p.requires_grad:
                n_trainable_params += n_params  # 可训练参数数量
            else:
                n_nontrainable_params += n_params  # 不可训练参数数量

        self.logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('training arguments:')

        for arg in vars(self.opt):
            self.logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))


    def _reset_params(self):
        '''重置参数'''
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)  # xavier_uniform_初始化
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)


    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        if self.opt.diff_lr:
            self.logger.info("layered learning rate on")
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)

        else:
            self.logger.info("bert learning rate on")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer


    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.opt.num_epoch):
            self.logger.info('>' * 60)
            self.logger.info('epoch: {}'.format(epoch+1))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, kl_loss = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                loss = criterion(outputs, targets) + kl_loss

                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    # n_correct和n_total使用+=的原因是它们统计的是在训练一代的过程中正确的个数和数据整体的个数
                    # 然后在整体测试集上测试效果，看模型在什么时候效果最好，如果效果好的话，直接保存下来
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()  # 训练几次以后在整个测试集上测试一下效果
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./log/state_dict'):
                                os.mkdir('./log/state_dict')
                            model_path = './log/state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name,
                                                                                                  self.opt.dataset,
                                                                                                  test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            self.logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    self.logger.info(
                        'loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc,
                                                                                         f1))
        return max_test_acc, max_f1, model_path


    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs, _ = self.model(inputs)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1


    def _test(self):
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        self.logger.info("Precision, Recall and F1-Score...")
        self.logger.info(test_report)
        self.logger.info("Confusion Matrix...")
        self.logger.info(test_confusion)


    def run(self):
        criterion = nn.CrossEntropyLoss()
        if 'bert' not in self.opt.model_name:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        else:
            optimizer = self.get_bert_optimizer(self.model)
        max_test_acc_overall = 0
        max_f1_overall = 0
        if 'bert' not in self.opt.model_name:
            self._reset_params()
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        self.logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)
        self.logger.info('>> saved: {}'.format(model_path))
        self.logger.info('#' * 60)
        self.logger.info('max_train_acc_overall:{}'.format(max_test_acc_overall))
        self.logger.info('max_f1_overall:{}'.format(max_f1_overall))
        self._test()

        final_result = {}
        final_result['acc'] = max_test_acc_overall
        final_result['f1'] = max_f1_overall
        final_result['model_path'] = model_path
        if 'bert' in self.opt.model_name:
            keys = ['learning_rate', 'bert_lr', 'l2reg', 'num_epoch', 'num_layers', 'batch_size', 'hidden_dim', 'bert_dim', 'input_dropout', 'gcn_dropout',
                    'bert_dropout', 'attention_heads', 'seed', 'fusion', 'alpha', 'beta', 'gama']
            base_dir = './results_bert/'
        else:
            keys = ['learning_rate', 'l2reg', 'num_epoch', 'batch_size', 'embed_dim', 'post_dim',
                    'pos_dim', 'hidden_dim', 'num_layers', 'input_dropout', 'gcn_dropout', 'rnn_hidden',
                    'attention_heads', 'seed', 'fusion', 'alpha', 'beta', 'gama']
            base_dir = './results'
        for arg in vars(self.opt):
            if arg in keys:
                final_result[arg] = (vars(self.opt)[arg])
        file_name = '{}-{}-{}.log'.format(self.opt.model_name, self.opt.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
        with open(os.path.join(base_dir, file_name), 'w', encoding='utf-8') as f:
            f.write(str(final_result))
            f.close()
