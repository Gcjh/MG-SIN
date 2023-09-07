import torch
import torch.nn as nn
from torch.nn.functional import softmax, gumbel_softmax, leaky_relu, relu
from transformers import BertPreTrainedModel, BertModel, RobertaModel
from model.layers import GraphConvolution, sparse_interaction
import torch.optim as optim
import numpy as np
from torch.nn.parameter import Parameter
import math

LAYERS = 8


class bert(nn.Module):
    def __init__(self, args):
        super(bert,self).__init__()
        self.hidden_dim = args.hidden_size
        self.batch_size = args.batch_size
        self.max_len = args.maxlen_train

        self.bert=BertModel.from_pretrained('bert-election2020-twitter-stance-biden-KE-MLM')
        self.dropout = nn.Dropout(0.5)
        
        self.fc0 = nn.Linear(self.hidden_dim, 2)
        self.fc1 = nn.Linear(self.hidden_dim, 3)
        

    def forward(self, input_ids, attention_mask, graph, target_len):
        ### input_ids.size : 3 batch, sentence_len, hidden_dim
        output = self.bert(input_ids[0], attention_mask[0]).last_hidden_state
        output = self.dropout(output)
       
        logits0 = output.mean(axis=1, keepdim=False)
        logits0 = self.dropout(logits0)
        logits0 = self.fc0(logits0)
        
        logits1 = output.mean(axis=1, keepdim=False)
        logits1 = self.dropout(logits1)
        logits1 = self.fc1(logits1)

        return logits0, logits1, 0, 0

class MGSIN(nn.Module):
    def __init__(self, args):
        super(MGSIN,self).__init__()
        self.hidden_dim = args.hidden_size
        self.batch_size = args.batch_size
        self.max_len = args.maxlen_train
        
        self.bert=BertModel.from_pretrained('covid-bert')
        #self.bert = RobertaModel.from_pretrained('roberta')
        self.dropout = nn.Dropout(0.5)
        self.bert_dropout = nn.Dropout(0.1)
        
        self.gcn0 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcn1 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcn2 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        
        self.gcn3 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcn4 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcn5 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        
        self.sparse0 =  sparse_interaction(self.max_len)
        self.sparse1 =  sparse_interaction(self.max_len)

        
        self.fc0 = nn.Linear(self.hidden_dim, 3)
        self.fc1 = nn.Linear(self.hidden_dim, 3)

    def attention(self, pool, output, feature, masks = None):
        f_s = torch.zeros_like(feature)
        if masks != None:
            for i in range(feature.shape[0]):
                f_s[i] = torch.mul(masks[i], feature[i])
        else:
            for i in range(feature.shape[0]):
                f_s[i] = feature[i]
        atten = torch.matmul(f_s, pool)
        atten = softmax(atten, dim = 1)
        logits = torch.mul(output, atten).sum(axis = 1, keepdim = False)
        return logits, atten

    def get_l_sh(self, x0, y0, x1, y1, layer):
        a = torch.sum(torch.abs(x0 - x1))
        a = (LAYERS - layer) / LAYERS * a
        b = torch.sum(torch.abs(y0 - y1))
        b = (LAYERS - layer) / LAYERS * b
        return (a + b) / 2

    def forward(self, input_ids, attention_mask, graph, target_len):
        ### input_ids.size : 3 batch, sentence_len, hidden_dim
        out = self.bert(input_ids[2], attention_mask[2])

        pool = out.pooler_output
        pool = pool.view(pool.shape[0], pool.shape[1], -1)
        
        output = out.last_hidden_state
        output = self.bert_dropout(output)
        outputs = torch.stack((output, output, output))
      
        graph0 = graph[0].squeeze(0)
        graph1 = graph[1].squeeze(0)
        graph2 = graph[2].squeeze(0)
      
        #1       
        feature0 = leaky_relu(self.gcn0(outputs[0], graph0))
        feature1 = leaky_relu(self.gcn1(outputs[1], graph1))
        feature2 = leaky_relu(self.gcn2(outputs[2], graph2))
        
        features = torch.stack((feature0, feature1, feature2))###MGSIN
        outputs, l1, x1, y1 = self.sparse0(features, 1)###MGSIN

        outputs[0] = self.dropout(outputs[0])
        outputs[1] = self.dropout(outputs[1])
        outputs[2] = self.dropout(outputs[2])
        
        #2
        feature0 = leaky_relu(self.gcn3(outputs[0], graph0))
        feature1 = leaky_relu(self.gcn4(outputs[1], graph1))
        feature2 = leaky_relu(self.gcn5(outputs[2], graph2))
        
        features = torch.stack((feature0, feature1, feature2))###MGSIN
        outputs, l2, x2, y2 = self.sparse1(features, 2)###MGSIN
        
       
        masks = []
        for i in range(output.shape[0]):
            mask = np.ones((output.shape[1] - target_len[i], 1))
            mask = np.pad(mask, ((target_len[i], 0), (0, 0)), 'constant')
            masks.append(mask)
        masks = torch.tensor(masks).cuda(0)
        
        p = pool.view(pool.shape[0], -1)
        
        logits0, atten0 = self.attention(pool, output, outputs[0]) ###MGSIN 
        logits0 = self.dropout(logits0)
        logits0 = self.fc0(logits0)
        
        logits1, atten1  = self.attention(pool, output, outputs[1])###MGSIN
        logits1 = self.dropout(logits1)
        logits1 = self.fc1(logits1)
        
        l_sp = (l1 + l2) / 2.0   ###MGSIN
        l_sh = 1 - torch.log(self.get_l_sh(x1, y1, x2, y2, 1))  ###MGSIN

        return logits0, logits1, l_sp, l_sh, [atten0, atten1]
        
