# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:54:28 2023

@author: 27812
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool

from transformers import AutoModel

# codebert_path = r'C:\Users\27812\.cache\torch\sentence_transformers\neulab_codebert-c'
# bert_path = r"C:\Users\27812\.cache\torch\sentence_transformers\bert-base-uncased"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        
        # logit_scale：clip里面默认的缩放因子
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, meta):
        # p：范数的阶数，默认为 2，表示使用 L2 范数进行归一化
        meta = F.normalize(meta, p=2, dim=1)
        # print('x:', x.shape)
        # print(meta.shape)
        logit_scale = self.logit_scale.exp()
        # cosine similarity as logits
        logits = logit_scale * x @ meta.t()
        # print('logits:', logits.shape)
        
        return logits


class Prompt(nn.Module):
    def __init__(self, bert_path):
        super(Prompt, self).__init__()
        
        self.bert = AutoModel.from_pretrained(bert_path)
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # bert out:768
        # bert-tiny out:128

        self.liner = nn.Linear(128*2, 16)
        
    def forward(self, x, prompt_vec):
        ctx_vec = prompt_vec[0].to(device)
        cls_vec = prompt_vec[1].to(device)
        # print('ctx_vec:', len(ctx_vec))
        
        with torch.no_grad():
            embed_prefix = self.bert(**ctx_vec).pooler_output
        # print('embedding:', embed_prefix.shape)
        
        with torch.no_grad():
            embed_cls = self.bert(**cls_vec).pooler_output
        
        cls_lst = []
        for c in embed_cls:
            n_cls= torch.cat((embed_prefix.squeeze(dim=0),c), dim=0)
            cls_lst.append(n_cls)
        
        embedding = torch.stack(cls_lst)
    
        ctx = self.liner(embedding)
        
        return ctx


class GCN(nn.Module):
    def __init__(self, in_channels):
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(in_channels, 64)

        self.conv2 = GCNConv(64, 64)
        
    def forward(self, x, edge_index, batch):
        # print('gcn_input:', x.shape)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = F.dropout(x, p=0.2)
        # print('gcn:', x.shape)
        
        x = global_mean_pool(x, batch)
        # print('global_mean_pool:', x.shape)
        return x


class BiLSTM(nn.Module):
    def __init__(self, num_hiddens, num_layers, reduction_out):
        super(BiLSTM, self).__init__()
        
        # bidirectional设为True即得到双向循环神经网络
        self.lstm = nn.LSTM(input_size=reduction_out,
                            hidden_size=num_hiddens,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=True)
        

    def forward(self, inputs):
        x = inputs
        # print('dfghj:', x.shape)
        output, (h, c) = self.lstm(x)  # output, (h, c)
        # print(output.shape)
        output = torch.cat((h[-2], h[-1]), -1)
        # print(output.shape)
        output = F.relu(output)
        # print(output.shape)

        return output


class MERG(nn.Module):
    def __init__(self, out_channels, args):
        super(MERG, self).__init__()
        num_hiddens = 32
        num_layers = 2
        reduction_out = 32
        
        codebert_path = args.codebert_path
        bert_path = args.bert_path
        
        self.codebert = AutoModel.from_pretrained(codebert_path)
        
        for param in self.codebert.parameters():
            param.requires_grad = False
            
        self.reduction = nn.Sequential(
            nn.Linear(768, reduction_out),
            nn.Tanh()
            )
        
        self.lstm = BiLSTM(num_hiddens, num_layers, reduction_out)
        
        self.GCN = GCN(reduction_out)
        
        self.prediction = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
            )
        
        self.prompt = Prompt(bert_path)
        
        self.d = D()
        
        
    def forward(self, n_input_ids, n_attention_mask, t_input_ids, t_attention_mask, edge_index, batch, prompt_vec):        
        # 1. Obtain node embeddings
        with torch.no_grad():
            n_embedd = self.codebert(n_input_ids, n_attention_mask).pooler_output   
        
        reduc_out1 = self.reduction(n_embedd)
        # print(n_embedd)
        output1 = self.GCN(reduc_out1, edge_index, batch)
        # print('GCN_out:', output1.size())
        
        # 2. Obtain txt embeddings 
        with torch.no_grad():    
            t_embedd = self.codebert(t_input_ids, t_attention_mask).last_hidden_state
        
        reduc_out2 = self.reduction(t_embedd)
        
        output2 = self.lstm(reduc_out2)
        # print('LSTM_out:', output2.size())
        
        # 3. Merge node-txt embeddings 
        output = torch.cat([output1, output2], 1)
        # print('model_output:', output.size())

        # x = self.linear(x)
        x = F.dropout(output, p=0.4)
        # print(x.size())
        x = self.prediction(x)
        # x = F.dropout(x, p=0.2)
        # print(x.size())
        
        meta = self.prompt(x, prompt_vec)
        # print('meta:', meta)
        prompt_out = self.d(x, meta)
        # print('prompt_out:', prompt_out)
        # print('x:', x)
        return F.log_softmax(prompt_out, dim=1)
