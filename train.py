# -*- coding: utf-8 -*-
"""
@author: 27812
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch
from torch.utils.data import DataLoader
from torch_geometric import loader

from transformers import AutoTokenizer

from init_prompt import init_prompt
from model import MERG
from config import load_args

import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_warning()


seed = 2023
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子

def run_time(begin_time):
    end_time = time.time()
    total_time = end_time - begin_time
    return total_time

def saveModel():
    path = f"./checkpoints/myPrompt_acc{int_acc}_epoch{num_epochs}.pth"
    torch.save(model.state_dict(), path)
    
    
def train(train_graph, train_code, prompt_vec):
    model.train()
    total_loss, total_acc, step, total = 0., 0., 0., 0.
    
    # Iterate in batches over the training dataset.
    for i, j in zip(train_graph, train_code):  
        # Graph_Loader
        n_input_ids=[]
        n_attention_mask=[]
        for n in i.x:           
            n_token = tokenizer(n,
                              padding = "max_length",
                              max_length = 32,
                              truncation=True,
                              return_tensors = "pt")
            
            input_ids = n_token['input_ids']
            attention_mask = n_token['attention_mask']
            
            n_input_ids.append(input_ids)
            n_attention_mask.append(attention_mask)
            
        n_input_ids = torch.cat(n_input_ids).to(device)
        n_attention_mask = torch.cat(n_attention_mask).to(device)
        # print(n_input_ids.shape)
        
        edge_index = i.edge_index.to(device)
        batch = i.batch.to(device)
        # print(batch.shape)
        y = i.y.to(device)
             
        # Code_Loader
        t_input_ids = j[0].to(device)
        t_attention_mask = j[1].to(device)
        
        prompt_vec = prompt_vec
        
        # print(x1.shape)
        
        optimizer.zero_grad()  # Clear gradients.
        
        with autocast():
            out = model(n_input_ids,
                        n_attention_mask,
                        t_input_ids,
                        t_attention_mask,
                        edge_index,
                        batch,
                        prompt_vec)  # Perform a single forward pass.
            
            # print('out:', out)
            # print(F.log_softmax(out, dim=1),out.argmax(dim=1),y-1)
            
            loss = criterion(out, y)  # Compute the loss.
    
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients. 
        
        # 半精度
        # gradscaler.scale(loss).backward()
        # gradscaler.step(optimizer)
        # gradscaler.update()
        
        # print(out.argmax(dim=1))
        # print(y)
        
        total_acc += (out.argmax(dim=1) == y).sum().cpu().item()
        total_loss += loss.cpu().item()
        
        step += 1
        total += y.shape[0]
        # print(out)

    return total_loss / step, total_acc / total* 100.
    

def test(test_graph, test_code, prompt_vec):
    model.eval()  
    total_acc, total_loss, total, step = 0., 0., 0. ,0.
    labels = []
    predicts = []
    # Iterate in batches over the training/test dataset.
    with torch.no_grad():
        for i, j in zip(test_graph, test_code):
            # # Graph_Loader
            # n_input_ids=[]
            # n_attention_mask=[]
            # print(i.x)
            # for n in i.x:           
            #     n_token = tokenizer(n,
            #                       padding = "max_length",
            #                       max_length = 32,
            #                       truncation=True,
            #                       return_tensors = "pt")
                
            #     input_ids = n_token['input_ids']
            #     attention_mask = n_token['attention_mask']
                
            #     n_input_ids.append(input_ids)
            #     n_attention_mask.append(attention_mask)
                
            # n_input_ids = torch.cat(n_input_ids).to(device)
            # n_attention_mask = torch.cat(n_attention_mask).to(device)
            # # print(n_input_ids.shape)
            
            flatten_list = list(chain.from_iterable(i.x))
            
            n_token = tokenizer(flatten_list,
                                padding = "max_length",
                                max_length = 32,
                                truncation=True,
                                return_tensors = "pt")
            
            n_input_ids = n_token['input_ids'].to(device)
            n_attention_mask = n_token['attention_mask'].to(device)
            
            edge_index = i.edge_index.to(device)
            batch = i.batch.to(device)
            # print(batch.shape)
            y = i.y.to(device)
            
            # Code_Loader
            t_input_ids = j[0].to(device)
            t_attention_mask = j[1].to(device)
            
            out = model(n_input_ids,
                        n_attention_mask,
                        t_input_ids,
                        t_attention_mask,
                        edge_index,
                        batch,
                        prompt_vec)
            
            # print(out,out.argmax(dim=1),y-1)
            losses = criterion(out, y)
                
            total_loss += losses.cpu().item()
                   
            total_acc += (out.argmax(dim=1) == y).sum().cpu().item()
            total += y.shape[0]
            
            step += 1
            
            labels.extend([i.item() for i in y])
            predicts.extend([i.item() for i in torch.argmax(out,1)])
        
        acc = accuracy_score(labels,predicts)
        prec = precision_score(labels,predicts,average='macro')
        recall = recall_score(labels,predicts,average='macro')
        f1 = f1_score(labels,predicts,average='macro')
            
    return total_loss / step, acc* 100., prec* 100., recall* 100., f1* 100.


def detection_collate(batch):
    
    token = tokenizer(batch,
                      padding = "max_length",
                      max_length = 512,
                      truncation=True,
                      return_tensors = "pt")
    
    input_ids=token['input_ids']
    attention_mask=token['attention_mask']

    return input_ids, attention_mask


if __name__ == '__main__':
    begin_time = time.time()
    
    args = load_args()
    
    print('DataSet is CodeXGLUE')
    
    train_tensor = torch.load(args.train_tensor)
    valid_tensor = torch.load(args.valid_tensor)
    test_tensor = torch.load(args.test_tensor)
    
    
    # txt, label, graph = test_tensor
    code_train, _, graph_train = train_tensor
    code_valid, _, graph_valid = valid_tensor
    code_test, _, graph_test = test_tensor
    
    data_scale = args.scale
    
    # 设置每个节点的长度
    node_len = args.node_len
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    
    tokenizer = AutoTokenizer.from_pretrained(args.codebert_path)
    
    # prompt 设置
    init_vac = args.init_vac
    
    prompt_vec, embed_len = init_prompt(init_vac, args.bert_path)
           
    # 划分数据集
    train_len = int(len(code_train)*data_scale)
    valid_len = int(len(code_valid)*data_scale)
    test_len = int(len(code_test)*data_scale)
    
    print('train_len:', train_len)
    print('valid_len:', valid_len)
    print('test_len:', test_len)
    
    # 加载Graph
    train_graph = loader.DataLoader(graph_train[:train_len], batch_size, shuffle=False)
    valid_graph = loader.DataLoader(graph_valid[:valid_len], batch_size, shuffle=False)
    test_graph = loader.DataLoader(graph_test[:test_len], batch_size, shuffle=False)
    
    # 加载Code
    train_code = DataLoader(code_train[:train_len], batch_size, shuffle=False, collate_fn=detection_collate)
    valid_code = DataLoader(code_valid[:valid_len], batch_size, shuffle=False, collate_fn=detection_collate)
    test_code = DataLoader(code_test[:test_len], batch_size, shuffle=False, collate_fn=detection_collate)
    
    # 判断GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 加载模型
    model = MERG(out_channels=embed_len, args=args).to(device)
    
    # 优化器、学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-3)   
    criterion = torch.nn.CrossEntropyLoss()
    
    
    best_acc = 0.0
    
    print('START Training--------------------------------------------')
    epoch_list, eval_acc = [], []
    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train(train_graph, train_code, prompt_vec)
        valid_loss, valid_acc, valid_prec, valid_recall, valid_f1 = test(valid_graph, valid_code, prompt_vec)
        test_loss, test_acc, test_prec, test_recall, test_f1 = test(test_graph, test_code, prompt_vec)
        
        print('Cur_lr: {:.4f}'.format(lr_scheduler.get_last_lr()[0]))
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}, Train Acc: {train_acc:.2f}, Valid Acc: {valid_acc:.2f}')
        print(f'Test Acc: {test_acc:.2f}, Test Prec: {test_prec:.2f}, Test Recall: {test_recall:.2f}, Test F1: {test_f1:.2f}')
        # print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}, Train Acc: {train_acc:.2f}, Valid Acc: {valid_acc:.2f}, Test Acc: {test_acc:.2f}')
        lr_scheduler.step()
        
        epoch_list.append(epoch)
        eval_acc.append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            int_acc = round(best_acc,2)
            saveModel()
    
    df_loss = pd.DataFrame({'epoch':epoch_list, 'acc':eval_acc})
    df_loss.to_csv(f'./estimate/prompt_acc{int_acc}_epoch{num_epochs}.csv', encoding='utf-8')
    
    plt.plot(epoch_list, eval_acc)
    plt.savefig(f'./estimate/prompt_acc{int_acc}_epoch{num_epochs}.png', dpi=300)
    print('best_acc: {:.2f}'.format(best_acc))
    
    run_time = run_time(begin_time)
    print("Run Time: ", time.strftime("%H:%M:%S", time.gmtime(run_time))) 
