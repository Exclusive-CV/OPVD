# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:48:14 2023

@author: 27812
"""

import torch
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from transformers import AutoTokenizer

import pandas as pd
import numpy as np
import time
# import collections
# import matplotlib.pyplot as plt

def run_time(begin_time):
    end_time = time.time()
    total_time = end_time - begin_time
    return total_time


def get_func_embedding(graph):         
    index = []
    vectors = []
    
    for i in graph:    
        values = list(i.values()) 
        number = list(i.keys())
   
        vec = tokenizer(values, padding = "max_length",
                        max_length = 32,
                        truncation=True,
                        return_tensors = "pt")
        
        vec = vec['input_ids'].tolist()
        vec = pd.DataFrame(vec)
        
        index.append(number)
        vectors.append(vec)

    return index, vectors


def get_edge(edge):
    edge_list=[]
    for row in edge:
        keys = list(row.keys()) 
        values = list(row.values()) 
        edges = []
        for i, j in zip(keys, values):
            #没有加边的属性j
            i_split = i.split("'")
            # print(i_split)
            eg = i_split[1], i_split[3]
            edges.append(eg)
        edge_list.append(edges)
    return edge_list


def build_graph_dataset(index, vectors_list, edge_list, label):
    grahp_dataset=[]
    for i, v, e, l in zip(index, vectors_list, edge_list, label):
        # build graph
        idx = np.array(i)
        idx_map = {j: i for i, j in enumerate(idx)}   #构造节点的字典
        edges_unordered = np.array(e)    #导入edge的数据
        #生成图的边，（x,y）其中x、y都是为以编号为索引得到的值
        edge = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                      dtype=np.int32).reshape(edges_unordered.shape)
        # 行列互换
        edge = edge.transpose()
        edge = torch.LongTensor(edge)
        
        feature = v
        feature = torch.FloatTensor(np.array(feature.values))
    
        data = Data(x=feature, edge_index=edge, y=torch.tensor(l))
        grahp_dataset.append(data)
    return grahp_dataset


def tensordata(path):
    data = pd.read_json(path, encoding='utf-8')
    
    code = data.code.to_list()
    label = data.label.to_list()

    graph = data.graph.to_list()
    edge = data.edge.to_list()

    token = tokenizer(code,
                      padding = "max_length",
                      max_length = 1024,
                      truncation=True,
                      return_tensors = "pt")

    X_train = torch.as_tensor(token["input_ids"], dtype=torch.float)
    dataset_code = TensorDataset(X_train, torch.tensor(label))
    
    index, vectors_list = get_func_embedding(graph)
    edge_list = get_edge(edge)

    dataset_graph = build_graph_dataset(index, vectors_list, edge_list, label)

    return dataset_code, dataset_graph



begin_time = time.time()


data_train_path = 'data/codexglue_train_data.json'
data_valid_path = 'data/codexglue_test_data.json'
data_test_path = 'data/codexglue_valid_data.json'

pretrained_path = r'C:\Users\27812\.cache\torch\sentence_transformers\neulab_codebert-c'
tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

train_code, train_graph = tensordata(data_train_path)
valid_code, valid_graph = tensordata(data_valid_path)
test_code, test_graph = tensordata(data_test_path)


train_data = (train_code, train_graph)
valid_data = (valid_code, valid_graph)
test_data = (test_code, test_graph)

torch.save(train_data, './data/codexglue_train_tensor_1024.pt', _use_new_zipfile_serialization=False)
torch.save(valid_data, './data/codexglue_valid_tensor_1024.pt', _use_new_zipfile_serialization=False)
torch.save(test_data, './data/codexglue_test_tensor_1024.pt', _use_new_zipfile_serialization=False)


print('Finish_Preprocessing')

run_time = run_time(begin_time)
print("Run Time: ", time.strftime("%H:%M:%S", time.gmtime(run_time)))   

# =============================================================================
# 查看列表中所有字符串的长度分布
# import collections
# import matplotlib.pyplot as plt
# lengths = [len(string) for string in code]
# counter = collections.Counter(lengths)
# 
# # 绘制直方图
# plt.bar(counter.keys(), counter.values())
# plt.xlabel('Length of string')
# plt.ylabel('Frequency')
# plt.show()
# =============================================================================
