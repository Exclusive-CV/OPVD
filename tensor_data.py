# -*- coding: utf-8 -*-
"""
@author: 27812
"""

import torch
from torch_geometric.data import Data

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
        number = list(i.keys())
        values = list(i.values()) 
   
        # vec = tokenizer(values, padding = "max_length",
        #                 max_length = 32,
        #                 truncation=True,
        #                 return_tensors = "pt")
        
        index.append(number)
        vectors.append(values)

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
    
        data = Data(x=v, edge_index=edge, y=torch.tensor(l))
        grahp_dataset.append(data)
    return grahp_dataset


def tensordata(path):
    data = pd.read_json(path, encoding='utf-8')
    
    code = data.code.to_list()
    label = data.label.to_list()

    graph = data.graph.to_list()
    edge = data.edge.to_list()

    # token = tokenizer(code,
    #                   padding = "max_length",
    #                   max_length = 512,
    #                   truncation=True,
    #                   return_tensors = "pt")
    
    # dataset_code = token
    
    index, vectors_list = get_func_embedding(graph)
    edge_list = get_edge(edge)

    dataset_graph = build_graph_dataset(index, vectors_list, edge_list, label)

    return code, label, dataset_graph



begin_time = time.time()


data_train_path = 'data/codexglue_train_raw.json'
data_valid_path = 'data/codexglue_valid_raw.json'
data_test_path = 'data/codexglue_test_raw.json'

train_code, train_label, train_graph = tensordata(data_train_path)
print("Finish train data")
valid_code, valid_label, valid_graph = tensordata(data_valid_path)
print("Finish valid data")
test_code, test_label, test_graph = tensordata(data_test_path)
print("Finish test data")

train_data = (train_code, train_label, train_graph)
valid_data = (valid_code, valid_label, valid_graph)
test_data = (test_code, test_label, test_graph)

# torch.save(train_data, './data/codexglue_train_tensor.pt', _use_new_zipfile_serialization=False)
# torch.save(valid_data, './data/codexglue_valid_tensor.pt', _use_new_zipfile_serialization=False)
# torch.save(test_data, './data/codexglue_test_tensor.pt', _use_new_zipfile_serialization=False)

# torch.save(train_data, './data/train.pt', _use_new_zipfile_serialization=False)
# torch.save(valid_data, './data/valid.pt', _use_new_zipfile_serialization=False)
# torch.save(test_data, './data/test.pt', _use_new_zipfile_serialization=False)

print('Finish_Preprocessing')

run_time = run_time(begin_time)
print("Run Time: ", time.strftime("%H:%M:%S", time.gmtime(run_time)))   
