# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:21:08 2023

@author: 27812
"""

# create dataset list: train, test, valid

import numpy as np
import pandas as pd
import time

import pygraphviz as pgv
import networkx as nx

from sklearn.utils import shuffle


def run_time(begin_time):
    end_time = time.time()
    total_time = end_time - begin_time
    return total_time

def load_code(path, folderlist, label):  
    codes=[]
    labels=[]
    for folder in folderlist:
        filepath = path + "//" + folder + ".c"
        # print(filepath)
        with open(filepath, "r") as file:
            code = file.read()
        file.close()
        code = code.replace("\n", "").split()
        code = ' '.join(code)
        label = int(label)
        codes.append(code)
        labels.append(label)
    return codes, labels


def load_graph(path, file_name):
    labels_code = []
    labels_edge = []

    for line in file_name:
        # print(name)
        Gtmp = pgv.AGraph(path+ "//" + '%s.dot'%line)
        # print(line)
        G = nx.Graph(Gtmp)
        #获取全部节点的value
        node_dict = nx.get_node_attributes(G,'label')
        label_code = dict()
        for label, all_code in node_dict.items():
            # print(all_code)
            # 逆序找到最后一个“）”分割
            code = all_code[1:-1].rsplit(')', 1)[0]
            split = code.split(",", 1)
            attr = split[0]
            if attr == 'BLOCK':
                func = attr
            else:
                func = split[1]
            # print(func)
            label_code[label] = func
        labels_code.append(label_code)
        
        # 获取全部边的value
        edge_dict =  nx.get_edge_attributes(G,'label')
        # print(edge_dict)
        label_edge = dict()
        for key, value in edge_dict.items():
            # print(value)
            edge = value[0:3]
            # print(edge)
            label_edge[key] = edge
        labels_edge.append(label_edge)
    return labels_code, labels_edge


def subset_code(data_list, codes_path, txt_name):
    for i in data_list:
        if i == 'no-vul':
            no_vul_address = codes_path + i
            no_vul_file_list = txt_name[i]
            no_vul_code, no_vul_label = load_code(no_vul_address, no_vul_file_list, '0')
        if i == 'vul':
            vul_address = codes_path + i
            vul_file_list = txt_name[i]
            vul_code, vul_label = load_code(vul_address, vul_file_list, '1')
            
    data_no_vul = pd.DataFrame({'code':no_vul_code, 'label':no_vul_label})
    data_vul = pd.DataFrame({'code':vul_code, 'label':vul_label})
    data = pd.concat([data_no_vul, data_vul], axis=0).reset_index(drop=True)
    return data


def sub_graph(data_list, graph_path, txt_name):
    for i in data_list:
        if i == 'no-vul':
            no_vul_address = graph_path + i
            no_vul_graph_list = txt_name[i]
            no_vul_graph, no_vul_edge = load_graph(no_vul_address, no_vul_graph_list)
        if i == 'vul':
            vul_address = graph_path + i
            vul_graph_list = txt_name[i]
            vul_graph, vul_edge = load_graph(vul_address, vul_graph_list)
            
    data_no_vul = pd.DataFrame({'graph':no_vul_graph, 'edge':no_vul_edge})
    data_vul = pd.DataFrame({'graph':vul_graph, 'edge':vul_edge})
    data = pd.concat([data_no_vul, data_vul], axis=0).reset_index(drop=True)
    return data


begin_time = time.time() 

# Load
load_dict = np.load('/root/CodeXGLUE/data/new_data_dic_list.npy', allow_pickle=True).item()

# list
train_list = load_dict['train']

valid_list = load_dict['valid']

test_list = load_dict['test']

data_list = ['no-vul', 'vul']

# path_txt
train_path_txt = "/root/CodeXGLUE/data/txt/train/"

valid_path_txt = "/root/CodeXGLUE/data/txt/valid/"

test_path_txt = "/root/CodeXGLUE/data/txt/test/"

# path_graph
train_path_graph = "/root/CodeXGLUE/data/cpgs/train/"

valid_path_graph = "/root/CodeXGLUE/data/cpgs/valid/"

test_path_graph = "/root/CodeXGLUE/data/cpgs/test/"

# load_txt
print('Loading TXT')
train_txt = subset_code(data_list, train_path_txt, train_list)
valid_txt = subset_code(data_list, valid_path_txt, valid_list)
test_txt = subset_code(data_list, test_path_txt, test_list)
print('Code Finish')

# loda graph
print('Loading Graph')
train_graph = sub_graph(data_list, train_path_graph, train_list)
valid_graph = sub_graph(data_list, valid_path_graph, valid_list)
test_graph = sub_graph(data_list, test_path_graph, test_list)
print('Graph Finish')

train_data = pd.concat([train_txt, train_graph], axis=1).reset_index(drop=True)
valid_data = pd.concat([valid_txt, valid_graph], axis=1).reset_index(drop=True)
test_data = pd.concat([test_txt, test_graph], axis=1).reset_index(drop=True)

train_data = shuffle(train_data)
valid_data = shuffle(valid_data)
test_data = shuffle(test_data)


train_data.to_json('/root/CodeXGLUE/data/codexglue_train_data.json', force_ascii=False)
valid_data.to_json('/root/CodeXGLUE/data/codexglue_valid_data.json', force_ascii=False)
test_data.to_json('/root/CodeXGLUE/data/codexglue_test_data.json', force_ascii=False)


run_time = run_time(begin_time)
print("Run Time: ", time.strftime("%H:%M:%S", time.gmtime(run_time)))   
