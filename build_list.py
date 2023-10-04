# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:58:59 2023

@author: 27812
"""

import os
import numpy as np

# 读取文件夹下所有指定后缀的文件名,并把他们用列表存起来
def data_dic(path):
    ls =['vul', 'no-vul']
    data_dic={}
    for i in ls:
        data_path = path+'/'+i
        names = os.listdir(data_path)
        filename_ls=[]
        for fn in names:
            # 从后往前分开文件名和后缀
            filename, extension = os.path.splitext(fn)
            if extension == '.dot':
                filename_ls.append(filename)
        data_dic[i] = filename_ls
    return data_dic

train_dic = data_dic("./data/cpgs/train")
test_dic = data_dic("./data/cpgs/test")
valid_dic = data_dic("./data/cpgs/valid")

data_dic_list = {'train': train_dic, 'valid': valid_dic, 'test': test_dic}

# # Save
# np.save('data_dic_list.npy', data_dic_list) # 注意带上后缀名
3-build_list.py
