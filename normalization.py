# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:29:08 2023

@author: 27812
"""
import os
import re

def pro_one_file(filepath):
    with open(filepath, "r") as file:
        code = file.readlines()
    file.close()
    new_lines = []
    for i in code:
        b = i.strip('\n')
        if b != "":
            new_text = re.sub(r"\s+", " ", b.strip())
            new_text = " " * (len(b) - len(b.lstrip())) + new_text
            new_lines.append(new_text)
    return new_lines

# 删除多余的换行符和字符串之间的空格
path = "./dataset/codes/train"
setfolderlist = os.listdir(path)
for setfolder in setfolderlist:
    catefolderlist = os.listdir(path + "//" + setfolder)
    #print(catefolderlist)
    for catefolder in catefolderlist:
        filepath = path + "//" + setfolder + "//" + catefolder
        a = pro_one_file(filepath)
        with open(filepath, "w") as file:
            file.writelines(line + '\n' for line in a)
        file.close()
