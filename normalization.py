# -*- coding: utf-8 -*-
"""
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
