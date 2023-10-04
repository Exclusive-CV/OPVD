# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:57:50 2023

@author: 27812
"""

import argparse


def load_args():
    parser = argparse.ArgumentParser()
   
    # load_data
    # parser.add_argument('--train_tensor', type=str, default='data/codexglue_train_tensor.pt')
    # parser.add_argument('--valid_tensor', type=str, default='data/codexglue_valid_tensor.pt')
    # parser.add_argument('--test_tensor', type=str, default='data/codexglue_test_tensor.pt')
    parser.add_argument('--train_tensor', type=str, default='data/train.pt')
    parser.add_argument('--valid_tensor', type=str, default='data/valid.pt')
    parser.add_argument('--test_tensor', type=str, default='data/test.pt')
     
    
    # graph
    parser.add_argument('--node_len', type=int, default=32)
   

    # prompt
    parser.add_argument('--prompt_len', type=int, default=16)
    parser.add_argument('--init_vac', type=bool, default=True)
     
    # Pre-trained
    parser.add_argument('--pretrain', type=bool, default=False)   
    parser.add_argument('--codebert_path', type=str, default='/Users/v/Desktop/code/huggingface/codebert-c')
    parser.add_argument('--bert_path', type=str, default='/Users/v/Desktop/code/huggingface/bert-tiny')
    parser.add_argument('--embed_dim', type=int, default=768)

    # parameter
    parser.add_argument('--scale', type=float, default=0.002) # 加载数据的比例
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=2)
    
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    

    # Network

    
    args = parser.parse_args()

    return args