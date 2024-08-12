# -*- coding: utf-8 -*-
"""
@author: 27812
"""

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel


def init_prompt(init_vac, bert_path):
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    
    if init_vac:
        # 设置提示词
        prompt_txt  = 'this c langauge function is {}.'
        prompt_cls = ['good', 'bug-like']
        prompts = [prompt_txt.format(name) for name in prompt_cls]
        # prompts = [prompt_prefix + " " + name + " " + prompt_suffix + "." for name in cls_prompt]
        
        print('init prompt vac:', prompts[0])

        prompt_ctx = tokenizer(prompt_txt,
                               padding="max_length",
                               max_length=12,
                               truncation=True,                 
                               return_tensors = "pt")
        
        prompt_cls = tokenizer(prompt_cls,
                               padding="max_length",
                               max_length=4,
                               truncation=True,                 
                               return_tensors = "pt")
        
        embed_len = len(prompt_ctx[0]) + len(prompt_cls[0])

        print('init prompt token len:', embed_len)
        
        prompt_vec = (prompt_ctx, prompt_cls)
        
    else:
        prompt_prefix  = torch.empty(3)
        nn.init.normal_(prompt_prefix, std=0.02)
        
        prompt_suffix = torch.empty(2)
        nn.init.normal_(prompt_suffix, std=0.02)
        
        cls_prompt = ['good', 'leakly']
        prompt_txt = tokenizer(cls_prompt,
                               padding="max_length",
                               max_length=3,
                               truncation=True,                               
                               return_tensors = "pt")['input_ids']
        # print(prompt_txt)
        prompts = [torch.cat((name[:1], prompt_prefix, name[1:-1], prompt_suffix, name[-1:]),dim=0) for name in prompt_txt]
        embed_len = len(prompts[0])
        print('without init prompt len:', embed_len)
        prompt_vec = torch.stack(prompts, dim=0)
    
    return prompt_vec, embed_len


if __name__ == '__main__':
    init_vac = True
    bert_path = r'C:\Users\27812\.cache\torch\sentence_transformers\bert-base-uncased'
    prompt_vec, embed_len = init_prompt(init_vac, bert_path)
    
    bert = AutoModel.from_pretrained(bert_path)

    embed_prefix = bert(**prompt_vec[0]).pooler_output
    embed_cls = bert(**prompt_vec[1]).pooler_output
    
    cls_lst = []
    for c in embed_cls:
        n_cls= torch.cat((embed_prefix.squeeze(dim=0),c), dim=0)
        cls_lst.append(n_cls)
        
    a = torch.stack(cls_lst)
        
        
        
        
        
        
