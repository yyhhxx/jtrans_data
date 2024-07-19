# 导入所需的库和模块
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import AdamW
import torch.nn.functional as F
import argparse
import wandb
import logging
import sys
import time
import data
from datetime import datetime
import pickle
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from data import help_tokenize, load_paired_data,FunctionDataset_CL
from datautils.playdata import DatasetBase as DatasetBase
from tokenizer import *

# 设置WANDB标志为True，表示使用Weights & Biases进行实验跟踪
WANDB = True

# 定义一个函数来获取日志记录器
def get_logger(name):
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(message)s',filename=name)
    logger = logging.getLogger(__name__)
    # 创建一个流处理器，将日志信息输出到控制台
    s_handle = logging.StreamHandler(sys.stdout)

    s_handle.setLevel(logging.INFO)
    # 设置流处理器的日志格式
    s_handle.setFormatter(logging.Formatter("%(asctime)s - %(filename)s[:%(lineno)d] - %(message)s"))
    # 将流处理器添加到日志记录器中
    logger.addHandler(s_handle)

    return logger


class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        # 将position_embeddings设置为word_embeddings，这是二进制BERT模型的一个特点
        # 表示position 不被单独编码 而是和word embedding 共享
        self.embeddings.position_embeddings=self.embeddings.word_embeddings


def gen_funcstr(f,convert_jump):
    cfg=f[3]
    #print(hex(f[0]))
    bb_ls,code_lst,map_id=[],[],{}
    for bb in cfg.nodes:
        bb_ls.append(bb)
    bb_ls.sort()
    for bx in range(len(bb_ls)):
        bb=bb_ls[bx]
        asm=cfg.nodes[bb]['asm']
        map_id[bb]=len(code_lst)
        for code in asm:
            operator,operand1,operand2,operand3,annotation=readidadata.parse_asm(code)
            code_lst.append(operator)
            if operand1!=None:
                code_lst.append(operand1)
            if operand2!=None:
                code_lst.append(operand2)
            if operand3!=None:
                code_lst.append(operand3)
    for c in range(len(code_lst)):
        op=code_lst[c]
        if op.startswith('hex_'):
            jumpaddr=int(op[4:],base=16)
            if map_id.get(jumpaddr):
                jumpid=map_id[jumpaddr]
                if jumpid < MAXLEN:
                    code_lst[c]='JUMP_ADDR_{}'.format(jumpid)
                else:
                    code_lst[c]='JUMP_ADDR_EXCEEDED'
            else:
                code_lst[c]='UNK_JUMP_ADDR'
            if not convert_jump:
                code_lst[c]='CONST'
    func_str=' '.join(code_lst)
    return func_str

# 加载pkl文件，读取函数名 注释 和特征
def load_unpair_data(file_path):
    functions=[]
    func_emb_data=[]
    sum = 0
    with open(file_path,'rb') as f:
        data = pickle.load(f)
    for func_name, func_data in data.items():
        functions.append([])
        # 加上了 funcname
        func_emb_data.append({'funcname':func_name})
        
        # Todo: 加上func_cmt 
        func_addr, asm_list, rawbytes_list, cfg, biai_featrue = func_data
        
        func_list = [func_addr, asm_list, rawbytes_list, cfg, biai_featrue]
        func_str=gen_funcstr(func_list,True)
        if len(func_str)>0:
            func_emb_data[-1]['ebds']=len(functions[-1])
            # 加上注释的内容
            # func_emb_data[-1]['func_cmt']=func_cmt
            functions[-1].append(func_str)
            sum+=1
    print('TOTAL ',sum)
    return functions,func_emb_data 


class FunctionDataset(torch.utils.data.Dataset):
    def __init__(self,file_path,tokenizer):
        functions,ebds = load_unpair_data(file_path)  # todo: 加上func_target
        self.datas=functions
        self.ebds=ebds
        self.tokenizer=tokenizer        

    def __len__(self):
        return len(self.datas)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="jtrans generate embedding to match")
    parser.add_argument("--model_path", type=str, default="./models/jTrans-finetune",help="model path")
    parser.add_argument("--data_path",type=str, default="./datautils/UE/mc/tezheng", help="data to generate embd")
    parser.add_argument("--output_path",type=str, default="./datautils/UE/mc/ebds", help="result file path")
    parser.add_argument("--tokenizer_path", default="./jtrans_tokenizer")
    
    args = parser.parse_args()

    # 加载日志文件
    now = datetime.now()
    tim = now.strftime("%Y-%m-%d %H:%M:%S")
    logger = get_logger(f"log_{tim}")

    logger.info(f"Loading pretrained and finetuned model from {args.model_path} ...")
    model = BinBertModel.from_pretrained(args.model_path)
    # 使用模型进行预测，评估模式
    model.eval()
    logger.info("Loading model done and evaluate ready")

    # 加载分词器
    logger.info(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    logger.info("Tokenizer ready")

    # 加载数据集
    for root,dirs,files in os.walk(args.data_path):
        for file in files:
            if file == 'saved_index.pkl':
                continue
            file_path = os.path.join(root,file)
            logger.info(f"Loading data from {file_path}...")

            ft_valid_dateset = FunctionDataset(file_path,tokenizer)

            for i in tqdm(range(len(ft_valid_dateset.datas))):
                func_str = ft_valid_dateset.datas[i]
                idx = ft_valid_dateset.ebds[i]
                ret1=tokenizer(func_str, add_special_tokens=True,max_length=512,padding='max_length',truncation=True,return_tensors='pt') #tokenize them
                seq1=ret1['input_ids']
                mask1=ret1['attention_mask']
                output=model(input_ids=seq1,attention_mask=mask1)
                anchor=output.pooler_output
                

                ft_valid_dateset.ebds[i]['ebds']=anchor.detach().cpu()
            


            logger.info(f"starts writing {file} embeddings")
            result_path = os.path.join(args.output_path,file)
            fi=open(result_path,'wb')
            # 只保存了embds？
            pickle.dump(ft_valid_dateset.ebds,fi)
            fi.close()
