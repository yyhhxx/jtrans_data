import logging
from datetime import datetime
import pickle
import sys
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
from torch.nn.functional import cosine_similarity
import random

from datautils.playdata import DatasetBase as DatasetBase

def get_logger(name):
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(message)s',filename=name)
    logger = logging.getLogger(__name__)
    s_handle = logging.StreamHandler(sys.stdout)

    s_handle.setLevel(logging.INFO)
    s_handle.setFormatter(logging.Formatter("%(asctime)s - %(filename)s[:%(lineno)d] - %(message)s"))
    logger.addHandler(s_handle)

    return logger


class FunctionDataset_Fast(torch.utils.data.Dataset): 
    def __init__(self,arr1,arr2): 
        self.arr1=arr1
        self.arr2=arr2
        assert(len(arr1)==len(arr2))
    def __getitem__(self, idx):            
        return self.arr1[idx].squeeze(0),self.arr2[idx].squeeze(0)
    def __len__(self):
        return len(self.arr1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="match evaluate")
    parser.add_argument("--source_file", type=str, default="./datautils/UE/smalltest/unsame/extractEmbd/libUnreal-deve.so-deve-1_extract.pkl",help="souce function ebds")
    # parser.add_argument("--source_file", type=str, default="./datautils/UE/smalltest/unsame/extractEmbd/unsame-deve.so-deve-2_extract.pkl",help="souce function ebds")
    parser.add_argument("--target_file",type=str, default="./datautils/UE/smalltest/unsame/extractEmbd/unsame-debug.so-debug-1_extract.pkl", help="file to match")
    
    args = parser.parse_args()
    POOLSIZE=32

    # 加载日志文件
    now = datetime.now()
    tim = now.strftime("%Y-%m-%d %H:%M:%S")
    logger = get_logger(f"log_funcname_{tim}")

    funcarr1=[]
    funcarr2=[]
    funcname=[]
    matchname=[]

    with open(args.source_file,'rb') as f:
        source = pickle.load(f)

        for _, dict in enumerate(source):
            # func_name, ebds
            funcname.append(dict['funcname'])
            funcarr1.append(dict['ebds']) #/ebds.norm())
        source_list = list(zip(funcname,funcarr1))
        
    with open(args.target_file,'rb') as f:
        target = pickle.load(f)
        
        for _, dict in enumerate(target):
            matchname.append('sub'+str(dict['funcname']))
            funcarr2.append(dict['ebds']) #/ebds.norm()
        target_list = list(zip(matchname,funcarr2))
    
    # same = 0
    # for i in range(len(target_list)):
    #     if matchname[i] == funcname[i]:
    #         same += 1 
        
    # print(f"before: all {len(funcarr2)} functions {same} have matched")

    same = 0

    for i in range(len(target_list)):
        funcpool = random.sample(source_list, 32)
        # 加入匹配的函数
        funcpool[0] = source_list[i]
        funcebds = target_list[i][1]

        max_similarity = 0.99
        similarity = []
        for j in range(32):
            similarity.append(cosine_similarity(funcpool[j][1], funcebds).item())
            # print(funcpool[j][1] )
            # print(funcebds)
            # exit()

        # print(max(similarity))
        if max(similarity) > max_similarity:
            matchname[i] = funcpool[similarity.index(max(similarity))][0]

            if matchname[i] == funcname[i]:
                same += 1 
        
    print(f"all {len(funcarr2)} functions {same} have matched")
