import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'main-service'))

import time

import pandas as pd

from Preprocess import *
from network_creator import *


# main.py 맨 위에 추가
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.cuda.is_available = lambda: False

def main():
    df = pd.read_csv("../kaggle_RC_2019-05.csv")
    text = df['body'][:10000]
    del df
    
    start = time.time()
    docs = Docs()
    docs.rawdata = text
    print(docs)
    end = time.time()
    print(f"🕐 Sentense Preprocessing: {end - start:.5f} sec") 
    
    print("Starting Graph")
    start = time.time()
    graph = graph_entity(docs)
    graph.create_co_occurrence_edges()
    end = time.time()
    print(f"🕐 Creating Graph: {end - start:.5f} sec") 
    graph.visualize(save_path='.')
    
main()