import os
import time
import functools

import pandas as pd
from pprint import pprint

from services import DocumentService, Word2VecService


def timer_with_result(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        print(f"{func.__name__} 실행 시간: {execution_time:.4f}초")
        return result, execution_time
    return wrapper

@timer_with_result
def run_create_sentence_list_with_time(documents, n_processes=None):
    doc_serv = DocumentService()
    return doc_serv.create_sentence_list(documents=documents, n_processes=n_processes)

def preprocess_time_test():
    """시간복잡도 테스트"""
    time_result = {}
    data = pd.read_csv("../kaggle_RC_2019-05.csv")['body']
    
    # 테스트할 n 값들
    n_values = [1000, 5000, 10000, 20000, 30000]
    
    print("=== 전처리 시간 복잡도 테스트 ===")
    for n in n_values:
        print(f"\nn={n:,} 테스트 중...")
        result, exec_time = run_create_sentence_list_with_time(
            data[:n].to_list(), 
            n_processes=os.cpu_count()
        )
        time_result[n] = exec_time
        print(f"n={n:,}: {exec_time:.4f}초 완료")
    
    return time_result

if __name__ == "__main__":
    # print(preprocess_time_test())
    data = pd.read_csv("./kaggle_RC_2019-05.csv").loc[:9999, 'body'].to_list()
    doc_serv = DocumentService()
    doc_serv.create_sentence_list(documents=data, n_processes=None)
    freq = doc_serv.get_top_words(500, True) 
    for word in freq[:100]:
        print(f"word: {word.content}")
        print(f" stopword: {word.is_stopword}")

    w2v = Word2VecService.create_default(doc_serv)
    w2v = w2v.train()
    vec = w2v.get_word_vector("leave")
    print(type(vec) )
    print(vec)