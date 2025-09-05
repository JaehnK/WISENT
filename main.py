import pandas as pd
import nltk

import argparse

import argparse
from colorama import init, Fore, Style


# 모델 개발 시 shell script 단에서 실행될 메인함수이다. 
# 개발이 완료된 최종 단계에서 작성 예정

if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", help="input text data csv file")
    args = parser.parse_args()
    print(args.text)



