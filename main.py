import pandas as pd
import nltk
from Preprocess import Docs
import argparse

import argparse
from colorama import init, Fore, Style

if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", help="input text data csv file")
    args = parser.parse_args()
    print(args.text)



