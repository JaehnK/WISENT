import pandas as pd
import nltk
from Preprocessing import preprocess



def main():
    df = pd.read_csv("./kaggle_RC_2019-05.csv")
    text = df.loc[:19999, 'body'].to_list()
    docs = preprocess.Docs()
    docs.rawdata = text
    
if "__main__" == __name__:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt_tab')
    main()