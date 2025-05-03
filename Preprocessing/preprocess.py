import spacy
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from typing import Optional, Union, List
import numpy as np
import numpy.typing as npt
import sys

class Word:
    TensorType = Union[np.ndarray, 'np.typing.NDArray[np.float32]']
    
    def __init__(self):
        self.content: Optional[str] = None
        self.idx: Optional[int] = None
        self.w2v_emb: Optional[Word.TensorType] = None  # Word2Vec 임베딩
        self.dbert_emb: Optional[Word.TensorType] = None  # DistilBERT 임베딩
        self.freq: Optional[int] = None
        self.isnode: Optional[bool] = None
        self.attention_emb: Optional[Word.TensorType] = None  # 어텐션 임베딩
        self.concat_emb: Optional[Word.TensorType] = None  # 연결된 임베딩
    
    def __str__(self):
        return self.content
    
class Sentence:
    def __init__(self):
        self._raw: Optional[str] = None
        self._lemmatised: Optional[List[str]] = None
        self._word_idices: Optional[List[int]] = None

class Docs:
    def __init__(self):
        self._rawdata: Optional[List[str]] = None
        self._words_list: Optional[list[Word]] = None
        self._sentence_list: Optional[list[Sentence]] = None
    
    @property
    def rawdata(self):
        return self._rawdata
    
    @rawdata.setter
    def rawdata(self, docs: List[str]):
        if len(docs) < 10:
            raise ValueError("Number of documents is less than 10. Not sufficient for clustering.")
        elif len(docs) < 500:
            print("Warning: Document count is low ({}). Clustering results may have reduced accuracy.".format(len(docs)), file=sys.stderr)
            self._rawdata = docs
        else:
            print("Processing {} documents".format(len(docs)), file=sys.stderr)
            self._rawdata = docs
    
    @property
    def words_list(self):
        return self._words_list
    
    
    def __str__(self):
        if self._rawdata is None:
            return ""
        return f"raw_data: {len(self._rawdata)}\nwords_list: {len(self._words_list)}\nsentence_list: {len(self._sentence_list)}"
    
    def __getitem__(self, idx):
        return self._rawdata[idx]

if "__main__" == __name__:
    docs = Docs()
    