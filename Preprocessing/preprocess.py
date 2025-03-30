import spacy
from nltk.stem import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class WordTokenizer:
    def __init__ (self, raw_documents:list[str]):
        """
        
        의미연결망에 사용되는 단어 리스트를 선정
        신경망 기반의 임베딩 모델을 통해 단어 임베딩을 추출함
        
        Args:
            raw_documents (list[str]): Raw Documents List 
        """
        
        if raw_documents is None:
            raise ValueError("Raw_documents cannot be None")
        
        self.raw_documents = raw_documents
        self.lemmatised_documents = None
        self.word_nodes = None

    @property
    def raw_documents(self):
        return self.raw_documents

    @property
    def lemmatised_documents(self):
        if self.lemmatised_documents is None:
            raise ValueError("Lemmatised_documents is None")
        return self.lemmatised_documents

    @lemmatised_documents.setter
    def lemmatised_documents(self, documents: list[str]):
        self.lemmatised_documents = documents

    def get_word_list(self, count:str = "freq", n:int = 500):
        """
        상위 단어 n개를 선택합니다.
        선택 기준은 count 인자를 통해 지정할 수 있습니다(기본값: freq).
        선택할 단어 개수는 n 인자를 통해 지정할 수 있습니다(기본값: 500).
    
        Args:
            count (str, optional): 단어 선택 기준. "freq" 또는 "tf-idf" 선택 가능. 기본값은 "freq".
            n (int, optional): 선택할 상위 단어의 개수. 기본값은 500.
            
        Returns:
            list: 선택된 상위 n개 단어 목록
        """
        
        if self.lemmatised_documents is None:
            raise ValueError("Lemmtise Documents First")
        if count == "freq":
            vectorizer = CountVectorizer() 
        elif count == "tf-idf":
            vectorizer = TfidfVectorizer()
        else:
            raise ValueError("count must be either 'freq' or 'tf-idf'")
        
        X = vectorizer.fit_transform(self.lemmatised_documents)
        feature_names = vectorizer.get_feature_names_out()
        
        if count == "freq":
            word_counts = X.sum(axis=0).A1
        else:  # tf-idf
            word_counts = X.mean(axis=0).A1
        
        top_indices = word_counts.argsort()[-n:][::-1]
        self.word_nodes = [feature_names[i] for i in top_indices]

    @property
    def word_nodes(self):
        return self.word_nodes

if "__main__" == __name__:
    print("Hello World")