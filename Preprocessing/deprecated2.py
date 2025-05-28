# import spacy
# import fasttext
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag
# from nltk.corpus import wordnet
# import re
# from sklearn.feature_extraction.text import CountVectorizer
# from typing import Optional, Union, List
# import numpy as np
# import numpy.typing as npt
# import sys

# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# import threading

# class Word:
#     TensorType = Union[np.ndarray, 'np.typing.NDArray[np.float32]']
    
#     def __init__(self):
#         self._content: Optional[str] = None
#         self._idx: Optional[int] = None
#         self._w2v_emb: Optional[Word.TensorType] = None  # Word2Vec 임베딩
#         self._dbert_emb: Optional[Word.TensorType] = None  # DistilBERT 임베딩
#         self._freq: Optional[int] = None
#         self._isnode: Optional[bool] = None
#         self._attention_emb: Optional[Word.TensorType] = None  # 어텐션 임베딩
#         self._concat_emb: Optional[Word.TensorType] = None  # 연결된 임베딩
    
#     def __str__(self):
#         return self.content
    
#     @property
#     def content(self):
#         return self.content
    
#     @content.setter
#     def content(self, word:str):
#         self._content = word

    
    
    
# class Sentence:
#     def __init__(self):
#         self._raw: Optional[str] = None
#         self._lemmatised: Optional[List[str]] = None
#         self._word_idices: Optional[List[int]] = None
#         self._lemmatizer = WordNetLemmatizer()
        
#     @property
#     def raw(self):
#         return self._raw

#     @raw.setter
#     def raw(self, sentence:str):
#         self._raw = sentence
#         self.lemmatise()
        
#     def _get_wordnet_pos(self, word_pos):
#         """POS 태그를 WordNet 형식으로 변환"""
#         tag = word_pos[0].upper()
#         tag_dict = {
#             "J": wordnet.ADJ,
#             "N": wordnet.NOUN,
#             "V": wordnet.VERB,
#             "R": wordnet.ADV
#         }
#         return tag_dict.get(tag, wordnet.NOUN)
    
#     def lemmatise(self):
#         """문장을 표제어 형태로 변환"""
#         if self._raw is None:
#             raise ValueError("Raw sentence is not set. Please set the raw property first.")
        
#         if self._lemmatised is not None:
#             return self._lemmatised
        
#         # 텍스트 전처리: 소문자 변환 및 특수문자 제거
#         cleaned_text = re.sub(r'[^\w\s]', '', self._raw.lower())
        
#         # 토큰화
#         tokens = word_tokenize(cleaned_text)
        
#         # POS 태깅
#         pos_tags = pos_tag(tokens)
        
#         # 표제어 변환 (모든 단어 유지)
#         lemmatised_words = []
#         for word, pos in pos_tags:
#             wordnet_pos = self._get_wordnet_pos(pos)
#             lemma = self._lemmatizer.lemmatize(word, wordnet_pos)
#             lemmatised_words.append(lemma)
        
#         self._lemmatised = lemmatised_words
#         return self._lemmatised
        

# class Docs:
#     def __init__(self):
#         self._rawdata: Optional[List[str]] = None
#         self._words_list: Optional[list[Word]] = None
#         self._sentence_list: Optional[list[Sentence]] = None
#         self._nlp: Optional[fasttext.FastText._FastText] = None
#         self._lock = threading.Lock()
    
#     @property
#     def rawdata(self):
#         return self._rawdata
    
#     @rawdata.setter
#     def rawdata(self, docs: List[str]):
#         if len(docs) < 10:
#             raise ValueError("Number of documents is less than 10. Not sufficient for clustering.")
#         elif len(docs) < 500:
#             print("Warning: Document count is low ({}). Clustering results may have reduced accuracy.".format(len(docs)), file=sys.stderr)
#             self._rawdata = docs
#         else:
#             print("Processing {} documents".format(len(docs)), file=sys.stderr)
#             self._rawdata = docs
            
#         self.create_sentence_list()
    
#     def _create_sentence_with_log(self, doc_text: str, index: int) -> Sentence:
#         """단일 Sentence 객체 생성 및 로그"""
#         sentence = Sentence()
#         sentence.raw = doc_text
        
#         # 스레드 안전한 로그 출력
#         thread_id = threading.current_thread().name
#         with self._lock:
#             if index % 5000 == 0:  # 5000개마다 로그
#                 print(f"Read document {index + 1}/{len(self._rawdata)}", file=sys.stderr)
        
#         return sentence

#     def create_sentence_list(self, max_workers: int = 4) -> None:
#         """rawdata로부터 Sentence 객체 리스트 생성 (멀티스레딩)"""
#         if self._rawdata is None:
#             self._sentence_list = None
#             return

#         print(f"Creating {len(self._rawdata)} sentences using {max_workers} threads...", file=sys.stderr)
        
#         # 멀티스레딩으로 Sentence 객체 생성
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # 각 문서에 대해 작업 제출
#             future_to_index = {
#                 executor.submit(self._create_sentence_with_log, doc_text, i): i 
#                 for i, doc_text in enumerate(self._rawdata)
#             }
            
#             # 결과를 순서대로 저장할 리스트
#             sentences = [None] * len(self._rawdata)
            
#             # 완료된 작업들 처리
#             for future in as_completed(future_to_index):
#                 index = future_to_index[future]
#                 try:
#                     sentence = future.result()
#                     sentences[index] = sentence
#                 except Exception as exc:
#                     print(f'Document {index} generated an exception: {exc}', file=sys.stderr)
        
#         self._sentence_list = sentences

#     @property
#     def words_list(self):
#         return self._words_list
    
    
#     @property
#     def sentence_list(self):
#         return self._sentence_list
    
    
#     def __str__(self):
#         if self._rawdata is None:
#             return ""
#         return f"raw_data: {len(self._rawdata)}\nwords_list: {len(self._words_list)}\nsentence_list: {len(self._sentence_list)}"
    
#     def __getitem__(self, idx):
#         return self._rawdata[idx]

# if "__main__" == __name__:
#     docs = Docs()
    
    
import spacy
import fasttext
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import re
from sklearn.feature_extraction.text import CountVectorizer
from typing import Optional, Union, List
import numpy as np
import numpy.typing as npt
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

class Word:
    TensorType = Union[np.ndarray, 'np.typing.NDArray[np.float32]']
    
    def __init__(self, content: str = None):
        self._content: Optional[str] = content
        self._idx: Optional[int] = None
        self._w2v_emb: Optional[Word.TensorType] = None  # Word2Vec 임베딩
        self._dbert_emb: Optional[Word.TensorType] = None  # DistilBERT 임베딩
        self._freq: int = 0  # 빈도수는 0으로 초기화
        self._isnode: Optional[bool] = None
        self._attention_emb: Optional[Word.TensorType] = None  # 어텐션 임베딩
        self._concat_emb: Optional[Word.TensorType] = None  # 연결된 임베딩
    
    def __str__(self):
        return self._content or ""
    
    @property
    def content(self):
        return self._content
    
    @content.setter
    def content(self, word: str):
        self._content = word
    
    @property
    def idx(self):
        return self._idx
    
    @idx.setter
    def idx(self, value: int):
        self._idx = value
    
    @property
    def freq(self):
        return self._freq
    
    def increment_freq(self):
        """빈도수 증가"""
        self._freq += 1

class TrieNode:
    """접두사 트리 노드"""
    def __init__(self):
        self.children = {}
        self.word_obj: Optional[Word] = None
        self.is_end_of_word = False

class WordTrie:
    """접두사 트리를 이용한 Word 객체 관리"""
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0  # 고유 단어 개수 (idx 할당용)
    
    def insert_or_get_word(self, word_content: str) -> Word:
        """단어를 트리에 삽입하거나 기존 Word 객체 반환"""
        node = self.root
        
        # 트리 탐색
        for char in word_content:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        # 단어 끝 지점에서 Word 객체 처리
        if node.is_end_of_word:
            # 기존 단어 - 빈도수만 증가
            node.word_obj.increment_freq()
            return node.word_obj
        else:
            # 새 단어 - Word 객체 생성 및 설정
            word_obj = Word(word_content)
            word_obj.idx = self.word_count
            word_obj.increment_freq()
            
            node.word_obj = word_obj
            node.is_end_of_word = True
            self.word_count += 1
            
            return word_obj
    
    def get_all_words(self) -> List[Word]:
        """모든 Word 객체 반환"""
        words = []
        self._dfs_collect_words(self.root, words)
        return words
    
    def _dfs_collect_words(self, node: TrieNode, words: List[Word]):
        """DFS로 모든 Word 객체 수집"""
        if node.is_end_of_word:
            words.append(node.word_obj)
        
        for child in node.children.values():
            self._dfs_collect_words(child, words)

class Sentence:
    def __init__(self, docs_ref=None):
        self._raw: Optional[str] = None
        self._lemmatised: Optional[List[str]] = None
        self._word_indices: Optional[List[int]] = None
        self._word_objects: Optional[List[Word]] = None
        self._lemmatizer = WordNetLemmatizer()
        self._docs_ref = docs_ref  # Docs 객체 참조
        
    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, sentence: str):
        self._raw = sentence
        self._lemmatised = None
        self._word_indices = None
        self._word_objects = None
        self.lemmatise()
        
    def _get_wordnet_pos(self, word_pos):
        """POS 태그를 WordNet 형식으로 변환"""
        tag = word_pos[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def lemmatise(self):
        """문장을 표제어 형태로 변환하고 Word 객체 생성"""
        if self._raw is None:
            raise ValueError("Raw sentence is not set. Please set the raw property first.")
        
        if self._lemmatised is not None:
            return self._lemmatised
        
        # 텍스트 전처리: 소문자 변환 및 특수문자 제거
        cleaned_text = re.sub(r'[^\w\s]', '', self._raw.lower())
        
        # 토큰화
        tokens = word_tokenize(cleaned_text)
        
        # POS 태깅
        pos_tags = pos_tag(tokens)
        
        # 표제어 변환 (모든 단어 유지)
        lemmatised_words = []
        word_objects = []
        word_indices = []
        
        for word, pos in pos_tags:
            wordnet_pos = self._get_wordnet_pos(pos)
            lemma = self._lemmatizer.lemmatize(word, wordnet_pos)
            lemmatised_words.append(lemma)
            
            # Docs 참조가 있으면 Word 객체 생성/관리
            if self._docs_ref is not None:
                word_obj = self._docs_ref.add_word(lemma)
                word_objects.append(word_obj)
                word_indices.append(word_obj.idx)
        
        self._lemmatised = lemmatised_words
        self._word_objects = word_objects
        self._word_indices = word_indices
        
        return self._lemmatised
    
    @property
    def word_indices(self):
        return self._word_indices
    
    @property
    def word_objects(self):
        return self._word_objects

class Docs:
    def __init__(self):
        self._rawdata: Optional[List[str]] = None
        self._words_list: Optional[List[Word]] = None
        self._sentence_list: Optional[List[Sentence]] = None
        self._nlp: Optional[fasttext.FastText._FastText] = None
        self._lock = threading.Lock()
        self._word_trie = WordTrie()  # 접두사 트리
    
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
            
        self.create_sentence_list()
    
    def add_word(self, word_content: str) -> Word:
        """단어를 트리에 추가하고 Word 객체 반환 (스레드 안전)"""
        with self._lock:
            return self._word_trie.insert_or_get_word(word_content)
    
    def _create_sentence_with_log(self, doc_text: str, index: int) -> Sentence:
        """단일 Sentence 객체 생성 및 로그"""
        sentence = Sentence(docs_ref=self)  # Docs 참조 전달
        sentence.raw = doc_text
        
        # 스레드 안전한 로그 출력
        with self._lock:
            if index % 5000 == 0:  # 5000개마다 로그
                print(f"Read document {index + 1}/{len(self._rawdata)}", file=sys.stderr)
        
        return sentence

    def create_sentence_list(self, max_workers: int = 4) -> None:
        """rawdata로부터 Sentence 객체 리스트 생성 (멀티스레딩)"""
        if self._rawdata is None:
            self._sentence_list = None
            return

        print(f"Creating {len(self._rawdata)} sentences using {max_workers} threads...", file=sys.stderr)
        
        # 멀티스레딩으로 Sentence 객체 생성
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 각 문서에 대해 작업 제출
            future_to_index = {
                executor.submit(self._create_sentence_with_log, doc_text, i): i 
                for i, doc_text in enumerate(self._rawdata)
            }
            
            # 결과를 순서대로 저장할 리스트
            sentences = [None] * len(self._rawdata)
            
            # 완료된 작업들 처리
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    sentence = future.result()
                    sentences[index] = sentence
                except Exception as exc:
                    print(f'Document {index} generated an exception: {exc}', file=sys.stderr)
        
        self._sentence_list = sentences
        
        # 모든 문장 처리 완료 후 words_list 업데이트
        self._words_list = self._word_trie.get_all_words()
        print(f"Created {len(self._words_list)} unique words", file=sys.stderr)

    @property
    def words_list(self):
        return self._words_list
    
    @property
    def sentence_list(self):
        return self._sentence_list
    
    def __str__(self):
        if self._rawdata is None:
            return ""
        words_count = len(self._words_list) if self._words_list else 0
        sentences_count = len(self._sentence_list) if self._sentence_list else 0
        return f"raw_data: {len(self._rawdata)}\nwords_list: {words_count}\nsentence_list: {sentences_count}"
    
    def __getitem__(self, idx):
        return self._rawdata[idx]

if __name__ == "__main__":
    # 사용 예시
    docs = Docs()
    
    # 테스트 데이터
    test_docs = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown fox is jumping over a lazy dog.",
        "The lazy dog sleeps under the tree.",
        "Quick foxes are running in the forest.",
        "Dogs and foxes are playing together.",
        "The forest is full of trees and animals.",
        "Animals like dogs and foxes live in forests.",
        "Trees provide shelter for many animals.",
        "The quick animals are running fast.",
        "Lazy animals prefer to sleep all day."
    ]
    
    docs.rawdata = test_docs
    
    print(docs)
    print(f"\nSample word frequencies:")
    for word in docs.words_list[:10]:  # 첫 10개 단어만 출력
        print(f"'{word.content}': freq={word.freq}, idx={word.idx}")
    
    print(f"\nSample sentence word indices:")
    for i, sentence in enumerate(docs.sentence_list[:3]):  # 첫 3개 문장만 출력
        print(f"Sentence {i+1}: {sentence.word_indices}")
        print(f"Words: {[word.content for word in sentence.word_objects]}")