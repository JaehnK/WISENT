import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from .sentense import Sentence
from .word import Word
from .trie import WordTrie

class Docs:
    def __init__(self):
        self._rawdata: Optional[List[str]] = None
        self._words_list: Optional[List[Word]] = None
        self._sentence_list: Optional[List[Sentence]] = None
        # self._nlp: Optional[fasttext.FastText._FastText] = None
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
