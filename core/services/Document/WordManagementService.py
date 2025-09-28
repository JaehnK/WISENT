import sys
from typing import List, Optional, Dict
import threading

import spacy
import numpy as np

from entities import *

class WordManagementService:
    """단어 관리 서비스"""
    
    def __init__(self, documents: Documents):
        self._documents = documents
        self._lock = threading.Lock()
    
    def add_word(self, word_content: str, pos_tag: str = None) -> Word:
        """단어를 트리에 추가하고 Word 객체 반환 (스레드 안전)"""
        with self._lock:
            return self._documents.add_word(word_content, pos_tag)
    
    def get_all_words(self) -> List[Word]:
        """모든 단어 조회"""
        return self._documents.words_list or []
    
    def get_word_stats(self) -> Dict[str, int]:
        """단어 통계"""
        return self._documents.word_trie.get_word_stats()
    
    def get_top_words(self, top_n: int = 500, exclude_stopwords: bool = True) -> List[Word]:
        """상위 빈도 단어들"""
        return self._documents.word_trie.get_top_words_by_pos(top_n, exclude_stopwords)
    
    def get_word2vec_mappings(self) -> tuple[Dict[str, int], Dict[int, str]]:
        """Word2Vec용 매핑 반환"""
        words = self.get_all_words()
        word2id = {w.content: w.idx for w in words}
        id2word = {w.idx: w.content for w in words}
        return word2id, id2word
    
    def update_word_bert_embedding(self, word_content: str, embedding: np.ndarray) -> None:
        """특정 단어의 BERT 임베딩을 업데이트 (스레드 안전)"""
        with self._lock:
            word_obj = self._documents.add_word(word_content)  # 없으면 생성, 있으면 반환
            word_obj.update_bert_embedding(embedding)
    
    def get_word_bert_embedding(self, word_content: str) -> Optional[np.ndarray]:
        """특정 단어의 BERT 임베딩 반환"""
        words = self.get_all_words()
        for word in words:
            if word.content == word_content:
                return word.bert_embedding
        return None
    
    def update_word_w2v_embedding(self, word_content: str, embedding: np.ndarray) -> None:
        """특정 단어의 Word2Vec 임베딩을 업데이트 (스레드 안전)"""
        with self._lock:
            word_obj = self._documents.add_word(word_content)
            word_obj.set_w2v_embedding(embedding)