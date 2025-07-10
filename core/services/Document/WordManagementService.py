import sys
from typing import List, Optional, Dict
import threading

import spacy

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