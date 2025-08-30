from abc import ABC, abstractmethod
from typing import List, Dict, Any
from entities import Word, Sentence, Document

class IWordAnalysisService(ABC):
    """단어 분석 서비스 인터페이스"""
    
    @abstractmethod
    def get_pos_category(self, word: Word) -> str:
        pass
    
    @abstractmethod
    def is_content_word(self, word: Word) -> bool:
        pass

class IWordStatisticsService(ABC):
    """단어 통계 서비스 인터페이스"""
    
    @abstractmethod
    def get_basic_stats(self, word: Word) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_top_words_by_frequency(self, words: List[Word], top_n: int) -> List[Word]:
        pass
