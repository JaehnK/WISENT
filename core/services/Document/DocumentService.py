import sys
from typing import List, Optional, Dict, Any
import threading

import spacy

from entities import *

from .TextPreprocssingService import TextPreprocessingService
from .SentenceProcessingService import SentenceProcessingService
from .WordManagementService import WordManagementService
from ..Word.wordAnalysisService import WordAnalysisService
from ..Word.wordStatisticsService import WordStatisticsService

class DocumentService:
    """문서 처리 Facade 서비스 - 다른 서비스들을 조율"""
    
    def __init__(self, model_name='en_core_web_sm', disable_components=None):
        # 데이터 레이어
        self._documents = Documents()
        
        # 서비스 레이어들
        self._preprocessing = TextPreprocessingService(model_name, disable_components)
        self._sentence_service = SentenceProcessingService(self._documents, self._preprocessing)
        self._word_service = WordManagementService(self._documents)
        self._word_analysis = WordAnalysisService()
        self._word_stats = WordStatisticsService()
    
    # === 기존 Docs 클래스 호환 인터페이스 ===
    
    @property
    def rawdata(self) -> Optional[List[str]]:
        """원본 데이터 조회"""
        return self._documents.rawdata
    
    @rawdata.setter
    def rawdata(self, docs: List[str]):
        """원본 데이터 설정 및 자동 처리"""
        self._documents.rawdata = docs
        self.create_sentence_list()
    
    @property
    def words_list(self) -> Optional[List[Word]]:
        """단어 리스트 조회"""
        return self._word_service.get_all_words()
    
    @property
    def sentence_list(self) -> Optional[List[Sentence]]:
        """문장 리스트 조회"""
        return self._documents.sentence_list
    
    def add_word(self, word_content: str, pos_tag: str = None) -> Word:
        """단어 추가"""
        return self._word_service.add_word(word_content, pos_tag)
    
    def create_sentence_list(self, max_workers: int = 1) -> None:
        """문장 리스트 생성 (기존 인터페이스 호환)"""
        if max_workers > 1:
            self.create_sentence_list_parallel()
        else:
            self._sentence_service.create_sentence_list_sequential()
    
    def create_sentence_list_parallel(self, batch_size: int = 100, n_process: int = None) -> None:
        """병렬 문장 리스트 생성"""
        self._sentence_service.create_sentence_list_parallel(batch_size, n_process)
    
    def get_co_occurrence_edges(self, word_to_node: Dict[str, int]):
        """공출현 엣지 생성 (GraphService에서 사용하기 위한 데이터 제공)"""
        return self._documents.get_co_occurrence_edges(word_to_node)
    
    def get_preprocessed_sentences(self)-> List[List[str]]:
        sentences = self._documents.sentence_list
        if sentences is None:
            return []
        return [sentence.lemmatised for sentence in sentences]
    
    def get_sentences_with_word2id(self)-> List[List[int]]:
        sentences = self._documents.sentence_list
        if sentences is None:
            return []
        return [sentence.word_indices for sentence in sentences]
    
    def get_word2vec_data(self, min_count: int = 5) -> Dict[str, Any]:
        """Word2Vec에 필요한 모든 데이터 반환"""
        words = self._word_service.get_all_words()
        
        # min_count 필터링
        filtered_words = [w for w in words if w.freq >= min_count]
        
        word2id = {w.content: w.idx for w in filtered_words}
        id2word = {w.idx: w.content for w in filtered_words}
        word_frequency = {w.idx: w.freq for w in filtered_words}
        total_tokens = sum(w.freq for w in filtered_words)
        
        return {
            'word2id': word2id,
            'id2word': id2word,
            'word_frequency': word_frequency,
            'vocab_size': len(filtered_words),
            'total_tokens': total_tokens
        }
    
    # === 새로운 Facade 메서드들 ===
    
    def analyze_word(self, word_content: str) -> Dict[str, Any]:
        """단어 분석 결과 반환"""
        word = self._word_service.get_word(word_content)
        if not word:
            return None
        
        return {
            'word': word,
            'pos_category': self._word_analysis.get_pos_category(word),
            'is_content_word': self._word_analysis.is_content_word(word),
            'is_function_word': self._word_analysis.is_function_word(word),
            'statistics': self._word_stats.get_basic_stats(word),
            'pos_distribution': self._word_stats.get_pos_distribution(word)
        }
    
    def get_top_words(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """상위 단어들의 상세 정보 반환"""
        words = self._word_service.get_all_words()
        top_words = self._word_stats.get_top_words_by_frequency(words, top_n)
        
        return [
            {
                'word': word,
                'analysis': self._word_analysis.get_pos_category(word),
                'stats': self._word_stats.get_basic_stats(word)
            }
            for word in top_words
        ]
    
    def get_word_statistics_summary(self) -> Dict[str, Any]:
        """전체 단어 통계 요약"""
        words = self._word_service.get_all_words()
        return self._word_stats.get_word_frequency_summary(words)
    
    def get_pos_statistics(self) -> Dict[str, int]:
        """품사별 통계"""
        words = self._word_service.get_all_words()
        return self._word_stats.get_pos_statistics(words)
    
    def get_words_by_pos_category(self, pos_category: str) -> List[Word]:
        """특정 품사 카테고리에 속하는 단어들 반환"""
        words = self._word_service.get_all_words()
        return [
            word for word in words 
            if self._word_analysis.get_pos_category(word) == pos_category
        ]
    
    def __str__(self):
        """기존 호환성"""
        return f"DocumentService(docs={self.get_document_count()}, sentences={self.get_sentence_count()}, words={self.get_word_count()})"
    
    def get_document_count(self) -> int:
        """문서 개수"""
        return len(self._documents.rawdata) if self._documents.rawdata else 0
    
    def get_sentence_count(self) -> int:
        """문장 개수"""
        return len(self._documents.sentence_list) if self._documents.sentence_list else 0
    
    def get_word_count(self) -> int:
        """고유 단어 개수"""
        return self._word_service.get_word_count() if self._word_service else 0