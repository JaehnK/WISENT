import sys
from typing import List, Optional, Dict
import threading

import spacy

from entities import *

from .TextPreprocssingService import TextPreprocessingService
from .SentenceProcessingService import SentenceProcessingService
from .WordManagementService import WordManagementService

class DocumentService:
    """문서 처리 Facade 서비스"""
    
    def __init__(self, model_name='en_core_web_sm', disable_components=None):
        # 데이터 레이어
        self._documents = Documents()
        
        # 서비스 레이어들 (문서 전처리만 담당)
        self._preprocessing = TextPreprocessingService(model_name, disable_components)
        self._sentence_service = SentenceProcessingService(self._documents, self._preprocessing)
        self._word_service = WordManagementService(self._documents)
    
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
    
    def __str__(self):
        """기존 호환성"""
        return str(self._documents)
    
    def __getitem__(self, idx):
        """기존 호환성"""
        return self._documents[idx]
    
    def __len__(self):
        return len(self._documents)
    
    # === 새로운 편의 메서드들 ===
    
    def get_stats(self) -> Dict[str, int]:
        """전체 통계"""
        doc_stats = self._documents.get_stats()
        word_stats = self._word_service.get_word_stats()
        return {**doc_stats, **word_stats}
    
    def get_top_words(self, top_n: int = 500, exclude_stopwords: bool = True) -> List[Word]:
        """상위 빈도 단어들"""
        print("get top Words called" )
        return self._word_service.get_top_words(top_n, exclude_stopwords)
    
    def reset(self):
        """모든 데이터 초기화"""
        self._documents.clear()
    
    # === 개별 서비스 접근 (필요시) ===
    
    @property
    def preprocessing_service(self) -> TextPreprocessingService:
        return self._preprocessing
    
    @property
    def sentence_service(self) -> SentenceProcessingService:
        return self._sentence_service
    
    @property
    def word_service(self) -> WordManagementService:
        return self._word_service