from typing import List, Optional, Union
import re
import sys

import nltk
import spacy
import contractions
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from .word import Word

class Sentence:
    def __init__(self, docs_ref=None):
        self._raw: Optional[str] = None
        self._lemmatised: Optional[List[str]] = None
        self._word_indices: Optional[List[int]] = None
        self._word_objects: Optional[List[Word]] = None
        self._docs_ref = docs_ref  # Docs 객체 참조
        self._spacy_doc = None # spaCy 문서 객체 캐시
        
    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, sentence: str):
        self._raw = sentence
        self._lemmatised = None
        self._word_indices = None
        self._word_objects = None
        self._spacy_doc = None
        
        if (self._docs_ref and hasattr(self._docs_ref, 'nlp')):
            try:
                expanded_text = self._expand_contractions(sentence)
                spacy_doc = self._docs_ref.nlp(expanded_text)
                self.set_from_spacy_doc(spacy_doc, sentence)
            except Exception as e:
                print(f"spaCy processing failed, using fallback: {e}", file=sys.stderr)
                self.lemmatise_fallback()
    
    def set_from_spacy_doc(self, spacy_doc, original_text: str):
        """spaCy 문서 객체로부터 직접 설정 (최적화된 방법)"""
        self._raw = original_text
        self._spacy_doc = spacy_doc
        self._lemmatised = None
        self._word_indices = None
        self._word_objects = None
        self._process_spacy_doc()
        
            
    def _expand_contractions(self, text):
        """축약형을 확장하는 함수 - contractions 라이브러리 사용"""
        try:
            expanded_text = contractions.fix(text)
            return expanded_text.lower()
        except Exception as e:
            print(f"Contraction expansion failed for text: {text[:50]}... Error: {e}", file=sys.stderr)
            return text.lower()
        
    def _process_spacy_doc(self):
        """spaCy 문서 객체로부터 lemmatised 정보 추출"""
        if self._spacy_doc is None:
            return
        try:
            lemmatised_words = []
            word_objects = []
            word_indices = []
            
            for token in self._spacy_doc:
                # 필터링 조건
                if (not token.is_punct and  # 구두점 제외
                    not token.is_space and  # 공백 제외
                    len(token.text.strip()) >= 1 and  # 빈 토큰 제외
                    token.text.strip()):  # 의미있는 단어 유지
                    
                    # spaCy의 lemma 사용 (더 정확함)
                    lemma = token.lemma_.lower()
                    
                    # spaCy POS 태그를 NLTK 스타일로 변환 (호환성)
                    pos_tag = self._convert_spacy_pos_to_nltk(token.pos_, token.tag_)
                    
                    lemmatised_words.append(lemma)
                    
                    # Docs 참조가 있으면 Word 객체 생성/관리
                    if self._docs_ref is not None:
                        word_obj = self._docs_ref.add_word(lemma, pos_tag)
                        if not word_obj.stopword_checked:
                            word_obj.set_stopword_status(token.is_stop)
                        word_objects.append(word_obj)
                        word_indices.append(word_obj.idx)
            
            self._lemmatised = lemmatised_words
            self._word_objects = word_objects
            self._word_indices = word_indices
        except Exception as e:
            print(f"spaCy processing failed for sentence: {self._raw[:50]}... Error: {e}", file=sys.stderr)
            # 실패시 빈 리스트로 초기화
            self._lemmatised = []
            self._word_objects = []
            self._word_indices = []
            
    def _convert_spacy_pos_to_nltk(self, spacy_pos, spacy_tag):
            """spaCy POS 태그를 NLTK 스타일로 변환 (기존 코드 호환성)"""
            # spaCy -> NLTK POS 매핑
            pos_mapping = {
                'ADJ': 'JJ',      # 형용사
                'ADP': 'IN',      # 전치사
                'ADV': 'RB',      # 부사
                'AUX': 'VB',      # 조동사
                'CONJ': 'CC',     # 접속사
                'CCONJ': 'CC',    # 등위접속사
                'DET': 'DT',      # 한정사
                'INTJ': 'UH',     # 감탄사
                'NOUN': 'NN',     # 명사
                'NUM': 'CD',      # 수사
                'PART': 'RP',     # 불변화사
                'PRON': 'PRP',    # 대명사
                'PROPN': 'NNP',   # 고유명사
                'PUNCT': '.',     # 구두점
                'SCONJ': 'IN',    # 종속접속사
                'SYM': 'SYM',     # 기호
                'VERB': 'VB',     # 동사
                'X': 'XX',        # 기타
                'SPACE': 'SP'     # 공백
            }
            
            return pos_mapping.get(spacy_pos, spacy_tag)
    
    def lemmatise_fallback(self):
        """레거시 NLTK 기반 폴백 메서드"""
        if self._raw is None:
            raise ValueError("Raw sentence is not set.")
        
        try:
            # 간단한 폴백: 공백 기준 분할 후 소문자 변환
            expanded_text = self._expand_contractions(self._raw)
            cleaned_text = re.sub(r'[^\w\s]', '', expanded_text.lower())
            tokens = cleaned_text.split()
            
            # 간단한 필터링
            lemmatised_words = [token for token in tokens 
                                if len(token) >= 2 or token in ['i', 'a']]
            
            word_objects = []
            word_indices = []
            
            # Word 객체 생성
            if self._docs_ref is not None:
                for word in lemmatised_words:
                    word_obj = self._docs_ref.add_word(word, 'NN')  # 기본 POS
                    word_objects.append(word_obj)
                    word_indices.append(word_obj.idx)
            
            self._lemmatised = lemmatised_words
            self._word_objects = word_objects
            self._word_indices = word_indices
            
        except Exception as e:
            print(f"Fallback lemmatisation failed: {e}", file=sys.stderr)
            self._lemmatised = []
            self._word_objects = []
            self._word_indices = []
    
    def lemmatise(self):
        """문장을 표제어 형태로 변환하고 Word 객체 생성"""
        """호환성을 위한 메서드 - 이미 처리된 경우 스킵"""
        if self._lemmatised is not None:
            return self._lemmatised
        
        if self._spacy_doc is not None:
            self._process_spacy_doc()
        else:
            self.lemmatise_fallback()
        
        return self._lemmatised
    
    @property
    def word_indices(self):
        return self._word_indices
    
    @property
    def word_objects(self):
        return self._word_objects
