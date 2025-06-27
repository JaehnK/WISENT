from typing import List, Optional, Dict, Any

from .word import Word

from typing import List, Optional, Any, Dict

from .word import Word


class Sentence:
    """순수 문장 데이터 클래스 - 기존 호환성 완벽 유지"""
    
    def __init__(self, raw: str = "", doc_id: Optional[str] = None, 
                 sentence_id: Optional[str] = None, docs_ref: Optional[Any] = None):
        # 원본 데이터 (private storage)
        self.__raw = raw
        self.doc_id = doc_id
        self.sentence_id = sentence_id or f"sent_{id(self)}"
        
        # 기존 호환성을 위한 docs_ref
        self.docs_ref = docs_ref
        
        # 처리된 데이터
        self.lemmatised: List[str] = []
        self.word_objects: List[Word] = []
        self.word_indices: List[int] = []
        
        # 언어학적 정보
        self.language: Optional[str] = None
        self.pos_tags: List[str] = []
        
        # 메타데이터
        self.char_count = len(raw) if raw else 0
        self.word_count = len(raw.split()) if raw else 0
        self.processing_errors: List[str] = []
        self.is_processed = False
    
    # === 기존 호환성을 위한 raw property ===
    
    @property 
    def raw(self) -> str:
        """기존 raw property getter"""
        return self.__raw
    
    @raw.setter
    def raw(self, sentence: str):
        """기존 raw property setter - 자동 처리 포함"""
        self.__raw = sentence
        
        # 기본 통계 업데이트
        self.char_count = len(sentence) if sentence else 0
        self.word_count = len(sentence.split()) if sentence else 0
        
        # 기존 방식: docs_ref가 있고 nlp 모델이 있으면 자동 처리
        if self.docs_ref and hasattr(self.docs_ref, 'nlp') and sentence:
            try:
                expanded_text = self._expand_contractions(sentence)
                spacy_doc = self.docs_ref.nlp(expanded_text)
                self.set_from_spacy_doc(spacy_doc, sentence)
            except Exception as e:
                print(f"spaCy processing failed, using fallback: {e}")
                self._process_with_fallback()
    
    # === 핵심 처리 메서드들 ===
    
    def set_from_spacy_doc(self, spacy_doc, original_text: str):
        """기존 호환성을 위한 메서드 - spaCy 문서 객체로부터 직접 설정"""
        self.__raw = original_text
        
        # spaCy 처리 로직
        try:
            lemmatised_words = []
            word_objects = []
            word_indices = []
            pos_tags = []
            
            for token in spacy_doc:
                if (not token.is_punct and not token.is_space and 
                    len(token.text.strip()) >= 1 and token.text.strip() and
                    self._is_valid_token(str(token))):
                    
                    lemma = token.lemma_.lower()
                    pos_tag = self._convert_spacy_pos_to_nltk(token.pos_, token.tag_)
                    
                    lemmatised_words.append(lemma)
                    pos_tags.append(pos_tag)
                    
                    if self.docs_ref is not None:
                        word_obj = self.docs_ref.add_word(lemma, pos_tag)
                        if not word_obj.stopword_checked:
                            if token.is_stop:
                                print(word_obj.content, "is StopWord")
                            if token.is_stop is False:
                                print(word_obj.content , "is NOT STOPWORD")
                            word_obj.set_stopword_status(token.is_stop)
                        word_objects.append(word_obj)
                        word_indices.append(word_obj.idx)
            
            self.set_processed_data(lemmatised_words, word_objects, word_indices, pos_tags)
            
        except Exception as e:
            self.add_processing_error(f"spaCy processing failed: {e}")
            print(f"❌ Error processing sentence: {self.get_text_preview()}... Error: {e}")
    
    def lemmatise(self) -> List[str]:
        """기존 호환성을 위한 메서드"""
        if self.lemmatised:
            return self.lemmatised
        
        # 처리되지 않았으면 폴백 처리
        if self.docs_ref:
            self._process_with_fallback()
        
        return self.lemmatised
    
    # === 데이터 처리 메서드들 ===
    
    def set_processed_data(self, 
                          lemmatised: List[str],
                          word_objects: List[Word],
                          word_indices: List[int],
                          pos_tags: List[str] = None) -> None:
        """처리 결과 데이터 설정"""
        self.lemmatised = lemmatised
        self.word_objects = word_objects
        self.word_indices = word_indices
        
        if pos_tags:
            self.pos_tags = pos_tags
        
        self.is_processed = True
        self.word_count = len(lemmatised)
    
    def add_word_data(self, word: Word, lemma: str, word_index: int, pos_tag: str = None) -> None:
        """단어 데이터 추가 (한 개씩)"""
        self.word_objects.append(word)
        self.lemmatised.append(lemma)
        self.word_indices.append(word_index)
        
        if pos_tag:
            self.pos_tags.append(pos_tag)
        
        self.word_count = len(self.lemmatised)
    
    def add_processing_error(self, error: str) -> None:
        """처리 오류 추가"""
        self.processing_errors.append(error)
    
    def clear_processing_errors(self) -> None:
        """처리 오류 초기화"""
        self.processing_errors.clear()
    
    # === Private 헬퍼 메서드들 ===
    
    def _expand_contractions(self, text: str) -> str:
        """축약형을 확장하는 함수"""
        try:
            import contractions
            expanded_text = contractions.fix(text)
            return expanded_text.lower()
        except Exception:
            return text.lower()
    
    def _process_with_fallback(self) -> None:
        """폴백 처리 방법"""
        try:
            import re
            expanded_text = self._expand_contractions(self.__raw)
            cleaned_text = re.sub(r'[^\w\s]', '', expanded_text.lower())
            tokens = cleaned_text.split()
            
            lemmatised_words = [token for token in tokens 
                               if len(token) >= 2 or token in ['i', 'a']]
            
            word_objects = []
            word_indices = []
            
            if self.docs_ref is not None:
                for word in lemmatised_words:
                    word_obj = self.docs_ref.add_word(word, 'NN')
                    word_objects.append(word_obj)
                    word_indices.append(word_obj.idx)
            
            self.set_processed_data(lemmatised_words, word_objects, word_indices)
            
        except Exception as e:
            self.add_processing_error(f"Fallback processing failed: {e}")
    
    def _is_valid_token(self, text: str) -> bool:
        """토큰 유효성 검사"""
        import re
        return bool(re.match(r'^[\w가-힣\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF]+$', text))
    
    def _convert_spacy_pos_to_nltk(self, spacy_pos: str, spacy_tag: str) -> str:
        """spaCy POS 태그를 NLTK 스타일로 변환"""
        pos_mapping = {
            'ADJ': 'JJ', 'ADP': 'IN', 'ADV': 'RB', 'AUX': 'VB',
            'CONJ': 'CC', 'CCONJ': 'CC', 'DET': 'DT', 'INTJ': 'UH',
            'NOUN': 'NN', 'NUM': 'CD', 'PART': 'RP', 'PRON': 'PRP',
            'PROPN': 'NNP', 'PUNCT': '.', 'SCONJ': 'IN', 'SYM': 'SYM',
            'VERB': 'VB', 'X': 'XX', 'SPACE': 'SP'
        }
        return pos_mapping.get(spacy_pos, spacy_tag)
    
    # === 분석 메서드들 ===
    
    def get_words_by_pos(self, pos_category: str) -> List[Word]:
        """특정 품사의 단어들만 반환"""
        if not self.pos_tags or len(self.pos_tags) != len(self.word_objects):
            return []
        
        result = []
        for word, pos in zip(self.word_objects, self.pos_tags):
            if pos.upper().startswith(pos_category.upper()):
                result.append(word)
        
        return result
    
    def get_content_words(self) -> List[Word]:
        """내용어들만 반환 (불용어 제외)"""
        return [word for word in self.word_objects if not word.get_stopword_status()]
    
    def get_sentence_stats(self) -> Dict[str, Any]:
        """문장 통계 정보"""
        unique_words = len(set(word.content for word in self.word_objects))
        content_words = len(self.get_content_words())
        
        return {
            'sentence_id': self.sentence_id,
            'doc_id': self.doc_id,
            'char_count': self.char_count,
            'word_count': self.word_count,
            'unique_words': unique_words,
            'content_words': content_words,
            'pos_count': len(set(self.pos_tags)) if self.pos_tags else 0,
            'has_errors': bool(self.processing_errors),
            'error_count': len(self.processing_errors),
            'is_processed': self.is_processed,
            'language': self.language
        }
    
    # === 유틸리티 메서드들 ===
    
    def is_valid(self) -> bool:
        """문장 유효성 검사"""
        return (
            bool(self.__raw and self.__raw.strip()) and
            self.char_count > 0 and
            self.word_count > 0
        )
    
    def is_processable(self) -> bool:
        """처리 가능 여부 (최소 길이 확인)"""
        return (
            self.is_valid() and
            self.word_count >= 2 and  # 최소 2단어
            self.char_count >= 5      # 최소 5자
        )
    
    def get_text_preview(self, max_chars: int = 50) -> str:
        """텍스트 미리보기"""
        if not self.__raw:
            return ""
        
        preview = self.__raw.strip()
        if len(preview) <= max_chars:
            return preview
        
        return preview[:max_chars] + "..."
    
    def copy(self) -> 'Sentence':
        """Sentence 객체 복사"""
        new_sentence = Sentence(
            raw=self.__raw,
            doc_id=self.doc_id,
            sentence_id=f"{self.sentence_id}_copy",
            docs_ref=self.docs_ref
        )
        
        # 처리된 데이터 복사
        new_sentence.lemmatised = self.lemmatised.copy()
        new_sentence.word_objects = self.word_objects.copy()
        new_sentence.word_indices = self.word_indices.copy()
        new_sentence.language = self.language
        new_sentence.pos_tags = self.pos_tags.copy()
        new_sentence.char_count = self.char_count
        new_sentence.word_count = self.word_count
        new_sentence.processing_errors = self.processing_errors.copy()
        new_sentence.is_processed = self.is_processed
        
        return new_sentence
    
    # === 기존 호환성을 위한 private attribute 접근 ===
    
    @property
    def _raw(self) -> str:
        """기존 코드 호환성을 위한 _raw 접근"""
        return self.__raw
    
    @_raw.setter  
    def _raw(self, value: str):
        """기존 코드 호환성을 위한 _raw 설정"""
        self.__raw = value
        self.char_count = len(value) if value else 0
        
    @property
    def _lemmatised(self) -> List[str]:
        """기존 코드 호환성을 위한 _lemmatised 접근"""
        return self.lemmatised
    
    @_lemmatised.setter
    def _lemmatised(self, value: List[str]):
        """기존 코드 호환성을 위한 _lemmatised 설정"""
        self.lemmatised = value
        
    @property
    def _word_objects(self) -> List[Word]:
        """기존 코드 호환성을 위한 _word_objects 접근"""
        return self.word_objects
    
    @_word_objects.setter
    def _word_objects(self, value: List[Word]):
        """기존 코드 호환성을 위한 _word_objects 설정"""
        self.word_objects = value
        
    @property  
    def _word_indices(self) -> List[int]:
        """기존 코드 호환성을 위한 _word_indices 접근"""
        return self.word_indices
    
    @_word_indices.setter
    def _word_indices(self, value: List[int]):
        """기존 코드 호환성을 위한 _word_indices 설정"""
        self.word_indices = value
    
    # === Magic Methods ===
    
    def __str__(self) -> str:
        return self.get_text_preview(100)
    
    def __repr__(self) -> str:
        return f"Sentence(id='{self.sentence_id}', words={self.word_count}, processed={self.is_processed})"
    
    def __len__(self) -> int:
        return self.word_count
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Sentence):
            return False
        return self.sentence_id == other.sentence_id
    
    def __hash__(self) -> int:
        return hash(self.sentence_id)


# === SentenceAnalysisService는 별도 파일에서 import ===

class SentenceAnalysisService:
    """문장 분석 서비스 - 정적 메서드들"""
    
    @staticmethod
    def expand_contractions(text: str) -> str:
        """축약형을 확장하는 함수"""
        try:
            import contractions
            expanded_text = contractions.fix(text)
            return expanded_text.lower()
        except Exception as e:
            print(f"Contraction expansion failed for text: {text[:50]}... Error: {e}")
            return text.lower()
    
    @staticmethod
    def is_valid_token(text: str) -> bool:
        """토큰 유효성 검사"""
        import re
        return bool(re.match(r'^[\w가-힣\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF]+$', text))
    
    @staticmethod
    def process_with_spacy(sentence: Sentence, spacy_doc, docs_ref) -> None:
        """spaCy 처리 (외부에서 호출용)"""
        sentence.set_from_spacy_doc(spacy_doc, sentence.raw)
    
    @staticmethod
    def process_with_fallback(sentence: Sentence, docs_ref) -> None:
        """폴백 처리 (외부에서 호출용)"""
        sentence._process_with_fallback()
    
    @staticmethod
    def get_sentence_complexity_score(sentence: Sentence) -> Dict[str, float]:
        """문장 복잡도 점수 계산"""
        if not sentence.is_processed:
            return {"error": "Sentence not processed"}
        
        avg_word_length = sum(len(word) for word in sentence.lemmatised) / len(sentence.lemmatised) if sentence.lemmatised else 0
        unique_word_ratio = len(set(sentence.lemmatised)) / len(sentence.lemmatised) if sentence.lemmatised else 0
        content_word_ratio = len(sentence.get_content_words()) / len(sentence.word_objects) if sentence.word_objects else 0
        
        return {
            "avg_word_length": avg_word_length,
            "unique_word_ratio": unique_word_ratio,
            "content_word_ratio": content_word_ratio,
            "sentence_length": sentence.word_count,
            "char_per_word": sentence.char_count / sentence.word_count if sentence.word_count > 0 else 0
        }