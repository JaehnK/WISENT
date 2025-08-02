from typing import List, Optional, Dict, Any

from .word import Word

from typing import List, Optional, Any, Dict

from .word import Word



class Sentence:
    """순수 문장 데이터 클래스"""
    
    def __init__(self, raw: str, doc_id: Optional[str] = None, sentence_id: Optional[str] = None):
        # 원본 데이터
        self.raw = raw
        self.doc_id = doc_id
        self.sentence_id = sentence_id or f"sent_{id(self)}"
        
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

    def add_processing_error(self, error: str) -> None:
        """처리 오류 추가"""
        self.processing_errors.append(error)

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
            bool(self.raw and self.raw.strip()) and
            self.char_count > 0 and
            self.word_count > 0
        )
    
    def get_text_preview(self, max_chars: int = 50) -> str:
        """텍스트 미리보기"""
        if not self.raw:
            return ""
        
        preview = self.raw.strip()
        if len(preview) <= max_chars:
            return preview
        
        return preview[:max_chars] + "..."
    
    def copy(self) -> 'Sentence':
        """Sentence 객체 복사"""
        new_sentence = Sentence(
            raw=self.raw,
            doc_id=self.doc_id,
            sentence_id=f"{self.sentence_id}_copy"
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
