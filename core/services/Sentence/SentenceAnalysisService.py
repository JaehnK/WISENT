from typing import List, Optional, Any, Dict
from entities import Sentence

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