
from typing import Dict
from ..entities import Sentence

class AnalysisService:
    """
    처리된 엔티티(Sentence, WordGraph 등)를 분석하여 의미있는 정보를 추출하는 서비스.
    """

    @staticmethod
    def get_sentence_complexity_score(sentence: Sentence) -> Dict[str, float]:
        """문장 복잡도 점수 계산"""
        if not sentence.is_processed:
            return {"error": "Sentence not processed"}
        
        lemmatised = sentence.lemmatised
        word_objects = sentence.word_objects

        if not lemmatised:
            return {
                "avg_word_length": 0,
                "unique_word_ratio": 0,
                "content_word_ratio": 0,
                "sentence_length": 0,
                "char_per_word": 0
            }

        avg_word_length = sum(len(word) for word in lemmatised) / len(lemmatised)
        unique_word_ratio = len(set(lemmatised)) / len(lemmatised)
        content_word_ratio = len(sentence.get_content_words()) / len(word_objects) if word_objects else 0
        
        return {
            "avg_word_length": avg_word_length,
            "unique_word_ratio": unique_word_ratio,
            "content_word_ratio": content_word_ratio,
            "sentence_length": sentence.word_count,
            "char_per_word": sentence.char_count / sentence.word_count if sentence.word_count > 0 else 0
        }
