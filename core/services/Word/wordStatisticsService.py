from typing import List, Dict, Any
from entities import Word

class WordStatisticsService:
    """단어 통계 서비스 - 빈도, 분포 등 통계 정보 제공"""
    
    @staticmethod
    def get_basic_stats(word: Word) -> Dict[str, Any]:
        """기본 통계 정보"""
        return {
            'content': word.content,
            'idx': word.idx,
            'frequency': word.freq,
            'dominant_pos': word.dominant_pos,
            'pos_count': len(word.pos_tags),
            'is_stopword': word.get_stopword_status(),
            'stopword_checked': word.stopword_checked,
            'isnode': word.isnode
        }
    
    @staticmethod
    def get_pos_distribution(word: Word) -> Dict[str, float]:
        """품사 분포 비율 반환"""
        if not word.pos_counts:
            return {}
        total = sum(word.pos_counts.values())
        return {pos: count / total for pos, count in word.pos_counts.items()}
    
    @staticmethod
    def get_top_words_by_frequency(words: List[Word], top_n: int = 10, contain_stopword: bool = False) -> List[Word]:
        """빈도순으로 상위 단어 반환"""
        return sorted(words, key=lambda w: w.freq, reverse=True)[:top_n]
    
    @staticmethod
    def get_word_frequency_summary(words: List[Word]) -> Dict[str, Any]:
        """전체 단어 빈도 요약"""
        if not words:
            return {}
        
        total_words = len(words)
        total_frequency = sum(w.freq for w in words)
        avg_frequency = total_frequency / total_words if total_words > 0 else 0
        
        return {
            'total_unique_words': total_words,
            'total_frequency': total_frequency,
            'average_frequency': avg_frequency,
            'max_frequency': max(w.freq for w in words) if words else 0,
            'min_frequency': min(w.freq for w in words) if words else 0
        }
    
    @staticmethod
    def get_pos_statistics(words: List[Word]) -> Dict[str, int]:
        """품사별 단어 개수 통계"""
        pos_stats = {}
        for word in words:
            for pos in word.pos_tags:
                pos_stats[pos] = pos_stats.get(pos, 0) + 1
        return pos_stats
