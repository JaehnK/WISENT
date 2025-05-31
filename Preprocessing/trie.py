from typing import List, Optional, Dict
from .word import Word

class TrieNode:
    """접두사 트리 노드"""
    def __init__(self):
        self.children = {}
        self.word_obj: Optional[Word] = None
        self.is_end_of_word = False

class WordTrie:
    """접두사 트리를 이용한 Word 객체 관리"""
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0  # 고유 단어 개수 (idx 할당용)
    
    def insert_or_get_word(self, word_content: str, pos_tag: str = None) -> Word:
        """단어를 트리에 삽입하거나 기존 Word 객체 반환"""
        node = self.root
        
        # 트리 탐색
        for char in word_content:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        # 단어 끝 지점에서 Word 객체 처리
        if node.is_end_of_word:
            # 기존 단어 - 빈도수 및 품사 정보 업데이트
            node.word_obj.increment_freq(pos_tag)
            return node.word_obj
        else:
            # 새 단어 - Word 객체 생성 및 설정
            word_obj = Word(word_content)
            word_obj.idx = self.word_count
            word_obj.increment_freq(pos_tag)
            
            node.word_obj = word_obj
            node.is_end_of_word = True
            self.word_count += 1
            
            return word_obj
    
    def get_all_words(self) -> List[Word]:
        """모든 Word 객체 반환"""
        words = []
        self._dfs_collect_words(self.root, words)
        return words
    
    def _dfs_collect_words(self, node: TrieNode, words: List[Word]):
        """DFS로 모든 Word 객체 수집"""
        if node.is_end_of_word:
            words.append(node.word_obj)
        
        for child in node.children.values():
            self._dfs_collect_words(child, words)
    
    def get_content_words(self) -> List[Word]:
        """내용어(content words)만 반환 - 불용어 제외"""
        all_words = self.get_all_words()
        return [word for word in all_words if not word.is_stopword]
    
    def get_stopwords(self) -> List[Word]:
        """불용어만 반환"""
        all_words = self.get_all_words()
        return [word for word in all_words if word.is_stopword]

    def get_top_words_by_pos(self, top_n: int = 500, exclude_stopwords: bool = True) -> List[Word]:
        """명사, 동사, 형용사만 필터링 후 빈도수 기준 상위 N개 단어 반환
        
        Args:
            top_n: 상위 몇 개까지 가져올지 (기본값: 500)
            exclude_stopwords: 불용어 제외 여부 (기본값: True)
        
        Returns:
            빈도수 내림차순으로 정렬된 Word 객체 리스트 (최대 top_n개)
        """
        # 모든 단어 수집
        all_words = self.get_all_words()
        
        # 명사, 동사, 형용사만 필터링
        filtered_words = []
        for word in all_words:
            # 불용어 제외 옵션 확인
            if exclude_stopwords and word.is_stopword:
                continue
                
            if word._dominant_pos and (word.is_noun() or word.is_verb() or word.is_adjective()):
                filtered_words.append(word)
        
        # 빈도수 기준 내림차순 정렬 후 상위 N개 반환
        return sorted(filtered_words, key=lambda w: w._freq, reverse=True)[:top_n]
    
    def get_word_stats(self) -> Dict[str, int]:
        """품사별 단어 통계 반환 (불용어 정보 포함)"""
        all_words = self.get_all_words()

        # 기본 품사 통계
        noun_count = sum(1 for w in all_words if w._dominant_pos and w.is_noun())
        verb_count = sum(1 for w in all_words if w._dominant_pos and w.is_verb())
        adj_count = sum(1 for w in all_words if w._dominant_pos and w.is_adjective())
        
        # 불용어 통계
        stopword_count = sum(1 for w in all_words if w.is_stopword)
        content_word_count = len(all_words) - stopword_count
        
        # 불용어 제외 품사 통계
        content_nouns = sum(1 for w in all_words if w._dominant_pos and w.is_noun() and not w.is_stopword)
        content_verbs = sum(1 for w in all_words if w._dominant_pos and w.is_verb() and not w.is_stopword)
        content_adjs = sum(1 for w in all_words if w._dominant_pos and w.is_adjective() and not w.is_stopword)

        return {
            'total_words': len(all_words),
            'stopwords': stopword_count,
            'content_words': content_word_count,
            'nouns': noun_count,
            'verbs': verb_count,
            'adjectives': adj_count,
            'content_nouns': content_nouns,
            'content_verbs': content_verbs,
            'content_adjectives': content_adjs,
            'other_pos': len(all_words) - noun_count - verb_count - adj_count
        }