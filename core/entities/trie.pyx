# cython: language_level=3
# distutils: language=c++

from typing import List, Optional, Dict
from cython.operator cimport dereference, preincrement
from libc.stdlib cimport malloc, free
from libc.string cimport strlen, strcpy, strcmp, strdup
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string


from .sentence import Sentence
from .word import Word

from typing import Optional


cdef class TrieNode:
    """접두사 트리 노드"""
    cdef:
        dict    children
        object  word_obj  # Word 객체
        bint    is_end_of_word
    
    def __init__(self):
        self.children = {}
        self.word_obj = None
        self.is_end_of_word = False

cdef void _dfs_collect_words(TrieNode node, list words):
    """내부용 DFS 함수"""
    if node.is_end_of_word:
        words.append(node.word_obj)
    
    # Python dict 순회 (C++ unordered_map 대신)
    cdef:
        object child_node
    
    for child_node in node.children.values():
        _dfs_collect_words(child_node, words)

cdef TrieNode _find_node(TrieNode node, str word_content, str pos_tag):
    cdef:
        char c

    for char in word_content:
        c = ord(char)
        if c not in node.children:
            node.children[c] = TrieNode()
        node = node.children[c]
    
    return (node)

class WordTrie:
    """접두사 트리를 이용하여 word 객체 저장"""

    # cdef public TrieNode    root
    # cdef public int         word_count
    # cdef:
    #     TrieNode    root
    #     int         word_count

    def __init__(self) -> None:
        self.root = TrieNode()
        self.word_count:int = 0 # idx count

    # cdef TrieNode _find_node(TrieNode node, str word_content, pos_tag:str):
    #     cdef:
    #         char        c

    #     for char in word_content:
    #         c = ord(char)
    #         if node.children.find(c) == node.children.end():
    #             node.children[c] = TrieNode()
    #         node = node.children[c]
        
    #     return (node)

    def insert_or_get_word(self, word_content: str, pos_tag: str = None) -> Word:
        
        node = _find_node(self.root, word_content, pos_tag)

        if node.is_end_of_word:
            node.word_obj.increment_freq(pos_tag)
            return (node.word_obj)
        else:
            word_obj = Word(word_content)
            word_obj.idx = self.word_count
            word_obj.increment_freq(pos_tag)

            node.word_obj = word_obj
            node.is_end_of_word = True
            self.word_count += 1

            return (word_obj)

    def get_all_words(self) -> List[Word]:
        """모든 Word 객체 반환"""

        words = []
        _dfs_collect_words(self.root, words)
        return (words)

    # cdef _dfs_collect_words(self, TrieNode node, list words):
        
    #     if node.is_end_of_word:
    #         words.append(node.word_obj)
        
    #     cdef:
    #         unordered_map[char, TrieNode].iterator it
    #         TrieNode    child_node
        
    #     it = node.children.begin()
    #     while it != node.children.end():
    #         child_node = dereference(it).second
    #         self._dfs_collect_words(child_node, words)
    #         preincrement(it)

    def get_content_words(self) -> List[Word]:
        """내용어(content words)만 반환 - 불용어 제외"""
        all_words = self.get_all_words()
        return [word for word in all_words if not word.is_stopword]
    
    def get_stopwords(self) -> List[Word]:
        """불용어만 반환"""
        all_words = self.get_all_words()
        return [word for word in all_words if word.is_stopword]

    # cdef list   _filter_words_by_pos(self, list all_words, bint exclude_stopwords):
    #     cdef:
    #         list    filtered_words = []
    #         object  word
    #         int     i
    #         int     words_count = len(all_words)

    #     for i in range(words_count):
    #         word = all_words[i]

    #         if exclude_stopwords and word.is_stopword:
    #             continue
            
    #         if word._dominant_pos and (word.is_noun() or word.is_verb() or word.is_adjective()):
    #             filtered_words.append(word)
        
    #     return (filtered_words)

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
        # print("All Words: ", len(all_words))
        # 명사, 동사, 형용사만 필터링
        filtered_words = []
        for word in all_words:
            # 불용어 제외 옵션 확인
            if exclude_stopwords and word.is_stopword:
                continue
            # print(f"{word.content} : {word._dominant_pos} {word.is_noun()} {word.is_verb()} {word.is_adjective()}")
            if word._dominant_pos and (word.is_noun() or word.is_verb() or word.is_adjective()):
                filtered_words.append(word)
        
        # 빈도수 기준 내림차순 정렬 후 상위 N개 반환
        print("filtered word: ", len(filtered_words))
        return sorted(filtered_words, key=lambda w: w.freq, reverse=True)[:top_n]


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