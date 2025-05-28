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