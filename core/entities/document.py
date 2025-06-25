from dataclasses import dataclass
from typing import List, Optional, Dict

from .word import Word
from .sentence import Sentence  
from .trie import WordTrie

@dataclass
class Document:
    """단순한 개별 문서 데이터"""
    content: str
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = f"doc_{id(self)}"
    
    @property
    def word_count(self) -> int:
        return len(self.content.split()) if self.content else 0
    
    def is_valid(self) -> bool:
        return bool(self.content and self.content.strip())


class Documents:
    """전체 문서 컬렉션의 데이터 레이어"""
    
    def __init__(self):
        # 핵심 데이터 3개
        self._rawdata: Optional[List[str]] = None
        self._sentence_list: Optional[List[Sentence]] = None
        self._word_trie = WordTrie()
        
        # 부가 정보
        self._document_objects: Optional[List[Document]] = None
    
    @property
    def rawdata(self) -> Optional[List[str]]:
        return self._rawdata
    
    @rawdata.setter 
    def rawdata(self, docs: List[str]):
        """원본 데이터 설정 및 기본 검증"""
        if len(docs) < 10:
            raise ValueError("Number of documents is less than 10. Not sufficient for clustering.")
        
        self._rawdata = docs
        
        # Document 객체들 생성 (선택적)
        self._document_objects = [
            Document(content=content, doc_id=f"doc_{i}") 
            for i, content in enumerate(docs)
        ]
        
        # 기존 처리 결과 초기화
        self._sentence_list = None
        self._word_trie = WordTrie()
    
    @property
    def sentence_list(self) -> Optional[List[Sentence]]:
        return self._sentence_list
    
    @sentence_list.setter
    def sentence_list(self, sentences: List[Sentence]):
        self._sentence_list = sentences
    
    @property
    def word_trie(self) -> WordTrie:
        return self._word_trie
    
    @property
    def document_objects(self) -> Optional[List[Document]]:
        return self._document_objects
    
    @property
    def words_list(self) -> Optional[List[Word]]:
        """WordTrie에서 모든 단어 추출"""
        if self._word_trie is None:
            return None
        
        try:
            all_words = self._word_trie.get_all_words()
            print(f"🔍 Documents.words_list: WordTrie returned {len(all_words) if all_words else 0} words")
            return all_words
        except Exception as e:
            print(f"❌ Error getting words from WordTrie: {e}")
            return None
    
    def get_document_count(self) -> int:
        """문서 개수"""
        return len(self._rawdata) if self._rawdata else 0
    
    def get_sentence_count(self) -> int:
        """문장 개수"""
        return len(self._sentence_list) if self._sentence_list else 0
    
    def get_word_count(self) -> int:
        """고유 단어 개수"""
        return self._word_trie.word_count if self._word_trie else 0
    
    def get_document(self, index: int) -> str:
        """인덱스로 원본 문서 조회 (기존 호환성)"""
        if not self._rawdata or index >= len(self._rawdata):
            raise IndexError("Document index out of range")
        return self._rawdata[index]
    
    def get_valid_documents(self) -> List[str]:
        """유효한 문서들만 반환"""
        if not self._rawdata:
            return []
        return [doc for doc in self._rawdata if doc and doc.strip()]
    
    def add_word(self, word_content: str, pos_tag: str = None) -> Word:
        """단어를 트리에 추가하고 Word 객체 반환"""
        try:
            word_obj = self._word_trie.insert_or_get_word(word_content, pos_tag)
            return word_obj
        except Exception as e:
            print(f"❌ Error in Documents.add_word: {e}")
            raise
    
    def get_co_occurrence_edges(self, word_to_node: Dict[str, int]):
        """공출현 엣지 생성 (기존 메서드 유지)"""
        from .co_occurence import build_cooccurrence_edges
        return build_cooccurrence_edges(word_to_node, self._sentence_list)
    
    def clear(self):
        """모든 데이터 초기화"""
        self._rawdata = None
        self._sentence_list = None
        self._word_trie = WordTrie()
        self._document_objects = None
    
    def get_stats(self) -> Dict[str, int]:
        """전체 통계"""
        return {
            'document_count': self.get_document_count(),
            'sentence_count': self.get_sentence_count(), 
            'unique_word_count': self.get_word_count(),
            'total_words': sum(doc.word_count for doc in (self._document_objects or [])),
            'valid_documents': len(self.get_valid_documents())
        }
    
    def __str__(self):
        """기존 Docs.__str__ 호환"""
        return (f"Documents: {self.get_document_count()} docs, "
                f"{self.get_sentence_count()} sentences, "
                f"{self.get_word_count()} unique words")
    
    def __getitem__(self, idx):
        """기존 Docs.__getitem__ 호환"""
        return self.get_document(idx)
    
    def __len__(self):
        return self.get_document_count()