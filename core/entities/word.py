from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Word:
    # 기본 정보
    content: str
    idx: Optional[int] = None
    
    # 빈도 정보
    freq: int = 0
    
    # 품사 정보
    pos_tags: List[str] = field(default_factory=list)  # 이 단어가 나타난 모든 품사들
    pos_counts: Dict[str, int] = field(default_factory=dict)  # 각 품사별 등장 횟수
    _dominant_pos: Optional[str] = None  # 가장 빈번한 품사
    
    # 불용어 정보
    is_stopword: Optional[bool] = None # 불용어 여부 (None=미확인)
    stopword_checked: bool = False  # 불용어 체크 완료 여부
    
    # 노드 정보 (그래프에서 사용)
    isnode: Optional[bool] = None
    
    def __post_init__(self):
        """초기화 후 검증"""
        if not self.content:
            raise ValueError("Word content cannot be empty")
        
        if self.pos_counts is None:
            self.pos_counts = {}
        if self.pos_tags is None:
            self.pos_tags = []
    
    def increment_freq(self, pos_tag: str = None):
        """빈도 수 증가 및 품사 정보 업데이트"""
        self.freq += 1
        
        if pos_tag:
            if pos_tag not in self.pos_counts:
                self.pos_counts[pos_tag] = 0
                self.pos_tags.append(pos_tag) # 이렇게 두개로 관리할 필요가 있나?
            self.pos_counts[pos_tag] += 1
        
        self.dominant_pos = max(self.pos_counts.items(), key=lambda x: x[1])[0]
    
    def set_stopword_status(self, is_stopword: bool):
        """불용어 상태를 직접 설정"""
        self.is_stopword = is_stopword
        self.stopword_checked = True
    
    def get_stopword_status(self) -> bool:
        """불용어 여부 반환 (확인되지 않은 경우 False 반환)"""
        return self.is_stopword if self.is_stopword is not None else False
    
    def get_pos_distribution(self) -> Dict[str, float]:
        """품사 분포 비율 반환"""
        if not self.pos_counts:
            return {}
        total = sum(self.pos_counts.values())
        return {pos: count / total for pos, count in self.pos_counts.items()}

    def get_basic_stats(self) -> Dict[str, Any]:
        """기본 통계 정보"""
        return {
            'content': self.content,
            'idx': self.idx,
            'frequency': self.freq,
            'dominant_pos': self.dominant_pos,
            'pos_count': len(self.pos_tags),
            'is_stopword': self.get_stopword_status(),
            'stopword_checked': self.stopword_checked,
            'isnode': self.isnode
        }
    
    def is_noun(self) -> bool:
        """명사 여부 확인"""
        if not self.dominant_pos:
            return False
        pos = self.dominant_pos.upper()
        print(pos)
        return any(pos.startswith(p) for p in ['N', 'NN', 'NNS', 'NNP', 'NNPS'])

    def is_verb(self) -> bool:
        """동사 여부 확인"""
        if not self.dominant_pos:
            return False
        pos = self.dominant_pos.upper()
        print(pos)
        return any(pos.startswith(p) for p in ['V', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

    def is_adjective(self) -> bool:
        """형용사 여부 확인"""
        if not self.dominant_pos:
            return False
        pos = self.dominant_pos.upper()
        print(pos)
        return any(pos.startswith(p) for p in ['J', 'JJ', 'JJR', 'JJS'])

    def is_adverb(self) -> bool:
        """부사 여부 확인"""
        if not self.dominant_pos:
            return False
        pos = self.dominant_pos.upper()
        print(pos)
        return any(pos.startswith(p) for p in ['R', 'RB', 'RBR', 'RBS'])

    def is_pronoun(self) -> bool:
        """대명사 여부 확인"""
        if not self.dominant_pos:
            return False
        pos = self.dominant_pos.upper()
        print(pos)
        return pos in ['PRP', 'PRP$', 'WP', 'WP$']
    
    def copy(self) -> 'Word':
        """Word 객체 복사"""
        return Word(
            content=self.content,
            idx=self.idx,
            freq=self.freq,
            pos_tags=self.pos_tags.copy(),
            pos_counts=self.pos_counts.copy(),
            dominant_pos=self.dominant_pos,
            is_stopword=self.is_stopword,
            stopword_checked=self.stopword_checked,
            isnode=self.isnode
        )
    
    