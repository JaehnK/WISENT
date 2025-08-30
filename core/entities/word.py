from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Word:
    """단어 엔티티 - 순수한 데이터 모델
    
    이 클래스는 단어의 기본 정보와 상태만을 저장합니다.
    비즈니스 로직은 WordAnalysisService와 WordStatisticsService에서 처리합니다.
    """
    
    # 기본 정보
    content: str  # 단어의 실제 텍스트 내용
    idx: Optional[int] = None  # 단어의 고유 인덱스
    
    # 빈도 정보
    freq: int = 0  # 단어가 등장한 총 횟수
    
    # 품사 정보
    pos_tags: List[str] = field(default_factory=list)  # 이 단어가 나타난 모든 품사들
    pos_counts: Dict[str, int] = field(default_factory=dict)  # 각 품사별 등장 횟수
    _dominant_pos: Optional[str] = None  # 가장 빈번한 품사 (내부용)
    
    # 불용어 정보
    is_stopword: Optional[bool] = None  # 불용어 여부 (None=미확인)
    stopword_checked: bool = False  # 불용어 체크 완료 여부
    
    # 노드 정보 (그래프에서 사용)
    isnode: Optional[bool] = None  # 그래프 노드로 사용되는지 여부
    
    def __post_init__(self):
        """초기화 후 검증만 수행 - 순수한 데이터 검증"""
        if not self.content:
            raise ValueError("Word content cannot be empty")
        
        if self.pos_counts is None:
            self.pos_counts = {}
        if self.pos_tags is None:
            self.pos_tags = []
    
    def increment_freq(self, pos_tag: str = None):
        """빈도 수 증가 및 품사 정보 업데이트
        
        Args:
            pos_tag: 품사 태그 (선택사항)
        """
        self.freq += 1
        
        if pos_tag:
            if pos_tag not in self.pos_counts:
                self.pos_counts[pos_tag] = 0
                self.pos_tags.append(pos_tag)
            self.pos_counts[pos_tag] += 1
        
        # dominant_pos 업데이트
        if self.pos_counts:
            self._dominant_pos = max(self.pos_counts.items(), key=lambda x: x[1])[0]
    
    def set_stopword_status(self, is_stopword: bool):
        """불용어 상태를 직접 설정
        
        Args:
            is_stopword: 불용어 여부
        """
        self.is_stopword = is_stopword
        self.stopword_checked = True
    
    def get_stopword_status(self) -> bool:
        """불용어 여부 반환 (확인되지 않은 경우 False 반환)
        
        Returns:
            bool: 불용어 여부
        """
        return self.is_stopword if self.is_stopword is not None else False
    
    @property
    def dominant_pos(self) -> Optional[str]:
        """dominant_pos getter - 외부에서 접근 가능한 인터페이스"""
        return self._dominant_pos
    
    # 🚫 제거된 메서드들 (비즈니스 로직):
    # - is_noun(), is_verb(), is_adjective(), is_adverb(), is_pronoun()
    #   → WordAnalysisService로 이동
    # - get_pos_distribution() 
    #   → WordStatisticsService로 이동
    # - copy() 
    #   → 필요시 다시 추가 (데이터 복사는 엔티티의 책임일 수 있음)
    
    