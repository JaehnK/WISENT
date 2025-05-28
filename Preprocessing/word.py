class Word:
    TensorType = Union[np.ndarray, 'np.typing.NDArray[np.float32]']
    
    def __init__(self, content: str = None):
        self._content: Optional[str] = content
        self._idx: Optional[int] = None
        self._w2v_emb: Optional[Word.TensorType] = None  # Word2Vec 임베딩
        self._dbert_emb: Optional[Word.TensorType] = None  # DistilBERT 임베딩
        self._freq: int = 0  # 빈도수는 0으로 초기화
        self._isnode: Optional[bool] = None
        self._attention_emb: Optional[Word.TensorType] = None  # 어텐션 임베딩
        self._concat_emb: Optional[Word.TensorType] = None  # 연결된 임베딩
        
        # 품사 정보 추가
        self._pos_tags: List[str] = []  # 이 단어가 나타난 모든 품사들
        self._pos_counts: dict = {}  # 각 품사별 등장 횟수
        self._dominant_pos: Optional[str] = None  # 가장 빈번한 품사
    
    def __str__(self):
        return self._content or ""
    
    @property
    def content(self):
        return self._content
    
    @content.setter
    def content(self, word: str):
        self._content = word
    
    @property
    def idx(self):
        return self._idx
    
    @idx.setter
    def idx(self, value: int):
        self._idx = value
    
    @property
    def freq(self):
        return self._freq
    
    @property
    def pos_tags(self):
        """이 단어가 나타난 모든 품사 목록"""
        return list(self._pos_tags)
    
    @property
    def pos_counts(self):
        """각 품사별 등장 횟수"""
        return dict(self._pos_counts)
    
    @property
    def dominant_pos(self):
        """가장 빈번하게 나타나는 품사"""
        return self._dominant_pos
    
    def increment_freq(self, pos_tag: str = None):
        """빈도수 증가 및 품사 정보 업데이트"""
        self._freq += 1
        
        if pos_tag:
            # 품사 정보 업데이트
            if pos_tag not in self._pos_counts:
                self._pos_counts[pos_tag] = 0
                self._pos_tags.append(pos_tag)
            
            self._pos_counts[pos_tag] += 1
            
            # 가장 빈번한 품사 업데이트
            self._dominant_pos = max(self._pos_counts.items(), key=lambda x: x[1])[0]
    
    def get_pos_category(self):
        """품사를 주요 카테고리로 분류"""
        if not self._dominant_pos:
            return "UNKNOWN"
        
        pos = self._dominant_pos.upper()
        
        if pos.startswith('N'):
            return "NOUN"
        # 동사 (Verb)
        elif pos.startswith('V'):
            return "VERB"
        # 형용사 (Adjective)
        elif pos.startswith('J'):
            return "ADJECTIVE"
        # 부사 (Adverb)
        elif pos.startswith('R'):
            return "ADVERB"
        # 대명사 (Pronoun)
        elif pos in ['PRP', 'PRP$', 'WP', 'WP$']:
            return "PRONOUN"
        # 전치사 (Preposition)
        elif pos.startswith('IN'):
            return "PREPOSITION"
        # 접속사 (Conjunction)
        elif pos in ['CC', 'IN']:
            return "CONJUNCTION"
        # 관사 (Determiner)
        elif pos in ['DT', 'WDT']:
            return "DETERMINER"
        # 기타
        else:
            return "OTHER"

    def is_noun(self):
        pos = self._dominant_pos.upper()
        if pos.startswith("N"):
            return True
        return False
    
    def is_verb(self):
        pos = self._dominant_pos.upper()
        if pos.startswith("V"):
            return True
        return False
    
    def is_adjective(self):
        pos = self._dominant_pos.upper()
        if pos.startswith("J"):
            return True
        return False

    def is_pronoun(self):
        pos = self._dominant_pos.upper()
        if pos in ['PRP', 'PRP$', 'WP', 'WP$']:
            return True
        return False

    def is_adverb(self):
        pos = self._dominant_pos.upper()
        if pos.startswith('R'):
            return True
        return False
    