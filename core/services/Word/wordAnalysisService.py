from typing import List, Dict, Any

from entities import Word

class WordAnalysisService:
    """단어 분석 서비스 - 품사 분석, 분류 등
    
    이 서비스는 Word 엔티티의 품사 정보를 분석하고 분류하는 역할을 담당합니다.
    엔티티에서 제거된 품사 판별 로직들이 여기로 이동되었습니다.
    """
    
    # POS 카테고리 매핑 - 영어 품사 태그를 주요 카테고리로 분류
    POS_CATEGORIES = {
        'NOUN': ['N', 'NN', 'NNS', 'NNP', 'NNPS'],        # 명사
        'VERB': ['V', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],  # 동사
        'ADJECTIVE': ['J', 'JJ', 'JJR', 'JJS'],            # 형용사
        'ADVERB': ['R', 'RB', 'RBR', 'RBS'],               # 부사
        'PRONOUN': ['PRP', 'PRP$', 'WP', 'WP$'],          # 대명사
        'PREPOSITION': ['IN'],                              # 전치사
        'CONJUNCTION': ['CC'],                              # 접속사
        'DETERMINER': ['DT', 'WDT'],                        # 관사
    }
    
    @staticmethod
    def get_pos_category(word: Word) -> str:
        """품사를 주요 카테고리로 분류
        
        Args:
            word: 분석할 Word 엔티티
            
        Returns:
            str: 품사 카테고리 (NOUN, VERB, ADJECTIVE 등)
        """
        if not word.dominant_pos:
            return "UNKNOWN"
        
        pos = word.dominant_pos.upper()
        
        for category, pos_list in WordAnalysisService.POS_CATEGORIES.items():
            if any(pos.startswith(p) for p in pos_list):
                return category
        
        return "OTHER"
    
    @staticmethod
    def is_noun(word: Word) -> bool:
        """명사 여부 확인
        
        Args:
            word: 확인할 Word 엔티티
            
        Returns:
            bool: 명사 여부
        """
        return WordAnalysisService.get_pos_category(word) == 'NOUN'
    
    @staticmethod
    def is_verb(word: Word) -> bool:
        """동사 여부 확인"""
        if not word.dominant_pos:
            return False
        pos = word.dominant_pos.upper()
        return any(pos.startswith(p) for p in WordAnalysisService.POS_CATEGORIES['VERB'])
    
    @staticmethod
    def is_adjective(word: Word) -> bool:
        """형용사 여부 확인"""
        if not word.dominant_pos:
            return False
        pos = word.dominant_pos.upper()
        return any(pos.startswith(p) for p in WordAnalysisService.POS_CATEGORIES['ADJECTIVE'])
    
    @staticmethod
    def is_adverb(word: Word) -> bool:
        """부사 여부 확인"""
        if not word.dominant_pos:
            return False
        pos = word.dominant_pos.upper()
        return any(pos.startswith(p) for p in WordAnalysisService.POS_CATEGORIES['ADVERB'])
    
    @staticmethod
    def is_pronoun(word: Word) -> bool:
        """대명사 여부 확인"""
        if not word.dominant_pos:
            return False
        pos = word.dominant_pos.upper()
        return pos in WordAnalysisService.POS_CATEGORIES['PRONOUN']
    
    @staticmethod
    def is_content_word(word: Word) -> bool:
        """내용어 여부 확인 (명사, 동사, 형용사, 부사)"""
        return (WordAnalysisService.is_noun(word) or 
                WordAnalysisService.is_verb(word) or 
                WordAnalysisService.is_adjective(word) or 
                WordAnalysisService.is_adverb(word))
    
    # @staticmethod
    # def check_stopword_with_spacy(word: Word, nlp_model) -> bool:
    #     """spaCy 모델을 사용해 불용어 여부 확인"""
    #     if word.stopword_checked:
    #         return word.get_stopword_status()
        
    #     if not word.content:
    #         is_stopword = False
    #     else:
    #         try:
    #             doc = nlp_model(word.content)
    #             is_stopword = len(doc) > 0 and doc[0].is_stop
    #         except:
    #             is_stopword = False
        
    #     word.set_stopword_status(is_stopword)
    #     return is_stopword
    
    @staticmethod
    def get_detailed_stats(word: Word) -> Dict[str, Any]:
        """상세 통계 정보 (분석 포함)"""
        basic_stats = word.get_basic_stats()
        
        analysis_stats = {
            'pos_category': WordAnalysisService.get_pos_category(word),
            'is_noun': WordAnalysisService.is_noun(word),
            'is_verb': WordAnalysisService.is_verb(word),
            'is_adjective': WordAnalysisService.is_adjective(word),
            'is_adverb': WordAnalysisService.is_adverb(word),
            'is_pronoun': WordAnalysisService.is_pronoun(word),
            'is_content_word': WordAnalysisService.is_content_word(word),
            'pos_distribution': word.get_pos_distribution()
        }
        
        return {**basic_stats, **analysis_stats}
    
    @staticmethod
    def filter_words_by_pos(words: List[Word], 
                        pos_categories: List[str] = None,
                        exclude_stopwords: bool = True) -> List[Word]:
        """품사별 단어 필터링"""
        if pos_categories is None:
            pos_categories = ['NOUN', 'VERB', 'ADJECTIVE']
        
        filtered = []
        for word in words:
            # 불용어 제외
            if exclude_stopwords and word.get_stopword_status():
                continue
            
            # 품사 필터링
            word_category = WordAnalysisService.get_pos_category(word)
            if word_category in pos_categories:
                filtered.append(word)
        
        return filtered
    
    @staticmethod
    def get_words_by_frequency_range(words: List[Word], 
                                    min_freq: int = 1, 
                                    max_freq: int = None) -> List[Word]:
        """빈도 범위별 단어 필터링"""
        if max_freq is None:
            max_freq = float('inf')
        
        return [word for word in words 
                if min_freq <= word.freq <= max_freq]