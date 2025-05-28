class Sentence:
    def __init__(self, docs_ref=None):
        self._raw: Optional[str] = None
        self._lemmatised: Optional[List[str]] = None
        self._word_indices: Optional[List[int]] = None
        self._word_objects: Optional[List[Word]] = None
        self._lemmatizer = WordNetLemmatizer()  # 각 인스턴스마다 생성
        self._docs_ref = docs_ref  # Docs 객체 참조
        
    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, sentence: str):
        self._raw = sentence
        self._lemmatised = None
        self._word_indices = None
        self._word_objects = None
        self.lemmatise()
        
    def _get_wordnet_pos(self, word_pos):
        """POS 태그를 WordNet 형식으로 변환"""
        tag = word_pos[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def _expand_contractions(self, text):
        """축약형을 확장하는 함수"""
        # 일반적인 축약형 매핑
        contractions_dict = {
            "don't": "do not",
            "doesn't": "does not", 
            "didn't": "did not",
            "can't": "cannot",
            "couldn't": "could not",
            "won't": "will not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "mustn't": "must not",
            "needn't": "need not",
            "daren't": "dare not",
            "mayn't": "may not",
            "oughtn't": "ought not",
            
            # be 동사
            "i'm": "i am",
            "you're": "you are", 
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "how's": "how is",
            
            # have 동사
            "i've": "i have",
            "you've": "you have",
            "we've": "we have", 
            "they've": "they have",
            "could've": "could have",
            "should've": "should have",
            "would've": "would have",
            "might've": "might have",
            "must've": "must have",
            
            # had 동사  
            "i'd": "i had",
            "you'd": "you had",
            "he'd": "he had",
            "she'd": "she had", 
            "it'd": "it had",
            "we'd": "we had",
            "they'd": "they had",
            
            # will 동사
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will", 
            "she'll": "she will",
            "it'll": "it will",
            "we'll": "we will",
            "they'll": "they will",
            "that'll": "that will",
            
            # 과거형
            "wasn't": "was not",
            "weren't": "were not",
            "isn't": "is not",
            "aren't": "are not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
        }
        
        # 소문자로 변환 후 축약형 확장
        text_lower = text.lower()
        for contraction, expansion in contractions_dict.items():
            text_lower = text_lower.replace(contraction, expansion)
        
        return text_lower
    
    def lemmatise(self):
        """문장을 표제어 형태로 변환하고 Word 객체 생성"""
        if self._raw is None:
            raise ValueError("Raw sentence is not set. Please set the raw property first.")
        
        if self._lemmatised is not None:
            return self._lemmatised
        
        try:
            # 1단계: 축약형 확장
            expanded_text = self._expand_contractions(self._raw)
            
            # 2단계: 특수문자 제거 (소유격 's 처리 개선)
            # 소유격 's를 공백으로 변환하여 분리
            cleaned_text = re.sub(r"'s\b", " s", expanded_text)  # 소유격 's 분리
            cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # 나머지 특수문자 제거
            
            # 3단계: 토큰화
            tokens = word_tokenize(cleaned_text)
            
            # 4단계: POS 태깅
            pos_tags = pos_tag(tokens)
            
            # 5단계: 표제어 변환 (모든 단어 유지)
            lemmatised_words = []
            word_objects = []
            word_indices = []
            
            for word, pos in pos_tags:
                # 너무 짧은 토큰 필터링 (단, 의미있는 단어는 유지)
                if len(word) >= 2 or word.lower() in ['i', 'a']:
                    wordnet_pos = self._get_wordnet_pos(pos)
                    lemma = self._lemmatizer.lemmatize(word, wordnet_pos)
                    lemmatised_words.append(lemma)
                    
                    # Docs 참조가 있으면 Word 객체 생성/관리 (품사 정보 포함)
                    if self._docs_ref is not None:
                        word_obj = self._docs_ref.add_word(lemma, pos)  # 품사 정보 전달
                        word_objects.append(word_obj)
                        word_indices.append(word_obj.idx)
            
            self._lemmatised = lemmatised_words
            self._word_objects = word_objects
            self._word_indices = word_indices
            
            return self._lemmatised
            
        except Exception as e:
            print(f"Lemmatisation failed for sentence: {self._raw[:50]}... Error: {e}", file=sys.stderr)
            # 실패시 빈 리스트로 초기화
            self._lemmatised = []
            self._word_objects = []
            self._word_indices = []
            return self._lemmatised
    
    @property
    def word_indices(self):
        return self._word_indices
    
    @property
    def word_objects(self):
        return self._word_objects
