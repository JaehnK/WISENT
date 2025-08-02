
import sys
from typing import List
import threading
import re
import spacy
import contractions

class TextProcessingService:
    """순수 텍스트 처리 서비스 (spaCy 모델 관리 및 저수준 텍스트 처리)"""
    
    def __init__(self, model_name='en_core_web_sm', disable_components=None):
        self._nlp = None
        self._model_name = model_name
        self._disable_components = disable_components or ['parser', 'ner']
        self._lock = threading.Lock()
        self._initialize_spacy()
    
    def _initialize_spacy(self):
        """spaCy 모델 초기화"""
        with self._lock:
            if self._nlp is None:
                try:
                    self._nlp = spacy.load(self._model_name, disable=self._disable_components)
                    self._nlp.max_length = 2000000
                    print(f"Loaded spaCy model: {self._model_name}", file=sys.stderr)
                except OSError:
                    print(f"spaCy model '{self._model_name}' not found. Downloading...", file=sys.stderr)
                    spacy.cli.download(self._model_name)
                    self._nlp = spacy.load(self._model_name, disable=self._disable_components)

    @property
    def nlp(self):
        return self._nlp
    
    def process_texts_to_spacy_docs(self, texts: List[str], batch_size: int = 100, n_process: int = -1) -> List:
        """텍스트 배치를 spaCy 문서 객체로 처리"""
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return []
        
        print(f"Processing {len(valid_texts)} texts with {n_process} processes...", file=sys.stderr)
        
        # spaCy의 pipe를 사용하여 효율적으로 처리
        return list(self._nlp.pipe(
            valid_texts,
            batch_size=batch_size,
            n_process=n_process
        ))

    @staticmethod
    def expand_contractions(text: str) -> str:
        """축약형을 확장하고 소문자로 변환"""
        try:
            # contractions.fix가 이미 소문자 변환을 일부 수행하지만, 일관성을 위해 .lower() 호출
            return contractions.fix(text).lower()
        except Exception:
            return text.lower()

    @staticmethod
    def is_valid_token(text: str) -> bool:
        """토큰이 유효한 단어인지 검사 (특수문자, 공백 등 제외)"""
        # 이모지 및 일반적인 단어 문자(유니코드 포함) 허용
        return bool(re.match(r'^[\w\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]+$', text))

    @staticmethod
    def convert_spacy_pos_to_nltk(spacy_pos: str, spacy_tag: str) -> str:
        """spaCy POS 태그를 NLTK 스타일(Penn Treebank)로 변환"""
        pos_mapping = {
            'ADJ': 'JJ', 'ADP': 'IN', 'ADV': 'RB', 'AUX': 'VB',
            'CONJ': 'CC', 'CCONJ': 'CC', 'DET': 'DT', 'INTJ': 'UH',
            'NOUN': 'NN', 'NUM': 'CD', 'PART': 'RP', 'PRON': 'PRP',
            'PROPN': 'NNP', 'PUNCT': '.', 'SCONJ': 'IN', 'SYM': 'SYM',
            'VERB': 'VB', 'X': 'XX', 'SPACE': 'SP'
        }
        return pos_mapping.get(spacy_pos, spacy_tag)
