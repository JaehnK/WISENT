import spacy
import fasttext
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from typing import Optional, Union, List
import numpy as np
import numpy.typing as npt
import sys
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# NLTK 데이터 다운로드 (필요시)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...", file=sys.stderr)
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet...", file=sys.stderr)
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading NLTK POS tagger...", file=sys.stderr)
    nltk.download('averaged_perceptron_tagger', quiet=True)

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

class Docs:
    def __init__(self):
        self._rawdata: Optional[List[str]] = None
        self._words_list: Optional[List[Word]] = None
        self._sentence_list: Optional[List[Sentence]] = None
        self._nlp: Optional[fasttext.FastText._FastText] = None
        self._lock = threading.Lock()
        self._word_trie = WordTrie()  # 접두사 트리
    
    @property
    def rawdata(self):
        return self._rawdata
    
    @rawdata.setter
    def rawdata(self, docs: List[str]):
        if len(docs) < 10:
            raise ValueError("Number of documents is less than 10. Not sufficient for clustering.")
        elif len(docs) < 500:
            print("Warning: Document count is low ({}). Clustering results may have reduced accuracy.".format(len(docs)), file=sys.stderr)
        else:
            print("Processing {} documents".format(len(docs)), file=sys.stderr)
        
        self._rawdata = docs
        self.create_sentence_list()
    
    def add_word(self, word_content: str, pos_tag: str = None) -> Word:
        """단어를 트리에 추가하고 Word 객체 반환 (스레드 안전)"""
        with self._lock:
            return self._word_trie.insert_or_get_word(word_content, pos_tag)
    
    def _create_sentence_with_log(self, doc_text: str, index: int) -> Sentence:
        """단일 Sentence 객체 생성 및 로그 (사용 안함 - 호환성용)"""
        sentence = Sentence(docs_ref=self)
        sentence.raw = doc_text
        return sentence

    def create_sentence_list(self, max_workers: int = 1) -> None:
        """rawdata로부터 Sentence 객체 리스트 생성"""
        if self._rawdata is None:
            self._sentence_list = None
            return

        print(f"Creating {len(self._rawdata)} sentences...", file=sys.stderr)
        
        # 단순한 순차 처리로 변경 (멀티스레딩 제거)
        sentences = []
        for i, doc_text in enumerate(self._rawdata):
            if i % 100 == 0 and i > 0:  # 100개마다 진행상황 출력
                print(f"Processed {i}/{len(self._rawdata)} documents", file=sys.stderr)
            
            try:
                sentence = Sentence(docs_ref=self)
                sentence.raw = doc_text
                sentences.append(sentence)
            except Exception as e:
                print(f'Document {i} processing failed: {e}', file=sys.stderr)
                # 실패한 경우 빈 Sentence 객체 생성
                sentence = Sentence(docs_ref=self)
                sentence._raw = doc_text
                sentence._lemmatised = []
                sentence._word_objects = []
                sentence._word_indices = []
                sentences.append(sentence)
        
        self._sentence_list = sentences
        
        # 모든 문장 처리 완료 후 words_list 업데이트
        self._words_list = self._word_trie.get_all_words()
        print(f"Created {len(self._words_list)} unique words", file=sys.stderr)

    @property
    def words_list(self):
        return self._words_list
    
    @property
    def sentence_list(self):
        return self._sentence_list
    
    def __str__(self):
        if self._rawdata is None:
            return ""
        words_count = len(self._words_list) if self._words_list else 0
        sentences_count = len(self._sentence_list) if self._sentence_list else 0
        return f"raw_data: {len(self._rawdata)}\nwords_list: {words_count}\nsentence_list: {sentences_count}"
    
    def __getitem__(self, idx):
        return self._rawdata[idx]

if __name__ == "__main__":
    # 사용 예시
    docs = Docs()
    
    # 축약형과 특수 케이스가 포함된 테스트 데이터
    test_docs = [
        "I don't think he'll come today.",
        "She can't find her keys anywhere.",
        "They won't be here until tomorrow.",
        "It's not what I've been looking for.",
        "We're going to John's house tonight.",
        "That wasn't the answer I'd expected.",
        "You shouldn't have done that.",
        "I'll be there in five minutes.",
        "The dog's tail is wagging happily.",
        "These aren't the droids you're looking for."
    ]
    
    print("🧪 TESTING CONTRACTIONS AND APOSTROPHES")
    print("="*60)
    start = time.time()
    docs.rawdata = test_docs
    end = time.time()
    print(f"{end - start:.5f} sec") 
    
    print("\n" + "="*60)
    print("DOCUMENT PROCESSING SUMMARY")
    print("="*60)
    print(docs)
    
    print("\n" + "="*60)
    print("CONTRACTION PROCESSING ANALYSIS")
    print("="*60)
    
    # 축약형이 어떻게 처리되었는지 확인
    print("Raw vs Processed comparison:")
    for i, sentence in enumerate(docs.sentence_list[:5]):
        if sentence and sentence.word_indices is not None:
            print(f"\nSentence {i+1}:")
            print(f"  📝 Raw: {sentence.raw}")
            
            # 전처리 과정 단계별 확인
            expanded = sentence._expand_contractions(sentence.raw)
            cleaned_step1 = re.sub(r"'s\b", " s", expanded)
            cleaned_final = re.sub(r'[^\w\s]', '', cleaned_step1)
            
            print(f"  🔄 Expanded: '{expanded}'")
            print(f"  🧹 Cleaned: '{cleaned_final}'")
            print(f"  🔤 Lemmatised: {sentence._lemmatised}")
            print(f"  📊 Word indices: {sentence.word_indices}")
            
            # 문제가 될 수 있는 단어들 체크
            problematic = []
            for word in sentence._lemmatised:
                if len(word) <= 1 or word in ['t', 's', 'll', 've', 're', 'm', 'd']:
                    problematic.append(word)
            
            if problematic:
                print(f"  ⚠️  Potentially problematic tokens: {problematic}")
            else:
                print(f"  ✅ No problematic tokens detected")
        else:
            print(f"\nSentence {i+1}: ❌ Processing failed")
    
    print("\n" + "="*60)
    print("PART-OF-SPEECH ANALYSIS")
    print("="*60)
    
    # 품사별 단어 분류
    pos_categories = {}
    for word in docs.words_list:
        category = word.get_pos_category()
        if category not in pos_categories:
            pos_categories[category] = []
        pos_categories[category].append(word)
    
    print("Words by POS category:")
    for category in sorted(pos_categories.keys()):
        words_in_category = pos_categories[category]
        print(f"\n📂 {category}: {len(words_in_category)} words")
        
        # 빈도순으로 정렬해서 상위 5개만 표시
        top_words = sorted(words_in_category, key=lambda w: w.freq, reverse=True)[:5]
        for word in top_words:
            pos_info = f"{word.dominant_pos}"
            if len(word.pos_counts) > 1:
                pos_info += f" (다중: {word.pos_counts})"
            print(f"    '{word.content}' - freq: {word.freq}, pos: {pos_info}")
        
        if len(words_in_category) > 5:
            print(f"    ... and {len(words_in_category) - 5} more {category.lower()}s")
    
    print(f"\n🔍 Multi-POS words (동일 단어의 다른 품사 용법):")
    multi_pos_words = [w for w in docs.words_list if len(w.pos_counts) > 1]
    if multi_pos_words:
        for word in sorted(multi_pos_words, key=lambda w: len(w.pos_counts), reverse=True)[:10]:
            print(f"  '{word.content}': {word.pos_counts} (주품사: {word.dominant_pos})")
    else:
        print("  No words with multiple POS tags found")
    
    print(f"\n📊 POS Category Distribution:")
    total_words = len(docs.words_list)
    for category in sorted(pos_categories.keys()):
        count = len(pos_categories[category])
        percentage = (count / total_words) * 100
        print(f"  {category}: {count} words ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("DETAILED WORD ANALYSIS")
    print("="*60)
    
    print("Sample detailed word information:")
    sample_words = sorted(docs.words_list, key=lambda w: w.freq, reverse=True)[:8]
    for word in sample_words:
        print(f"\n🔤 Word: '{word.content}'")
        print(f"   📊 Frequency: {word.freq}")
        print(f"   🔢 Index: {word.idx}")
        print(f"   📝 Category: {word.get_pos_category()}")
        print(f"   🏷️  Dominant POS: {word.dominant_pos}")
        print(f"   📋 All POS tags: {word.pos_counts}")
        
        # 편의 메서드 테스트
        pos_checks = []
        if word.is_noun(): pos_checks.append("NOUN")
        if word.is_verb(): pos_checks.append("VERB") 
        if word.is_adjective(): pos_checks.append("ADJECTIVE")
        if word.is_pronoun(): pos_checks.append("PRONOUN")
        if word.is_adverb(): pos_checks.append("ADVERB")
        
        if pos_checks:
            print(f"   ✅ Type checks: {', '.join(pos_checks)}")
        else:
            print(f"   ℹ️  Type: OTHER ({word.get_pos_category()})")
    
    print("\n" + "="*60)
    print("CONTRACTION DETECTION")
    print("="*60)
    
    # 원본 문장에서 축약형 패턴 찾기
    contraction_patterns = [
        "don't", "can't", "won't", "it's", "we're", "wasn't", 
        "I'd", "shouldn't", "I'll", "aren't", "you're", "he'll", "I've"
    ]
    
    found_contractions = []
    for sentence in docs.sentence_list:
        if sentence and sentence.raw:
            for pattern in contraction_patterns:
                if pattern.lower() in sentence.raw.lower():
                    found_contractions.append((pattern, sentence.raw))
    
    print(f"Found contractions in original text:")
    for contraction, sentence in found_contractions[:5]:  # 처음 5개만
        print(f"  '{contraction}' in: {sentence}")
    
    # 전체 토큰 수 vs 예상 토큰 수 비교
    total_original_tokens = sum(len(doc.split()) for doc in test_docs)
    total_processed_tokens = sum(len(s.word_indices) if s.word_indices else 0 for s in docs.sentence_list)
    
    print(f"\n📊 Token count comparison:")
    print(f"  Original word count (space-split): {total_original_tokens}")
    print(f"  Processed token count: {total_processed_tokens}")
    print(f"  Difference: {total_processed_tokens - total_original_tokens}")
    
    if total_processed_tokens > total_original_tokens:
        print("  ℹ️  More tokens after processing (contractions likely split)")
    elif total_processed_tokens < total_original_tokens:
        print("  ⚠️  Fewer tokens after processing (some tokens lost)")
    else:
        print("  ✅ Same token count")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    suspicious_words = [w for w in docs.words_list if w.content in ['t', 's', 'll', 've', 're', 'm', 'd', 'nt']]
    if suspicious_words:
        print("🚨 Issues detected with contraction handling:")
        print("   - Contractions are being split into fragments")
        print("   - Consider using a contraction expansion library")
        print("   - Or modify preprocessing to handle apostrophes better")
    else:
        print("✅ No obvious contraction handling issues detected")
    
    print("\nSuggested improvements:")
    print("1. Use contractions library: pip install contractions")
    print("2. Expand contractions before lemmatization")
    print("3. Or modify regex to preserve apostrophes in specific contexts")