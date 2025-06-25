import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict

import spacy

from .sentense import Sentence
from .word import Word
from .trie import WordTrie
from .co_occurence import build_cooccurrence_edges


class Docs:
    def __init__(self, model_name='en_core_web_sm', disable_components=None):
        """
        Args:
            model_name: spaCy 모델 이름 (기본값: en_core_web_sm)
            disable_components: 비활성화할 파이프라인 컴포넌트 리스트
                            예: ['parser', 'ner'] - 속도 향상을 위해
        """
        self._rawdata: Optional[List[str]] = None
        self._words_list: Optional[List[Word]] = None
        self._sentence_list: Optional[List[Sentence]] = None
        self._lock = threading.Lock()
        self._word_trie = WordTrie() 
        
        try:
            if disable_components is None:
                disable_components = ['parser', 'ner']  # 속도 향상
            
            self.nlp = spacy.load(model_name, disable=disable_components)
            print(f"Loaded spaCy model: {model_name}", file=sys.stderr)
            
            # 배치 처리 최적화
            self.nlp.max_length = 2000000 
        except OSError as e:
            print(f"spaCy model '{model_name}' not found. Please install it:", file=sys.stderr)
            print(f"python -m spacy download {model_name}", file=sys.stderr)
            raise
            
    
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
    
    
    def create_sentence_list_parallel(self, batch_size:int = 100, n_process: int = None) -> None:
        """spaCy와 병렬처리를 사용한 고속 Sentence 리스트 생성"""
        if self._rawdata is None:
            self._sentence_list = None
            return
        
        print(f"Creating {len(self._rawdata)} sentences with spaCy...", file=sys.stderr)
        
        try:
            if n_process is None:
                n_process = min(4, len(self._rawdata) // 50)  # 동적 프로세스 수 결정
            
            valid_docs = [(i, doc) for i, doc in enumerate(self._rawdata) if doc.strip()]
            if not valid_docs:
                self._sentence_list = []
                self._words_list = []
                return
            
            # spaCy 병렬 처리
            texts = [doc for _, doc in valid_docs]
            
            print(f"Processing {len(texts)} documents with {n_process} processes...", file=sys.stderr)
            
            # spaCy pipe로 병렬 처리
            processed_docs = list(self.nlp.pipe(
                texts,
                batch_size=batch_size,
                n_process=n_process if n_process > 1 else 1
            ))
            
            # Sentence 객체 생성
            sentences = [None] * len(self._rawdata)  # 원본 인덱스 유지
            
            for (original_idx, _), spacy_doc in zip(valid_docs, processed_docs):
                try:
                    sentence = Sentence(docs_ref=self)
                    sentence.set_from_spacy_doc(spacy_doc, self._rawdata[original_idx])
                    sentences[original_idx] = sentence
                    
                    # 진행상황 출력
                    if original_idx % 100 == 0 and original_idx > 0:
                        print(f"Processed {original_idx}/{len(self._rawdata)} documents", file=sys.stderr)
                        
                except Exception as e:
                    print(f'Document {original_idx} processing failed: {e}', file=sys.stderr)
                    # 실패한 경우 빈 Sentence 객체 생성
                    sentence = Sentence(docs_ref=self)
                    sentence._raw = self._rawdata[original_idx]
                    sentence._lemmatised = []
                    sentence._word_objects = []
                    sentence._word_indices = []
                    sentences[original_idx] = sentence
            
            # None인 항목들 처리 (빈 문서들)
            for i, sentence in enumerate(sentences):
                if sentence is None:
                    sentence = Sentence(docs_ref=self)
                    sentence._raw = self._rawdata[i]
                    sentence._lemmatised = []
                    sentence._word_objects = []
                    sentence._word_indices = []
                    sentences[i] = sentence
            
            self._sentence_list = sentences
            
        except Exception as e:
            print(f"Parallel processing failed, falling back to sequential: {e}", file=sys.stderr)
            self._create_sentence_list_sequential()
        
        # 모든 문장 처리 완료 후 words_list 업데이트
        self._words_list = self._word_trie.get_all_words()
        print(f"Created {len(self._words_list)} unique words", file=sys.stderr)
            
        
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

    
    def get_co_occurrence_edges(self, word_to_node: Dict[str, int]):
        return build_cooccurrence_edges(word_to_node, self._sentence_list)

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
