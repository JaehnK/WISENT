import sys
from typing import List, Tuple

from ..entities import Documents, Sentence, Word
from .text_processing_service import TextProcessingService

class EntityCreationService:
    """텍스트를 처리하고 엔티티(Sentence, Word)를 생성하는 서비스"""

    def __init__(self, text_processor: TextProcessingService):
        self._text_processor = text_processor

    def create_sentences_from_documents(self, documents: Documents) -> None:
        """Documents 객체에 있는 원본 텍스트를 처리하여 Sentence 리스트를 생성하고 채워넣습니다."""
        raw_docs = documents.rawdata
        if not raw_docs:
            documents.sentence_list = []
            return

        print(f"Starting sentence creation for {len(raw_docs)} documents...", file=sys.stderr)

        # 문서와 인덱스를 함께 처리하기 위해 튜플로 묶음
        docs_with_indices = list(enumerate(raw_docs))
        
        # 유효한(비어있지 않은) 문서만 필터링
        valid_docs_with_indices = [(i, doc) for i, doc in docs_with_indices if doc and doc.strip()]
        valid_texts = [doc for _, doc in valid_docs_with_indices]

        if not valid_texts:
            documents.sentence_list = [self._create_empty_sentence(doc, i) for i, doc in docs_with_indices]
            return

        # 텍스트 처리 서비스로 spaCy 문서 객체 생성
        spacy_docs = self._text_processor.process_texts_to_spacy_docs(valid_texts)

        # 최종 결과를 담을 리스트 (원본 순서 유지)
        all_sentences: List[Sentence] = [None] * len(raw_docs)

        # spaCy 처리 결과와 원본 인덱스를 매핑하여 Sentence 객체 생성
        processed_idx = 0
        for original_idx, original_text in docs_with_indices:
            if original_text and original_text.strip():
                if processed_idx < len(spacy_docs):
                    spacy_doc = spacy_docs[processed_idx]
                    sentence = self._create_sentence_from_spacy_doc(original_text, original_idx, spacy_doc, documents)
                    all_sentences[original_idx] = sentence
                    processed_idx += 1
                else:
                    # spaCy 처리 중 실패한 경우
                    all_sentences[original_idx] = self._create_empty_sentence(original_text, original_idx)
            else:
                # 원본 문서가 비어있는 경우
                all_sentences[original_idx] = self._create_empty_sentence(original_text, original_idx)

        documents.sentence_list = all_sentences
        print(f"Sentence creation completed.", file=sys.stderr)

    def _create_sentence_from_spacy_doc(self, raw_text: str, doc_id: int, spacy_doc, documents: Documents) -> Sentence:
        """spaCy Doc에서 단일 Sentence 객체를 생성합니다."""
        sentence = Sentence(raw=raw_text, doc_id=str(doc_id))
        
        lemmatised_words = []
        word_objects = []
        word_indices = []
        pos_tags = []

        try:
            for token in spacy_doc:
                # 유효 토큰 검사 (TextProcessingService의 static method 사용)
                if not token.is_punct and not token.is_space and self._text_processor.is_valid_token(token.text):
                    lemma = token.lemma_.lower()
                    pos_tag = self._text_processor.convert_spacy_pos_to_nltk(token.pos_, token.tag_)

                    lemmatised_words.append(lemma)
                    pos_tags.append(pos_tag)

                    # Word 객체 생성 및 Documents의 Trie에 추가
                    word_obj = documents.add_word(lemma, pos_tag)
                    if not word_obj.stopword_checked:
                        word_obj.set_stopword_status(token.is_stop)
                    
                    word_objects.append(word_obj)
                    word_indices.append(word_obj.idx)

            sentence.set_processed_data(lemmatised_words, word_objects, word_indices, pos_tags)

        except Exception as e:
            error_msg = f"spaCy processing failed: {e}"
            sentence.add_processing_error(error_msg)
            print(f"❌ Error processing document {doc_id}: {error_msg}", file=sys.stderr)

        return sentence

    def _create_empty_sentence(self, raw_text: str, doc_id: int) -> Sentence:
        """비어 있거나 처리 실패한 경우를 위한 빈 Sentence 객체 생성"""
        sentence = Sentence(raw=raw_text, doc_id=str(doc_id))
        sentence.is_processed = True # 비어있지만 "처리됨"으로 간주
        return sentence