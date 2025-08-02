
from typing import List, Optional

from ..entities import Documents
from .text_processing_service import TextProcessingService
from .entity_creation_service import EntityCreationService

class MainService:
    """
    전체 문서 처리 파이프라인을 총괄하는 메인 서비스 (Facade).
    """

    def __init__(self, spacy_model_name: str = 'en_core_web_sm'):
        """
        필요한 모든 하위 서비스를 초기화합니다.
        """
        self._text_processor = TextProcessingService(model_name=spacy_model_name)
        self._entity_creator = EntityCreationService(self._text_processor)
        self._documents: Optional[Documents] = None

    def process_documents(self, raw_docs: List[str]) -> Documents:
        """
        원본 텍스트 리스트를 받아 전체 처리 파이프라인을 실행하고,
        처리된 Documents 객체를 반환합니다.

        Args:
            raw_docs: 문서 원본 텍스트의 리스트.

        Returns:
            모든 처리가 완료된 Documents 객체.
        """
        # 1. Documents 객체 생성
        self._documents = Documents()
        self._documents.rawdata = raw_docs

        # 2. EntityCreationService를 사용하여 문장 및 단어 엔티티 생성
        self._entity_creator.create_sentences_from_documents(self._documents)

        print("All documents have been processed.")
        return self._documents

    def get_processed_documents(self) -> Optional[Documents]:
        """
        가장 최근에 처리된 Documents 객체를 반환합니다.
        """
        if not self._documents:
            print("Warning: No documents have been processed yet. Call process_documents() first.")
        return self._documents
