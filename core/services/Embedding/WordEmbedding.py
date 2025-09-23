from core.entities.document import Documents
from core.services.DBert import BertService
from core.services.Document import DocumentService
from core.services.Word2Vec import Word2VecService


class WordEmbedding:
    def __init__(self, docs: DocumentService) -> None:
        self.documents = docs
        self.w2v = Word2VecService.create_default(docs)
        self.dbert = BertService(docs)

    def calculate_embed(self, w2v_bool:bool, dbert_bool:bool) -> None:
        if w2v_bool:
            self.w2v = self.w2v.train()
        
        if dbert_bool:
            self.dbert.train