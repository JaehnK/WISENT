from typing import List
import torch
from sklearn.decomposition import PCA
from ..Document.DocumentService import DocumentService
from core.services.DBert import BertService
from core.services.Word2Vec import Word2VecService
from entities import Word


class NodeFeatureHandler:
    """
    Handles node features for graph nodes.
    """

    def __init__(self, docs: DocumentService):
        self.documents = docs
        self.w2v = Word2VecService.create_default(docs)
        self.dbert = BertService(docs)

    def calculate_embeddings(self, words: List[Word], method: str = 'concat', embed_size: int = 64) -> torch.Tensor:
        """
        단어 리스트에 대한 임베딩 계산

        Args:
            words: 임베딩을 계산할 단어들
            method: 임베딩 방법 ('concat', 'w2v', 'bert')
            embed_size: 임베딩 크기

        Returns:
            [num_words, embedding_dim] 형태의 텐서
        """
        if method not in ['concat', 'w2v', 'bert']:
            raise ValueError(f"Unsupported embedding method: {method}")

        if method == 'w2v':
            return self._get_w2v_embeddings(words, embed_size)
        elif method == 'bert':
            return self._get_bert_embeddings(words)
        elif method == 'concat':
            return self._get_concat_embeddings(words, embed_size)

    def _get_w2v_embeddings(self, words: List[Word], embed_size: int) -> torch.Tensor:
        """Word2Vec 임베딩 계산"""
        self.w2v.train()
        embeddings = []
        for word in words:
            # Word2Vec 모델에서 임베딩 추출
            embedding = self.w2v.get_word_embedding(word.content, embed_size)
            embeddings.append(embedding)
        return torch.tensor(embeddings, dtype=torch.float32)

    def _get_bert_embeddings(self, words: List[Word]) -> torch.Tensor:
        """BERT 임베딩 계산"""
        embeddings = []
        for word in words:
            # WordTrie에서 BERT 임베딩 추출
            embedding = self.dbert.get_word_embedding(word.content)
            embeddings.append(embedding)
        return torch.tensor(embeddings, dtype=torch.float32)

    def _get_concat_embeddings(self, words: List[Word], embed_size: int) -> torch.Tensor:
        """Word2Vec + BERT 연결 임베딩 (PCA로 차원 축소 후 concat)"""
        w2v_embeddings = self._get_w2v_embeddings(words, embed_size)
        bert_embeddings = self._get_bert_embeddings(words)

        # PCA를 사용해 각각을 embed_size/2 차원으로 축소
        target_dim = embed_size // 2

        # Word2Vec 차원 축소 (이미 embed_size이지만 target_dim으로 축소)
        if w2v_embeddings.shape[1] > target_dim:
            pca_w2v = PCA(n_components=target_dim)
            w2v_reduced = torch.tensor(pca_w2v.fit_transform(w2v_embeddings.numpy()), dtype=torch.float32)
        else:
            w2v_reduced = w2v_embeddings

        # BERT 차원 축소 (768 -> target_dim)
        if bert_embeddings.shape[1] > target_dim:
            pca_bert = PCA(n_components=target_dim)
            bert_reduced = torch.tensor(pca_bert.fit_transform(bert_embeddings.numpy()), dtype=torch.float32)
        else:
            bert_reduced = bert_embeddings

        # 축소된 임베딩들을 concat
        return torch.cat([w2v_reduced, bert_reduced], dim=1)
