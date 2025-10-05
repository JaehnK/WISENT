from typing import Optional, List, Dict, Tuple, Any, Union
import numpy as np
import torch
from torch_geometric.data import Data
from dataclasses import dataclass, field
from enum import Enum

from .word import Word

class NodeFeatureType(Enum):
    """노드 특성 타입"""
    FREQUENCY = "frequency"           # 단어 빈도 (스칼라)
    BERT_EMBEDDING = "bert_embedding" # BERT 임베딩 (768차원)
    WORD2VEC = "word2vec"            # Word2Vec 임베딩
    GRAPHMAE = "graphmae"            # GraphMAE 사전훈련 임베딩
    CUSTOM = "custom"                # 사용자 정의


class EdgeFeatureType(Enum):
    """엣지 특성 타입"""
    CO_OCCURRENCE = "co_occurrence"           # 공출현 빈도
    SIMILARITY = "similarity"                # 유사도 (w2v, bert 등)
    COMBINED = "combined"                    # 공출현 + 유사도 조합
    CUSTOM = "custom"  
    
    
@dataclass
class GraphMetadata:
    """그래프 메타데이터"""
    num_nodes: int
    num_edges: int
    node_feature_dim: int
    edge_feature_dim: int
    node_feature_type: NodeFeatureType
    edge_feature_type: EdgeFeatureType
    creation_timestamp: float = field(default_factory=lambda: __import__('time').time())
    description: str = ""
    
    def to_dict(self):
        return {
                'num_nodes' : self.num_nodes,
                'num_edges' : self.num_edges,
                'node_feature_dim' : self.node_feature_dim,
                'edge_feature_dim' : self.edge_feature_dim,
                'node_feature_type' : self.node_feature_type,
                'edge_feature_type' : self.edge_feature_type,
                'creation_timestamp' : self.creation_timestamp,
                'description' : self.description
        }

class WordGraph:
    def __init__(self,
                words: List[Word],
                word_to_node_id: Optional[Dict[str, int]] = None):
        """
            Args:
                words: 그래프 노드가 될 단어들
                word_to_node_id: 단어 → 노드 ID 매핑 (None이면 자동 생성)
        """
        # 기본 데이터
        self._words = words
        self._word_to_node_id = word_to_node_id or {word.content: i for i, word in enumerate(words)}
        self._node_id_to_word = {i: word for word, i in self._word_to_node_id.items()}
        
        # 그래프 구조
        self._edge_index: Optional[torch.Tensor] = None  # [2, num_edges]
        self._edge_attr: Optional[torch.Tensor] = None   # [num_edges, edge_feature_dim]
        
        # 노드 특성
        self._node_features: Optional[torch.Tensor] = None  # [num_nodes, node_feature_dim]
        self._node_feature_type: NodeFeatureType = NodeFeatureType.FREQUENCY
        
        # 엣지 특성 정보
        self._edge_feature_type: EdgeFeatureType = EdgeFeatureType.COMBINED
        
        # 메타데이터
        self._metadata: Optional[GraphMetadata] = None
        
        # 기본 노드 특성 초기화 (빈도 기반)
        self._initialize_frequency_features()
    
        # === Properties ===
    
    @property
    def num_nodes(self) -> int:
        return len(self._words)
    
    @property
    def num_edges(self) -> int:
        return self._edge_index.shape[1] if self._edge_index is not None else 0
    
    @property
    def words(self) -> List[Word]:
        return self._words.copy()
    
    @property
    def word_to_node_id(self) -> Dict[str, int]:
        return self._word_to_node_id.copy()
    
    @property
    def edge_index(self) -> Optional[torch.Tensor]:
        return self._edge_index
    
    @property
    def edge_attr(self) -> Optional[torch.Tensor]:
        return self._edge_attr
    
    @property
    def node_features(self) -> Optional[torch.Tensor]:
        return self._node_features
    
    @property
    def node_feature_type(self) -> NodeFeatureType:
        return self._node_feature_type
    
    @property
    def edge_feature_type(self) -> EdgeFeatureType:
        return self._edge_feature_type
    
    @property
    def metadata(self) -> Optional[GraphMetadata]:
        return self._metadata
    
    # === 노드 특성 설정 ===
    
    def set_node_features_from_frequency(self) -> None:
        """빈도 기반 노드 특성 설정"""
        self._initialize_frequency_features()
        self._node_feature_type = NodeFeatureType.FREQUENCY
        self._update_metadata()
    
    def set_node_features_from_bert(self, bert_embeddings: Union[torch.Tensor, np.ndarray]) -> None:
        """
        BERT 임베딩으로 노드 특성 설정
        
        Args:
            bert_embeddings: [num_nodes, 768] 형태의 BERT 임베딩
        """
        if isinstance(bert_embeddings, np.ndarray):
            bert_embeddings = torch.tensor(bert_embeddings, dtype=torch.float32)
        
        if bert_embeddings.shape[0] != self.num_nodes:
            raise ValueError(f"BERT embeddings size mismatch: expected {self.num_nodes}, got {bert_embeddings.shape[0]}")
        
        if bert_embeddings.shape[1] != 768:
            raise ValueError(f"BERT embeddings should be 768-dimensional, got {bert_embeddings.shape[1]}")
        
        self._node_features = bert_embeddings
        self._node_feature_type = NodeFeatureType.BERT_EMBEDDING
        self._update_metadata()
        
    def set_node_features_custom(self, 
                                features: Union[torch.Tensor, np.ndarray],
                                feature_type: NodeFeatureType = NodeFeatureType.CUSTOM) -> None:
        """
        사용자 정의 노드 특성 설정
        
        Args:
            features: [num_nodes, feature_dim] 형태의 특성
            feature_type: 특성 타입
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        
        if features.shape[0] != self.num_nodes:
            raise ValueError(f"Feature size mismatch: expected {self.num_nodes}, got {features.shape[0]}")
        
        self._node_features = features
        self._node_feature_type = feature_type
        self._update_metadata()
        
    def set_edges_from_co_occurrence(self,
                                    edge_list: List[Tuple[int, int]],
                                    co_occurrence_weights: List[float]) -> None:
        """
        공출현 기반 엣지 설정

        Args:
            edge_list: [(src_node_id, dst_node_id), ...] 형태의 엣지 리스트
            co_occurrence_weights: 공출현 가중치 리스트
        """
        if len(edge_list) != len(co_occurrence_weights):
            raise ValueError("Edge list and weights must have same length")

        # 빈 엣지 리스트 처리
        if len(edge_list) == 0:
            self._edge_index = torch.empty((2, 0), dtype=torch.long)
            self._edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            # PyTorch Geometric 형태로 변환
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2, num_edges]
            edge_attr = torch.tensor(co_occurrence_weights, dtype=torch.float32).unsqueeze(1)  # [num_edges, 1]

            self._edge_index = edge_index
            self._edge_attr = edge_attr

        self._edge_feature_type = EdgeFeatureType.CO_OCCURRENCE
        self._update_metadata()
    
    def set_edges_combined(self, 
                            edge_list: List[Tuple[int, int]], 
                            co_occurrence_weights: List[float],
                            similarity_scores: List[float]) -> None:
        """
        공출현 + 유사도 조합 엣지 설정
        
        Args:
            edge_list: [(src_node_id, dst_node_id), ...] 형태의 엣지 리스트
            co_occurrence_weights: 공출현 가중치 리스트
            similarity_scores: 유사도 점수 리스트 (Word2Vec, BERT 등)
        """
        if not (len(edge_list) == len(co_occurrence_weights) == len(similarity_scores)):
            raise ValueError("Edge list, co-occurrence weights, and similarity scores must have same length")
        
        # PyTorch Geometric 형태로 변환
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2, num_edges]
        
        # 두 특성을 결합 [num_edges, 2] --> 추후 알파 베타 값으로 수정 필요
        edge_attr = torch.stack([
            torch.tensor(co_occurrence_weights, dtype=torch.float32),
            torch.tensor(similarity_scores, dtype=torch.float32)
        ], dim=1)
        
        self._edge_index = edge_index
        self._edge_attr = edge_attr
        self._edge_feature_type = EdgeFeatureType.COMBINED
        self._update_metadata()
    
    def set_edges_custom(self, 
                        edge_index: Union[torch.Tensor, np.ndarray],
                        edge_attr: Union[torch.Tensor, np.ndarray],
                        feature_type: EdgeFeatureType = EdgeFeatureType.CUSTOM) -> None:
        """
        사용자 정의 엣지 설정
        
        Args:
            edge_index: [2, num_edges] 형태의 엣지 인덱스
            edge_attr: [num_edges, edge_feature_dim] 형태의 엣지 특성
            feature_type: 엣지 특성 타입
        """
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        if isinstance(edge_attr, np.ndarray):
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        
        if edge_index.shape[1] != edge_attr.shape[0]:
            raise ValueError("Number of edges in edge_index and edge_attr must match")
        
        self._edge_index = edge_index
        self._edge_attr = edge_attr
        self._edge_feature_type = feature_type
        self._update_metadata()
    
    def to_pytorch_geometric(self) -> Data:
        """PyTorch Geometric Data 객체로 변환"""
        if self._node_features is None:
            raise ValueError("Node features not set")
        if self._edge_index is None:
            raise ValueError("Edges not set")
        
        return Data(
            x=self._node_features,
            edge_index=self._edge_index,
            edge_attr=self._edge_attr,
            num_nodes=self.num_nodes
        )
    
    def export_to_networkx(self, include_weights: bool = True) -> 'nx.Graph':
        """NetworkX 그래프로 변환"""
        import networkx as nx
        
        if self.edge_index is None or self.words is None:
            raise ValueError("Graph not constructed.")
        
        G = nx.Graph()
        
        # 노드 추가 (단어 정보 포함)
        for i, word in enumerate(self.words):
            G.add_node(i, 
                    content=word.content,
                    frequency=word.freq,
                    pos=word.dominant_pos)
        
        # 엣지 추가
        edges_np = self.edge_index.numpy()
        
        if self.edge_attr is not None:
            weights_np = self.edge_attr.numpy()
            for i, (src, dst) in enumerate(edges_np.T):
                if include_weights:
                    weight = weights_np[i] if weights_np.ndim > 1 else weights_np[i, 0]
                    G.add_edge(src, dst, weight=float(weight))
                else:
                    G.add_edge(src, dst)
        else:
            for src, dst in edges_np.T:
                G.add_edge(src, dst)
        
        return G
    
    def get_word_by_node_id(self, node_id: int) -> Word:
        """노드 ID로 단어 조회"""
        if node_id >= self.num_nodes or node_id < 0:
            raise IndexError(f"Node ID {node_id} out of range")
        return self._words[node_id]
    
    def get_node_id_by_word(self, word_content: str) -> Optional[int]:
        """단어로 노드 ID 조회"""
        return self._word_to_node_id.get(word_content)
    
    def get_edge_between_words(self, word1: str, word2: str) -> Optional[torch.Tensor]:
        """두 단어 사이의 엣지 특성 조회"""
        node1_id = self.get_node_id_by_word(word1)
        node2_id = self.get_node_id_by_word(word2)
        
        if node1_id is None or node2_id is None:
            return None
        
        if self._edge_index is None or self._edge_attr is None:
            return None
        
        # 엣지 찾기
        edges = self._edge_index.t()  # [num_edges, 2]
        for i, (src, dst) in enumerate(edges):
            if (src == node1_id and dst == node2_id) or (src == node2_id and dst == node1_id):
                return self._edge_attr[i]
        
        return None
    
    def get_node_neighbors(self, node_id: int) -> List[int]:
        """노드의 이웃 노드들 반환"""
        if self._edge_index is None:
            return []
        
        neighbors = []
        edges = self._edge_index.t()  # [num_edges, 2]
        
        for src, dst in edges:
            if src == node_id:
                neighbors.append(int(dst))
            elif dst == node_id:
                neighbors.append(int(src))
        
        return list(set(neighbors))  # 중복 제거
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """그래프 통계 정보"""
        stats = {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'node_feature_dim': self._node_features.shape[1] if self._node_features is not None else 0,
            'edge_feature_dim': self._edge_attr.shape[1] if self._edge_attr is not None else 0,
            'node_feature_type': self._node_feature_type.value,
            'edge_feature_type': self._edge_feature_type.value,
            'density': 0,
            'avg_degree': 0
        }
        
        if self.num_edges > 0 and self.num_nodes > 1:
            max_edges = self.num_nodes * (self.num_nodes - 1) // 2  # 무방향 그래프
            stats['density'] = self.num_edges / max_edges
            stats['avg_degree'] = (2 * self.num_edges) / self.num_nodes
        
        return stats
    
    def save_to_disk(self, filepath: str) -> None:
        """그래프를 디스크에 저장"""
        save_data = {
            'words': [{'content': w.content, 'freq': w.freq, 'idx': w.idx} for w in self._words],
            'word_to_node_id': self._word_to_node_id,
            'edge_index': self._edge_index.numpy() if self._edge_index is not None else None,
            'edge_attr': self._edge_attr.numpy() if self._edge_attr is not None else None,
            'node_features': self._node_features.numpy() if self._node_features is not None else None,
            'node_feature_type': self._node_feature_type.value,
            'edge_feature_type': self._edge_feature_type.value,
            'metadata': self._metadata.to_dict() if self._metadata else None
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    @classmethod
    def load_from_disk(cls, filepath: str) -> 'WordGraph':
        """디스크에서 그래프 로드"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Word 객체 재구성
        words = [Word(content=w['content']) for w in data['words']]
        for i, w_data in enumerate(data['words']):
            words[i].freq = w_data['freq']
            words[i].idx = w_data['idx']
        
        # 그래프 객체 생성
        graph = cls(words, data['word_to_node_id'])
        
        # 데이터 복원
        if data['edge_index'] is not None:
            graph._edge_index = torch.tensor(data['edge_index'])
        if data['edge_attr'] is not None:
            graph._edge_attr = torch.tensor(data['edge_attr'])
        if data['node_features'] is not None:
            graph._node_features = torch.tensor(data['node_features'])
        
        graph._node_feature_type = NodeFeatureType(data['node_feature_type'])
        graph._edge_feature_type = EdgeFeatureType(data['edge_feature_type'])
        
        return graph
    
    # === Private Methods ===
    
    def _initialize_frequency_features(self) -> None:
        """빈도 기반 노드 특성 초기화"""
        frequencies = torch.tensor([word.freq for word in self._words], dtype=torch.float32)
        self._node_features = frequencies.unsqueeze(1)  # [num_nodes, 1]
    
    def _update_metadata(self) -> None:
        """메타데이터 업데이트"""
        node_feature_dim = self._node_features.shape[1] if self._node_features is not None else 0
        edge_feature_dim = self._edge_attr.shape[1] if self._edge_attr is not None else 0
        
        self._metadata = GraphMetadata(
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            node_feature_type=self._node_feature_type,
            edge_feature_type=self._edge_feature_type
        )
    
    def __str__(self) -> str:
        return (f"WordGraph(nodes={self.num_nodes}, edges={self.num_edges}, "
                f"node_feat={self._node_feature_type.value}, edge_feat={self._edge_feature_type.value})")
    
    def __repr__(self) -> str:
        return self.__str__()