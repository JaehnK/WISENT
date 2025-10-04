"""
GRACE (GRAph-based Clustering with Enhanced embeddings) Configuration
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class GRACEConfig:
    """GRACE 파이프라인 설정"""

    # === 데이터 로딩 설정 ===
    csv_path: str
    num_documents: int = 1000
    text_column: str = 'body'

    # === 그래프 구축 설정 ===
    top_n_words: int = 500
    exclude_stopwords: bool = True
    max_sentence_length: int = -1  # -1은 제한 없음

    # === 임베딩 설정 ===
    embedding_method: Literal['concat', 'w2v', 'bert'] = 'concat'
    embed_size: int = 64  # concat일 경우 w2v_dim + bert_dim
    w2v_dim: int = 32
    bert_dim: int = 32

    # === GraphMAE 설정 ===
    graphmae_epochs: int = 100
    graphmae_lr: float = 0.001
    graphmae_weight_decay: float = 0.0
    graphmae_device: Optional[str] = None  # None이면 자동 (CUDA 우선)
    mask_rate: float = 0.75

    # === 클러스터링 설정 ===
    clustering_method: Literal['kmeans', 'dbscan', 'hierarchical'] = 'kmeans'
    num_clusters: Optional[int] = None  # None이면 자동 탐색
    min_clusters: int = 3
    max_clusters: int = 20

    # === 평가 설정 ===
    eval_metrics: list = field(default_factory=lambda: [
        'silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi'
    ])

    # === 출력 설정 ===
    save_results: bool = True
    output_dir: str = './grace_output'
    save_graph_viz: bool = True
    save_embeddings: bool = True

    # === 디버그 설정 ===
    verbose: bool = True
    log_interval: int = 10  # epoch마다 로그 출력 간격

    @classmethod
    def create_default(cls, csv_path: str, num_documents: int = 1000) -> 'GRACEConfig':
        """기본 설정으로 생성"""
        return cls(
            csv_path=csv_path,
            num_documents=num_documents
        )

    @classmethod
    def create_for_testing(cls, csv_path: str) -> 'GRACEConfig':
        """테스트용 빠른 설정"""
        return cls(
            csv_path=csv_path,
            num_documents=100,
            top_n_words=100,
            graphmae_epochs=10,
            embed_size=32,
            w2v_dim=16,
            bert_dim=16,
            save_results=False,
            save_graph_viz=False,
            save_embeddings=False
        )

    def validate(self) -> None:
        """설정값 검증"""
        if self.embedding_method == 'concat':
            if self.embed_size != self.w2v_dim + self.bert_dim:
                raise ValueError(
                    f"concat 모드에서 embed_size({self.embed_size})는 "
                    f"w2v_dim({self.w2v_dim}) + bert_dim({self.bert_dim})와 같아야 합니다."
                )

        if self.num_clusters is not None:
            if not (self.min_clusters <= self.num_clusters <= self.max_clusters):
                raise ValueError(
                    f"num_clusters({self.num_clusters})는 "
                    f"{self.min_clusters}와 {self.max_clusters} 사이여야 합니다."
                )

        if self.top_n_words < self.max_clusters:
            raise ValueError(
                f"top_n_words({self.top_n_words})는 "
                f"max_clusters({self.max_clusters})보다 커야 합니다."
            )

    def __post_init__(self):
        """초기화 후 검증"""
        self.validate()
