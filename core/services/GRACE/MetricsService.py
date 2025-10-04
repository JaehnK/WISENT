"""
평가 지표 계산 서비스

클러스터링 품질 평가를 위한 다양한 메트릭 계산
"""

import numpy as np
import torch
from typing import Dict, List, Optional


class MetricsService:
    """클러스터링 평가 지표 계산 서비스"""

    def __init__(self):
        """평가 지표 서비스 초기화"""
        pass

    def calculate_metrics(
        self,
        embeddings: torch.Tensor,
        labels: np.ndarray,
        metric_names: List[str]
    ) -> Dict[str, float]:
        """
        지정된 평가 지표들을 계산

        Args:
            embeddings: 노드 임베딩 (Tensor)
            labels: 클러스터 라벨
            metric_names: 계산할 메트릭 이름 리스트
                - 'silhouette': Silhouette Score
                - 'davies_bouldin': Davies-Bouldin Score
                - 'calinski_harabasz': Calinski-Harabasz Score

        Returns:
            {metric_name: value} 딕셔너리
        """
        embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

        results = {}

        if 'silhouette' in metric_names:
            results['silhouette'] = self.silhouette_score(embeddings_np, labels)

        if 'davies_bouldin' in metric_names:
            results['davies_bouldin'] = self.davies_bouldin_score(embeddings_np, labels)

        if 'calinski_harabasz' in metric_names:
            results['calinski_harabasz'] = self.calinski_harabasz_score(embeddings_np, labels)

        # TODO: NPMI 계산 추가 (공출현 정보 필요)
        if 'npmi' in metric_names:
            # results['npmi'] = self.npmi_score(...)
            pass

        return results

    def silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Silhouette Score 계산

        Args:
            embeddings: 노드 임베딩
            labels: 클러스터 라벨

        Returns:
            Silhouette Score (높을수록 좋음, -1 ~ 1)
        """
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(embeddings, labels))

    def davies_bouldin_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Davies-Bouldin Score 계산

        Args:
            embeddings: 노드 임베딩
            labels: 클러스터 라벨

        Returns:
            Davies-Bouldin Score (낮을수록 좋음, 0 ~)
        """
        from sklearn.metrics import davies_bouldin_score
        return float(davies_bouldin_score(embeddings, labels))

    def calinski_harabasz_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Calinski-Harabasz Score (Variance Ratio Criterion) 계산

        Args:
            embeddings: 노드 임베딩
            labels: 클러스터 라벨

        Returns:
            Calinski-Harabasz Score (높을수록 좋음)
        """
        from sklearn.metrics import calinski_harabasz_score
        return float(calinski_harabasz_score(embeddings, labels))

    def npmi_score(
        self,
        cluster_words: Dict[int, List[str]],
        cooccurrence_matrix: np.ndarray,
        word_to_idx: Dict[str, int],
        total_docs: int
    ) -> float:
        """
        Normalized Pointwise Mutual Information (NPMI) 계산

        클러스터 내 단어들의 의미적 일관성(coherence) 측정

        Args:
            cluster_words: {cluster_id: [word1, word2, ...]} 딕셔너리
            cooccurrence_matrix: 단어 공출현 행렬
            word_to_idx: 단어 -> 인덱스 매핑
            total_docs: 전체 문서 수

        Returns:
            평균 NPMI score (높을수록 좋음, -1 ~ 1)
        """
        # TODO: 공출현 행렬 기반 NPMI 계산 로직 구현
        raise NotImplementedError("NPMI 계산은 아직 구현되지 않았습니다.")

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        평가 지표를 포맷에 맞춰 출력

        Args:
            metrics: 계산된 지표 딕셔너리
        """
        print("\n평가 지표:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
