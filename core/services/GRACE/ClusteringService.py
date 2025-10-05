"""
클러스터링 서비스

K-means 및 기타 클러스터링 알고리즘 수행
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from datetime import datetime


class ClusteringService:
    """클러스터링 서비스"""

    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: 재현성을 위한 랜덤 시드
        """
        self.random_state = random_state
        self.cluster_labels: Optional[np.ndarray] = None
        self.inertias: Optional[List[float]] = None
        self.silhouette_scores: Optional[List[float]] = None
        self.best_k: Optional[int] = None

    def kmeans_clustering(
        self,
        embeddings: torch.Tensor,
        n_clusters: int,
        n_init: int = 10
    ) -> np.ndarray:
        """
        K-means 클러스터링 수행

        Args:
            embeddings: 노드 임베딩 (Tensor)
            n_clusters: 클러스터 수
            n_init: 초기화 횟수

        Returns:
            클러스터 라벨 배열
        """
        from sklearn.cluster import KMeans

        embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=n_init
        )
        self.cluster_labels = kmeans.fit_predict(embeddings_np)

        return self.cluster_labels

    def auto_clustering_elbow(
        self,
        embeddings: torch.Tensor,
        min_clusters: int = 3,
        max_clusters: int = 20,
        n_init: int = 10
    ) -> Tuple[np.ndarray, int, List[float], List[float]]:
        """
        Elbow Method로 최적 클러스터 수 탐색 후 클러스터링

        Args:
            embeddings: 노드 임베딩 (Tensor)
            min_clusters: 최소 클러스터 수
            max_clusters: 최대 클러스터 수
            n_init: 초기화 횟수

        Returns:
            (cluster_labels, best_k, inertias, silhouette_scores)
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

        self.inertias = []
        self.silhouette_scores = []
        k_range = range(min_clusters, max_clusters + 1)

        # 각 k에 대해 클러스터링 수행
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=n_init)
            labels = kmeans.fit_predict(embeddings_np)
            self.inertias.append(kmeans.inertia_)
            self.silhouette_scores.append(silhouette_score(embeddings_np, labels))

        # Elbow point 찾기
        self.best_k = self._find_elbow_point(list(k_range), self.inertias)

        # 최적 k로 최종 클러스터링
        kmeans = KMeans(n_clusters=self.best_k, random_state=self.random_state, n_init=n_init)
        self.cluster_labels = kmeans.fit_predict(embeddings_np)

        return self.cluster_labels, self.best_k, self.inertias, self.silhouette_scores

    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """
        Elbow Method로 최적 클러스터 수 찾기

        Args:
            k_values: 클러스터 수 리스트
            inertias: 각 k에 대한 inertia 값

        Returns:
            최적 클러스터 수
        """
        if len(k_values) < 3:
            return k_values[0]

        # 정규화된 inertia 계산
        inertias_norm = np.array(inertias)
        inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min())

        # 2차 미분 (곡률) 계산
        second_derivative = np.diff(inertias_norm, 2)

        # 곡률이 최대인 지점 = elbow point
        elbow_idx = np.argmax(second_derivative) + 1  # diff로 인한 인덱스 조정

        return k_values[elbow_idx]

    def get_cluster_distribution(self) -> Dict[int, int]:
        """
        클러스터별 데이터 포인트 수 반환

        Returns:
            {cluster_id: count} 딕셔너리
        """
        if self.cluster_labels is None:
            raise RuntimeError("클러스터링이 수행되지 않았습니다.")

        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def save_elbow_curve(
        self,
        k_values: List[int],
        output_path: str
    ) -> None:
        """
        Elbow curve 시각화 저장

        Args:
            k_values: 클러스터 수 리스트
            output_path: 저장 경로
        """
        if self.inertias is None or self.silhouette_scores is None or self.best_k is None:
            raise RuntimeError("auto_clustering_elbow()를 먼저 수행해야 합니다.")

        import matplotlib.pyplot as plt

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Inertia plot (Elbow curve)
        ax1.plot(k_values, self.inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=self.best_k, color='r', linestyle='--', linewidth=2, label=f'Elbow Point (k={self.best_k})')
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Silhouette score plot
        ax2.plot(k_values, self.silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=self.best_k, color='r', linestyle='--', linewidth=2, label=f'Selected k={self.best_k}')
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
