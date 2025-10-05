"""
NPMI (Normalized Pointwise Mutual Information) 메트릭
클러스터 내 단어들의 공출현 기반 응집도 측정
"""

from typing import Any, Dict

import numpy as np

from .BaseMetric import BaseMetric


class NPMIMetric(BaseMetric):
    name = "npmi"

    def compute(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        **kwargs: Dict[str, Any],
    ) -> float:
        """
        NPMI 계산

        Args:
            embeddings: 사용되지 않음 (인터페이스 호환성)
            labels: 클러스터 라벨 [num_nodes]
            **kwargs:
                - word_graph: WordGraph 객체 (공출현 정보 포함)
                - total_docs: 전체 문서 수

        Returns:
            클러스터별 NPMI 평균값 (높을수록 좋음, 범위: [-1, 1])
        """
        word_graph = kwargs.get('word_graph')
        total_docs = kwargs.get('total_docs')

        if word_graph is None:
            raise ValueError("NPMI requires 'word_graph' in kwargs")
        if total_docs is None or total_docs == 0:
            raise ValueError("NPMI requires 'total_docs' > 0 in kwargs")

        # 공출현 행렬 구성 (edge_index, edge_attr 활용)
        cooc_matrix = self._build_cooc_matrix(word_graph)

        # 각 클러스터별 NPMI 계산 후 평균
        cluster_npmis = []
        unique_labels = np.unique(labels)

        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) < 2:
                continue  # 단어 1개 이하인 클러스터는 스킵

            npmi = self._compute_cluster_npmi(
                cluster_indices, cooc_matrix, word_graph, total_docs
            )
            cluster_npmis.append(npmi)

        return float(np.mean(cluster_npmis)) if cluster_npmis else 0.0

    def _build_cooc_matrix(self, word_graph) -> np.ndarray:
        """WordGraph에서 공출현 행렬 구성"""
        num_nodes = word_graph.num_nodes
        cooc_matrix = np.zeros((num_nodes, num_nodes))

        if word_graph.edge_index is None or word_graph.edge_attr is None:
            return cooc_matrix

        edge_index = word_graph.edge_index.numpy()  # [2, num_edges]
        edge_weights = word_graph.edge_attr.numpy().flatten()  # [num_edges]

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            weight = edge_weights[i]
            cooc_matrix[src, dst] = weight
            cooc_matrix[dst, src] = weight  # 대칭 행렬

        return cooc_matrix

    def _compute_cluster_npmi(
        self,
        cluster_indices: np.ndarray,
        cooc_matrix: np.ndarray,
        word_graph,
        total_docs: int
    ) -> float:
        """
        클러스터 내 단어 쌍들의 NPMI 평균 계산

        NPMI(w_i, w_j) = PMI(w_i, w_j) / -log(P(w_i, w_j))
        PMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) * P(w_j)))
        """
        npmis = []

        # 클러스터 내 모든 단어 쌍 순회
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                idx_i = cluster_indices[i]
                idx_j = cluster_indices[j]

                # 공출현 빈도
                co_freq = cooc_matrix[idx_i, idx_j]
                if co_freq == 0:
                    continue  # 공출현하지 않은 쌍은 스킵

                # 개별 단어 빈도
                freq_i = word_graph.words[idx_i].freq
                freq_j = word_graph.words[idx_j].freq

                # 확률 계산 (스무딩 적용)
                eps = 1e-10
                p_i = (freq_i + eps) / (total_docs + eps)
                p_j = (freq_j + eps) / (total_docs + eps)
                p_ij = (co_freq + eps) / (total_docs + eps)

                # PMI 계산
                pmi = np.log(p_ij / (p_i * p_j))

                # NPMI 계산 (-log(p_ij)로 정규화)
                npmi = pmi / (-np.log(p_ij))
                npmis.append(npmi)

        return float(np.mean(npmis)) if npmis else 0.0



