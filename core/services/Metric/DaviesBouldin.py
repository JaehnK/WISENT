"""
Davies-Bouldin Score 메트릭 클래스
"""

import numpy as np

from .BaseMetric import BaseMetric


class DaviesBouldinMetric(BaseMetric):
    name = "davies_bouldin"

    def compute(self, embeddings: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        embeddings_np = self.ensure_numpy(embeddings)
        if len(np.unique(labels)) < 2:
            return float("nan")
        from sklearn.metrics import davies_bouldin_score

        return float(davies_bouldin_score(embeddings_np, labels))




