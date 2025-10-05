"""
Silhouette Score 메트릭 클래스
"""

import numpy as np

from .BaseMetric import BaseMetric


class SilhouetteMetric(BaseMetric):
    name = "silhouette"

    def compute(self, embeddings: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        embeddings_np = self.ensure_numpy(embeddings)
        # 군집이 2개 미만이면 정의되지 않으므로 NaN 반환
        if len(np.unique(labels)) < 2:
            return float("nan")
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(embeddings_np, labels))




