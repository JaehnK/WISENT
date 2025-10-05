"""
Calinski-Harabasz Score 메트릭 클래스
"""

import numpy as np

from .BaseMetric import BaseMetric


class CalinskiHarabaszMetric(BaseMetric):
    name = "calinski_harabasz"

    def compute(self, embeddings: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        embeddings_np = self.ensure_numpy(embeddings)
        if len(np.unique(labels)) < 2:
            return float("nan")
        from sklearn.metrics import calinski_harabasz_score

        return float(calinski_harabasz_score(embeddings_np, labels))



