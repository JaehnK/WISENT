"""
평가 지표 계산 서비스

클러스터링 품질 평가를 위한 다양한 메트릭 계산
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional

from .BaseMetric import BaseMetric
from .Manager import MetricManager


class MetricsService:
    """클러스터링 평가 지표 계산 서비스"""

    def __init__(self, manager: Optional[MetricManager] = None):
        """평가 지표 서비스 초기화 (MetricManager 주입/사용)"""
        self._manager: MetricManager = manager or MetricManager.default()

    def register_metric(self, metric: BaseMetric) -> None:
        """메트릭 인스턴스를 이름으로 등록 (매니저 위임)"""
        self._manager.register_metric(metric)

    def calculate_metrics(
        self,
        embeddings: torch.Tensor,
        labels: np.ndarray,
        metric_names: List[str],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        지정된 평가 지표들을 계산

        Args:
            embeddings: 노드 임베딩 (Tensor)
            labels: 클러스터 라벨
            metric_names: 계산할 메트릭 이름 리스트
            **kwargs: 일부 메트릭에서 요구하는 추가 파라미터 (예: NPMI)

        Returns:
            {metric_name: value} 딕셔너리
        """
        embeddings_np = (
            embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        )
        return self._manager.compute(embeddings_np, labels, metric_names, **kwargs)

    # 개별 스코어 메서드는 클래스로 대체되었으므로 제거

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        평가 지표를 포맷에 맞춰 출력

        Args:
            metrics: 계산된 지표 딕셔너리
        """
        print("\n평가 지표:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
