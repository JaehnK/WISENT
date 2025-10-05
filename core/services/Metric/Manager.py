"""
메트릭 레지스트리 및 계산을 담당하는 매니저
"""

from typing import Any, Dict, List, Optional

import numpy as np

from . import (
    BaseMetric,
    SilhouetteMetric,
    DaviesBouldinMetric,
    CalinskiHarabaszMetric,
    NPMIMetric,
)


class MetricManager:
    """메트릭 등록/관리 및 일괄 계산 매니저"""

    _default_instance: Optional["MetricManager"] = None

    def __init__(self, initial_metrics: Optional[List[BaseMetric]] = None) -> None:
        self._metrics: Dict[str, BaseMetric] = {}
        if initial_metrics:
            for metric in initial_metrics:
                self.register_metric(metric)

    @classmethod
    def default(cls) -> "MetricManager":
        if cls._default_instance is None:
            cls._default_instance = cls(
                initial_metrics=[
                    SilhouetteMetric(),
                    DaviesBouldinMetric(),
                    CalinskiHarabaszMetric(),
                    NPMIMetric(),
                ]
            )
        return cls._default_instance

    def register_metric(self, metric: BaseMetric) -> None:
        self._metrics[metric.name] = metric

    def unregister_metric(self, name: str) -> None:
        self._metrics.pop(name, None)

    def get_metric(self, name: str) -> Optional[BaseMetric]:
        return self._metrics.get(name)

    def available_metrics(self) -> List[str]:
        return sorted(self._metrics.keys())

    def compute(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        metric_names: List[str],
        **kwargs: Any,
    ) -> Dict[str, float]:
        # 입력 임베딩은 numpy 로 통일
        embeddings_np = (
            BaseMetric.ensure_numpy(embeddings)  # type: ignore[arg-type]
            if hasattr(BaseMetric, "ensure_numpy")
            else embeddings
        )
        results: Dict[str, float] = {}
        for name in metric_names:
            metric = self._metrics.get(name)
            if metric is None:
                print(f"Warning: Metric '{name}' not registered, skipping.")
                continue
            try:
                results[name] = metric.compute(embeddings_np, labels, **kwargs)
            except NotImplementedError:
                print(f"Warning: Metric '{name}' not implemented, returning NaN.")
                results[name] = float("nan")
            except Exception as e:
                print(f"Error computing metric '{name}': {e}")
                import traceback
                traceback.print_exc()
                results[name] = float("nan")
        return results




