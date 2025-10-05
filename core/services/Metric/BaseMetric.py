"""
메트릭 기본 베이스 클래스 정의
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch


class BaseMetric(ABC):
    """모든 개별 메트릭이 상속하는 베이스 클래스"""

    name: str

    def __init__(self, name: Optional[str] = None) -> None:
        # 하위 클래스의 기본 이름을 덮어쓸 수 있도록 옵션 제공
        if name is not None:
            self.name = name

    @staticmethod
    def ensure_numpy(array: Any) -> np.ndarray:
        """입력 텐서를 numpy.ndarray 로 변환"""
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        return array

    @abstractmethod
    def compute(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        **kwargs: Dict[str, Any],
    ) -> float:
        """메트릭 값 계산 및 반환"""
        raise NotImplementedError



