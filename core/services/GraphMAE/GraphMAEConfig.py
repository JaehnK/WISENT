from dataclasses import dataclass
from typing import Optional

@dataclass
class GraphMAEConfig:
    """GraphMAE 설정 클래스"""

    # 모델 구조
    hidden_dim: int = 256
    num_layers: int = 2
    num_dec_layers: int = 1
    num_remasking: int = 1
    nhead: int = 4
    nhead_out: int = 1

    # 마스킹 설정
    mask_rate: float = 0.3
    remask_rate: float = 0.5
    remask_method: str = "random"  # "random" or "fixed"
    mask_method: str = "random"

    # 훈련 설정
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    max_epochs: int = 1000

    # 손실 함수
    loss_fn: str = "sce"  # "mse" or "sce"
    alpha_l: float = 2.0
    lam: float = 1.0

    # 기타
    encoder_type: str = "gat"
    decoder_type: str = "gat"
    activation: str = "relu"
    feat_drop: float = 0.2
    attn_drop: float = 0.1
    negative_slope: float = 0.2
    residual: bool = True
    norm: Optional[str] = "layernorm"

    # EMA 설정
    delayed_ema_epoch: int = 0
    momentum: float = 0.996

    # 기타 설정
    replace_rate: float = 0.0
    zero_init: bool = False
    drop_edge_rate: float = 0.0
    device: str = "cuda"

    @classmethod
    def create_default(cls, embed_size: int = 64) -> 'GraphMAEConfig':
        """기본 설정으로 GraphMAEConfig 생성"""
        return cls(
            hidden_dim=embed_size,
            num_layers=2,
            mask_rate=0.3,
            learning_rate=0.001,
            max_epochs=1000
        )