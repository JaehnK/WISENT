# GraphMAE2 구현 및 실험 보고서

## 목차
1. [모델 선택 및 적용](#1-모델-선택-및-적용)
2. [실험 환경](#2-실험-환경)
3. [하이퍼파라미터 설정](#3-하이퍼파라미터-설정)
4. [학습 절차](#4-학습-절차)
5. [평가 방법](#5-평가-방법)
6. [구현 세부사항](#6-구현-세부사항)

---

## 1. 모델 선택 및 적용

### 1.1 GraphMAE2를 선택한 이유

**GraphMAE2**(Graph Masked Autoencoder 2)는 WWW'23에서 발표된 Self-Supervised Graph Learning 모델로, 다음과 같은 이유로 본 연구에 적용되었습니다:

- **Self-supervised Learning**: 라벨 없이 그래프 구조와 노드 특성만으로 학습 가능
- **Masked Autoencoding**: BERT와 유사한 마스킹 전략을 통해 노드 특성을 재구성하며 의미있는 표현 학습
- **Decoding Enhancement**: GraphMAE의 후속 연구로, 디코더 향상을 통해 더 나은 성능 달성
- **대규모 그래프 지원**: Full-batch 및 Mini-batch 학습 모두 지원

**성능 우수성**: 주요 벤치마크에서 SOTA 달성
- Ogbn-arxiv: **71.89%** (기존 GraphMAE 71.03%)
- Ogbn-products: **81.59%** (기존 GraphMAE 78.89%)
- Ogbn-papers100M: **64.89%** (기존 GraphMAE 62.54%)

### 1.2 본 연구 과제에의 적용

본 연구는 **GRACE(GRaph-bAsed Context Extraction)** 파이프라인에 GraphMAE2를 통합하여 문서 클러스터링 과제를 수행합니다:

```
[Reddit Comments]
    ↓
[Document Processing & Preprocessing]
    ↓
[Semantic Graph Construction (Word Graph)]
    ↓
[Node Feature Initialization (Word2Vec + BERT)]
    ↓
[GraphMAE2 Pre-training] ← 본 연구의 핵심
    ↓
[Node Embeddings Extraction]
    ↓
[Clustering (K-Means with Elbow Method)]
    ↓
[Evaluation (Silhouette, NPMI, etc.)]
```

**적용 방식**:
1. Reddit 댓글 코퍼스에서 상위 빈도 단어들로 **Semantic Graph** 구축
2. 각 노드(단어)에 **Word2Vec + BERT** 멀티모달 임베딩 부여
3. GraphMAE2로 그래프 구조를 학습하여 **contextualized embeddings** 생성
4. 생성된 임베딩으로 K-Means 클러스터링 수행

---

## 2. 실험 환경

### 2.1 프레임워크

| 프레임워크 | 버전 | 용도 |
|-----------|------|------|
| **PyTorch** | 2.5.1+cu121 | 딥러닝 프레임워크 |
| **DGL (Deep Graph Library)** | 최신 | 그래프 신경망 연산 |
| **scikit-learn** | 최신 | 클러스터링 및 평가 |
| **transformers** | 최신 | BERT 임베딩 |
| **gensim** | 최신 | Word2Vec 학습 |

### 2.2 하드웨어 환경

- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **CUDA**: 12.1
- **OS**: Linux 6.14.0-33-generic
- **Python**: 3.7+

### 2.3 GraphMAE2 구현체 출처

- **공식 Repository**: [THUDM/GraphMAE2](https://github.com/THUDM/GraphMAE)
- **논문**: [GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner](https://arxiv.org/abs/2304.04779) (WWW'23)
- **구현 경로**: `/core/GraphMAE2/`
- **서비스 래퍼**: `/core/services/GraphMAE/GraphMAEService.py`

### 2.4 의존성 패키지

```
pyyaml
tqdm
tensorboardX
scikit-learn
ogb
torch
dgl
```

---

## 3. 하이퍼파라미터 설정

### 3.1 기본 설정 (GraphMAEConfig)

본 프로젝트에서 사용하는 **기본 하이퍼파라미터 설정**은 다음과 같습니다:

#### 모델 구조
```python
hidden_dim: 256              # 은닉층 차원
num_layers: 2                # 인코더 레이어 수
num_dec_layers: 1            # 디코더 레이어 수
num_remasking: 1             # 디코더 리마스킹 횟수
nhead: 4                     # 어텐션 헤드 수 (인코더)
nhead_out: 1                 # 어텐션 헤드 수 (디코더)
```

#### 마스킹 전략
```python
mask_rate: 0.3               # 노드 마스킹 비율 (30%)
remask_rate: 0.5             # 디코더 리마스킹 비율 (50%)
remask_method: "random"      # 리마스킹 방법 (random/fixed)
mask_method: "random"        # 마스킹 방법
replace_rate: 0.0            # 마스크 토큰 대체 비율
```

#### 훈련 설정
```python
learning_rate: 0.001         # 학습률
weight_decay: 5e-4           # 가중치 감쇠 (L2 regularization)
max_epochs: 1000             # 최대 에폭 수
optimizer: "Adam"            # 옵티마이저
```

#### 손실 함수
```python
loss_fn: "sce"               # Scaled Cosine Error
alpha_l: 2.0                 # SCE Loss의 alpha 파라미터
lam: 1.0                     # Latent loss 가중치
```

#### 기타 설정
```python
encoder_type: "gat"          # Graph Attention Network
decoder_type: "gat"          # Graph Attention Network
activation: "relu"           # 활성화 함수
feat_drop: 0.2               # Feature dropout 비율
attn_drop: 0.1               # Attention dropout 비율
negative_slope: 0.2          # LeakyReLU negative slope
residual: True               # Residual connection 사용
norm: "layernorm"            # Normalization 방법
drop_edge_rate: 0.0          # Edge dropping 비율
```

#### EMA (Exponential Moving Average)
```python
delayed_ema_epoch: 0         # EMA 시작 에폭
momentum: 0.996              # EMA 모멘텀
```

### 3.2 데이터셋별 참고 설정 (Cora 기준)

공식 구현체의 Cora 데이터셋 설정 (참고용):

```yaml
lr: 0.001
num_hidden: 1024             # 더 큰 차원 (Cora는 작은 데이터셋)
num_heads: 8                 # 더 많은 헤드
num_layers: 2
mask_rate: 0.5               # 50% 마스킹
encoder: gat
decoder: gat
loss_fn: sce
alpha_l: 3                   # 더 큰 alpha
remask_method: fixed
momentum: 0                  # EMA 미사용
lam: 0.15                    # 더 작은 latent loss 가중치
```

### 3.3 Ablation Study에서 실험한 설정

#### 고정 파라미터
```python
num_documents: 10,000        # Reddit 댓글 수
top_n_words: 1,000          # 상위 빈도 단어 수
embed_size: 64              # 총 임베딩 차원
  - w2v_dim: 32             # Word2Vec 차원
  - bert_dim: 32            # BERT 차원
embedding_method: 'concat'   # Word2Vec + BERT 결합
graphmae_epochs: 100        # GraphMAE 학습 에폭
graphmae_lr: 0.001          # 학습률
clustering_method: 'kmeans' # 클러스터링 방법
```

#### 실험 대상 파라미터
```python
mask_rate: [0.3, 0.5, 0.75] # 마스킹 비율 변화 실험
```

---

## 4. 학습 절차

### 4.1 Pre-training 과정 (Self-Supervised Learning)

GraphMAE2의 학습은 다음 단계로 진행됩니다:

#### Step 1: 그래프 및 특성 준비
```python
# 1. Word Graph 구축
word_graph = GraphService.build_graph(documents, top_n_words=1000)

# 2. 노드 초기 특성 생성 (Word2Vec + BERT)
input_features = NodeFeatureHandler.calculate_embeddings(
    words=word_graph.words,
    method='concat',  # Word2Vec(32) + BERT(32) = 64
    embed_size=64
)

# 3. DGL 그래프 변환
dgl_graph = GraphService.wordgraph_to_dgl(word_graph, input_features)
```

#### Step 2: GraphMAE2 모델 초기화
```python
model = PreModel(
    in_dim=64,               # 입력 차원
    num_hidden=256,          # 은닉층 차원
    num_layers=2,            # 인코더 레이어 수
    num_dec_layers=1,        # 디코더 레이어 수
    mask_rate=0.3,           # 마스킹 비율
    encoder_type="gat",      # GAT 인코더
    decoder_type="gat",      # GAT 디코더
    loss_fn="sce",           # Scaled Cosine Error
    # ... (기타 하이퍼파라미터)
)
```

#### Step 3: 훈련 루프
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)

for epoch in range(max_epochs):
    # 1. Forward pass
    loss = model(dgl_graph, x=input_features, epoch=epoch)

    # 내부 동작:
    # - 노드 일부를 mask_rate(30%) 비율로 마스킹
    # - GAT 인코더로 마스크된 그래프 인코딩
    # - GAT 디코더로 마스크된 노드 특성 재구성
    # - SCE Loss 계산 (재구성 오차)
    # - EMA 업데이트 (momentum=0.996)

    # 2. Backward pass
    loss.backward()
    optimizer.step()
```

#### Step 4: 임베딩 추출
```python
model.eval()
with torch.no_grad():
    embeddings = model.embed(dgl_graph, input_features)
    # Shape: [num_nodes, hidden_dim] = [1000, 256]
```

### 4.2 학습 동작 메커니즘

**GraphMAE2 학습의 핵심 단계**:

1. **Masking**: 노드의 30%를 랜덤하게 선택하여 마스킹
2. **Encoding**: GAT로 마스크되지 않은 노드들의 정보 집계
3. **Re-masking** (GraphMAE2의 핵심 개선):
   - 디코더에서 추가로 50% 노드를 리마스킹
   - 더 robust한 표현 학습 유도
4. **Decoding**: 마스크된 노드의 원본 특성 재구성
5. **Loss 계산**:
   - **SCE Loss** (Scaled Cosine Error):
     ```
     L = (1 - cos(pred, target)) * α
     ```
   - α (alpha_l=2.0): 코사인 유사도를 스케일링하여 그래디언트 안정화

### 4.3 Fine-tuning 전략

본 프로젝트는 **Linear Probing** 방식을 사용합니다:

```python
# Pre-training으로 얻은 임베딩을 고정
embeddings = graphmae_service.pretrain_and_extract(word_graph)

# 임베딩을 직접 K-Means에 입력
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(embeddings)
```

**특징**:
- End-to-end fine-tuning 대신 **frozen embeddings** 사용
- Downstream task: **Unsupervised Clustering**
- 사전학습된 표현의 품질을 직접 평가

### 4.4 검증 방법

#### Train/Val/Test Split
- 본 연구는 **비지도 학습**이므로 전통적인 train/val/test split 없음
- 대신 **전체 그래프**에 대해 pre-training 수행
- Clustering 평가 지표로 모델 성능 검증

#### 재현성 확보
```python
# 랜덤 시드 고정
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 5. 평가 방법

### 5.1 Downstream Task 정의

**Task**: **Document-level Unsupervised Clustering**

- Reddit 댓글 코퍼스를 의미론적으로 유사한 그룹으로 클러스터링
- GraphMAE2로 학습한 단어 임베딩 기반
- K-Means 알고리즘 사용 (클러스터 수는 Elbow Method로 자동 결정)

**Downstream 파이프라인**:
```
[GraphMAE2 Embeddings]
    ↓
[K-Means Clustering (k=5~20)]
    ↓
[Cluster Assignments]
    ↓
[Evaluation Metrics]
```

### 5.2 평가 지표

본 연구는 다음 4가지 메트릭으로 클러스터링 품질을 평가합니다:

#### 1) Silhouette Score (실루엣 계수)
- **범위**: [-1, 1]
- **높을수록 좋음**
- **의미**: 클러스터 내 응집도(cohesion)와 클러스터 간 분리도(separation) 측정
- **공식**:
  ```
  s(i) = (b(i) - a(i)) / max(a(i), b(i))
  ```
  - a(i): 같은 클러스터 내 평균 거리
  - b(i): 가장 가까운 다른 클러스터까지 평균 거리

#### 2) Davies-Bouldin Index
- **범위**: [0, ∞)
- **낮을수록 좋음**
- **의미**: 클러스터 내 분산 대비 클러스터 간 거리 비율
- **공식**:
  ```
  DB = (1/k) Σ max_j[(σ_i + σ_j) / d(c_i, c_j)]
  ```

#### 3) Calinski-Harabasz Index (Variance Ratio Criterion)
- **범위**: [0, ∞)
- **높을수록 좋음**
- **의미**: 클러스터 간 분산 대비 클러스터 내 분산 비율
- **공식**:
  ```
  CH = [tr(B_k) / (k-1)] / [tr(W_k) / (n-k)]
  ```
  - B_k: 클러스터 간 분산 행렬
  - W_k: 클러스터 내 분산 행렬

#### 4) NPMI (Normalized Pointwise Mutual Information)
- **범위**: [-1, 1]
- **높을수록 좋음**
- **의미**: 클러스터 내 단어들의 의미적 일관성(semantic coherence) 측정
- **특징**: 본 연구에서 **추가 구현한 도메인 특화 메트릭**
- **공식**:
  ```
  NPMI(w_i, w_j) = [log P(w_i, w_j) - log P(w_i) - log P(w_j)] / [-log P(w_i, w_j)]
  ```

### 5.3 Baseline 모델들

본 연구의 Ablation Study에서 비교하는 baseline들:

| Baseline | 설명 | 구현 위치 |
|----------|------|----------|
| **Word2Vec Only** | Word2Vec 임베딩만 사용 (GraphMAE 없음) | `embedding_method='w2v'` |
| **BERT Only** | BERT 임베딩만 사용 (GraphMAE 없음) | `embedding_method='bert'` |
| **No GraphMAE** | Word2Vec+BERT 결합만 (GraphMAE 없음) | `use_graphmae=False` |
| **GraphMAE (w/ concat)** | Word2Vec+BERT + GraphMAE | `embedding_method='concat'` |
| **Mask Rate Variants** | mask_rate = 0.3, 0.5, 0.75 변화 | Ablation parameter |

### 5.4 평가 실행 코드

```python
from core.services.Experiment.AblationService import AblationService

# Ablation Study 실행
ablation = AblationService(base_config=config)
ablation.prepare_shared_data()

# 1. Embedding Method Ablation
embedding_results = ablation.run_embedding_ablation()

# 2. GraphMAE Ablation (유/무 비교)
graphmae_results = ablation.run_graphmae_ablation()

# 3. Hyperparameter Ablation (mask_rate)
hyperparam_results = ablation.run_hyperparameter_ablation(
    param_name='mask_rate',
    param_values=[0.3, 0.5, 0.75]
)
```

---

## 6. 구현 세부사항

### 6.1 프로젝트 구조

```
/home/jaehun/lab/SENTIMENT/
├── core/
│   ├── GraphMAE2/                        # 공식 GraphMAE2 구현체
│   │   ├── models/
│   │   │   ├── edcoder.py               # PreModel (Encoder-Decoder)
│   │   │   ├── gat.py                   # Graph Attention Network
│   │   │   ├── gcn.py                   # Graph Convolutional Network
│   │   │   └── loss_func.py             # SCE Loss 등
│   │   ├── configs/                      # 데이터셋별 YAML 설정
│   │   │   ├── cora.yaml
│   │   │   ├── citeseer.yaml
│   │   │   └── ...
│   │   ├── datasets/                     # 데이터 로딩 및 샘플링
│   │   ├── main_full_batch.py           # Full-batch 학습 스크립트
│   │   ├── main_large.py                # Mini-batch 학습 스크립트
│   │   └── requirements.txt
│   │
│   ├── services/
│   │   ├── GraphMAE/
│   │   │   ├── GraphMAEService.py       # GraphMAE 서비스 래퍼
│   │   │   └── GraphMAEConfig.py        # 설정 클래스
│   │   ├── GRACE/
│   │   │   ├── GRACEPipeline.py         # GRACE 전체 파이프라인
│   │   │   └── GRACEConfig.py           # GRACE 설정
│   │   ├── Experiment/
│   │   │   └── AblationService.py       # Ablation Study 서비스
│   │   ├── Graph/
│   │   │   ├── GraphService.py          # 그래프 구축
│   │   │   └── NodeFeatureHandler.py    # 노드 특성 계산
│   │   └── ...
│   │
│   └── entities/
│       ├── WordGraph.py                  # 그래프 엔티티
│       └── Word.py                       # 단어 엔티티
│
└── examples/
    ├── custom_ablation_10k.py           # 10k 문서 Ablation 실험
    ├── compare_grace_with_traditional.py
    └── ...
```

### 6.2 핵심 클래스 및 메서드

#### GraphMAEService
```python
class GraphMAEService:
    """GraphMAE 기반 노드 임베딩 서비스"""

    def create_mae_model(self, input_dim: int) -> PreModel:
        """GraphMAE2 모델 생성"""

    def prepare_input_features(self, words, embed_size, method='bert'):
        """입력 특성 준비 (Word2Vec/BERT/Concat)"""

    def pretrain_and_extract(self, word_graph, embed_size=64):
        """사전훈련 및 임베딩 추출 (주요 메서드)"""
```

#### PreModel (GraphMAE2 핵심)
```python
class PreModel(nn.Module):
    """GraphMAE2 Encoder-Decoder 모델"""

    def forward(self, g, x, epoch):
        """Forward pass: Masking → Encoding → Decoding → Loss"""

    def embed(self, g, x):
        """학습된 임베딩 추출"""

    def encoding_mask_noise(self, g, x, mask_rate):
        """노드 마스킹"""
```

### 6.3 주요 기능

1. **멀티모달 노드 특성**:
   - Word2Vec (32차원) + BERT (32차원) = 64차원
   - 각 단어는 분포적 의미(Word2Vec) + 문맥적 의미(BERT) 결합

2. **재현성 보장**:
   - 모든 랜덤 시드 고정 (random_state=42)
   - CUDA deterministic mode 활성화

3. **Elbow Method 자동화**:
   - 클러스터 수 k를 5~20 범위에서 자동 탐색
   - Silhouette Score 기반 최적 k 선택

4. **손실 함수 (SCE Loss)**:
   ```python
   def sce_loss(x, y, alpha=2):
       """Scaled Cosine Error"""
       x = F.normalize(x, p=2, dim=-1)
       y = F.normalize(y, p=2, dim=-1)
       loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
       return loss.mean()
   ```

### 6.4 실험 재현 방법

#### 환경 설정
```bash
# 1. Conda 환경 활성화
conda activate SENTIMENT

# 2. 의존성 설치
pip install torch dgl transformers gensim scikit-learn pyyaml tqdm
```

#### 실험 실행
```bash
# Ablation Study 실행 (10k 문서, mask_rate 변화)
python examples/custom_ablation_10k.py

# 또는 서비스 직접 사용
python -c "
from core.services.GRACE.GRACEPipeline import GRACEPipeline
from core.services.GRACE.GRACEConfig import GRACEConfig

config = GRACEConfig(
    csv_path='kaggle_RC_2019-05.csv',
    num_documents=10000,
    graphmae_epochs=100,
    mask_rate=0.3
)

pipeline = GRACEPipeline(config)
results = pipeline.run()
print(results['metrics'])
"
```

---

## 참고 문헌

1. **GraphMAE2 논문**:
   - Hou, Z., He, Y., Cen, Y., Liu, X., Dong, Y., Kharlamov, E., & Tang, J. (2023). GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner. *Proceedings of the ACM Web Conference 2023 (WWW'23)*.
   - Paper: https://arxiv.org/abs/2304.04779

2. **GraphMAE (Predecessor)**:
   - Hou, Z., et al. (2022). GraphMAE: Self-Supervised Masked Graph Autoencoders. *arXiv preprint arXiv:2205.10803*.

3. **공식 구현체**:
   - Repository: https://github.com/THUDM/GraphMAE
   - Pre-trained checkpoints: [Google Drive](https://drive.google.com/drive/folders/1GiuP0PtIZaYlJWIrjvu73ZQCJGr6kGkh)

---

## 부록

### A. 하이퍼파라미터 튜닝 가이드

| 파라미터 | 작은 그래프 (<5k nodes) | 큰 그래프 (>10k nodes) |
|---------|----------------------|---------------------|
| `hidden_dim` | 256-512 | 128-256 |
| `num_layers` | 2-3 | 2 |
| `mask_rate` | 0.3-0.5 | 0.5-0.75 |
| `learning_rate` | 0.001 | 0.0001-0.001 |
| `max_epochs` | 500-1000 | 100-500 |

### B. 문제 해결

**Q: CUDA Out of Memory 오류**
```python
# 배치 크기 줄이기
config.graphmae_epochs = 50  # 더 적은 에폭
config.hidden_dim = 128       # 더 작은 차원
```

**Q: 학습이 너무 느림**
```python
# Mini-batch 학습 사용 (대규모 그래프)
# core/GraphMAE2/main_large.py 참고
```

---

**보고서 작성일**: 2025-10-17
**프로젝트**: SENTIMENT - GRACE Pipeline with GraphMAE2
**작성자**: Claude (Anthropic)
