# GRACE 석사 논문 연구 로드맵

> **현재 상태**: 기술 구현 85% 완료, 연구 프레임워크 60% 완료
> **목표**: 학술적으로 타당한 석사 학위논문 완성

---

## 📋 목차

1. [현재 코드 상태 요약](#1-현재-코드-상태-요약)
2. [코드 수정 사항 (이미 계획됨)](#2-코드-수정-사항-이미-계획됨)
3. [논문 작성 시 핵심 요구사항](#3-논문-작성-시-핵심-요구사항)
4. [논문 구조 상세 가이드](#4-논문-구조-상세-가이드)
5. [실험 설계 체크리스트](#5-실험-설계-체크리스트)
6. [타임라인 제안](#6-타임라인-제안)

---

## 1. 현재 코드 상태 요약

### ✅ 구현 완료 (잘 된 부분)

```
core/
├── entities/          # Word, WordGraph, Sentence 등
├── services/
│   ├── Document/      # 텍스트 전처리
│   ├── Graph/         # 의미연결망 구축
│   ├── Word2Vec/      # Word2Vec 임베딩
│   ├── DBert/         # BERT 임베딩
│   ├── GraphMAE/      # GraphMAE 자기지도학습
│   └── GRACE/
│       ├── GRACEPipeline.py      # 파이프라인 오케스트레이션
│       ├── ClusteringService.py   # K-means, Elbow Method
│       ├── MetricsService.py      # Silhouette, Davies-Bouldin 등
│       └── GRACEConfig.py         # 설정 관리
```

**강점:**
- 모듈화된 아키텍처
- GraphMAE 통합 완료
- 멀티모달 임베딩 (Word2Vec + BERT)
- Elbow Method 자동 클러스터 탐색

---

## 2. 코드 수정 사항 (이미 계획됨)

### 당신이 제시한 계획에 포함된 사항들

#### ✅ 2.1. 전통적 그래프 클러스터링 추가
**위치**: `GraphClusteringService.py` (신규 파일)

```python
# 그래프 구조 기반 클러스터링 (의미연결망에 직접 적용)
- louvain_clustering()        # Louvain (Modularity 최적화)
- leiden_clustering()         # Leiden (Louvain 개선판)
- girvan_newman_clustering()  # Girvan-Newman (Edge betweenness)
```

**논문 기여**:
- 전통적 그래프 클러스터링과 비교 → "구조만 활용하는 방법의 한계" 증명
- GRACE의 차별점: GraphMAE로 구조 + 의미 임베딩 동시 학습 → **특화된 방법의 필요성** 입증

---

#### ✅ 2.2. Baseline 구현
**새 파일**: `BaselineService.py`

```python
class BaselineService:
    - tfidf_kmeans()           # TF-IDF + K-means
    - word2vec_only_kmeans()   # Word2Vec만 사용
    - bert_only_kmeans()       # BERT만 사용
    - no_graphmae_kmeans()     # GraphMAE 없이
```

**논문 기여**: Ablation Study, 각 컴포넌트 효과 검증

---

#### ✅ 2.3. ExperimentService 구현
**새 파일**: `ExperimentService.py`

```python
class ExperimentService:
    - run_ablation_study()     # 하이퍼파라미터 조합 자동 실험
    - compare_baselines()      # 여러 방법 비교
    - statistical_test()       # t-test, Wilcoxon signed-rank
    - generate_result_table()  # LaTeX 표 생성
```

**논문 기여**: 재현 가능한 실험, 통계적 유의성 검증

---

#### ✅ 2.4. 재현성 보장
**수정 위치**: 모든 서비스

```python
# 추가 필요
- DocumentService: random seed 설정
- Word2VecService: seed 고정
- BertService: torch.manual_seed()
- GraphMAEService:
    - torch.manual_seed(seed)
    - torch.cuda.manual_seed_all(seed)
    - np.random.seed(seed)
- ClusteringService: 이미 구현됨 ✅
```

**논문 기여**: 실험 재현 가능성 (학술 연구 필수)

---

#### ✅ 2.5. 통계적 검증
**위치**: `MetricsService.py` 확장

```python
# 추가 메서드
- paired_ttest()              # 두 방법 비교
- wilcoxon_test()             # 비모수 검정
- compute_confidence_interval() # 95% CI
- multiple_runs_statistics()  # 평균±표준편차
```

**논문 기여**: "통계적으로 유의미한 개선" 주장 가능

---

#### ✅ 2.6. 시각화
**새 파일**: `VisualizationService.py`

```python
class VisualizationService:
    - plot_tsne()              # t-SNE 2D 클러스터 시각화
    - plot_umap()              # UMAP 시각화
    - plot_comparison_bar()    # 방법별 메트릭 막대그래프
    - plot_ablation_heatmap()  # 하이퍼파라미터 히트맵
    - plot_cluster_wordcloud() # 클러스터별 워드클라우드
```

**논문 기여**: 정성적 분석, 독자 이해도 향상

---

### 코드 수정 요약

**네, 맞습니다. 당신이 제시한 계획에 모두 포함되어 있습니다.**

| 항목 | 파일 | 상태 |
|------|------|------|
| 전통 클러스터링 | `ClusteringService.py` | 확장 필요 |
| Baseline | `BaselineService.py` | 신규 작성 |
| 실험 자동화 | `ExperimentService.py` | 신규 작성 |
| 재현성 | 모든 서비스 | 시드 추가 |
| 통계 검증 | `MetricsService.py` | 확장 필요 |
| 시각화 | `VisualizationService.py` | 신규 작성 |

**예상 작업량**: 코드 2-3주 + 실험 1-2주 = **1개월**

---

## 3. 논문 작성 시 핵심 요구사항

### 🎯 3.1. 연구 문제 명확화 (가장 중요!)

#### ❌ 피해야 할 서술
> "GraphMAE는 최신 그래프 자기지도학습 기법이다. 이를 텍스트 클러스터링에 적용해본다."

**문제점**: Technology-driven, "왜?" 에 답 없음

---

#### ✅ 올바른 서술 (예시)

```markdown
### 연구 배경 및 동기

**문제점 1**: 소셜 미디어 텍스트의 특수성
- Reddit, Twitter 등의 댓글/게시글은 평균 50-100 단어의 짧은 텍스트
- 비공식적 언어 사용 (은어, 줄임말, 오타)
- 맥락 의존적 (이전 댓글 참조, 암묵적 주제)

**문제점 2**: 전통적 방법의 한계
- TF-IDF: 단어 빈도만 고려, 의미 관계 무시
  → Sparse representation, 동의어/유의어 처리 불가
- Word2Vec: 지역적 문맥만 학습, 문서 수준 구조 간과
- BERT: 단일 문서 인코딩, 전역적 단어-단어 관계 미반영

**문제점 3**: 기존 그래프 기반 방법의 한계
- TextGCN 등은 레이블 필요 (지도학습)
- 클러스터링은 비지도 문제 → 적용 불가

**Research Gap**:
짧은 소셜 미디어 텍스트에서 단어 간 전역적 의미 관계를 비지도 방식으로
학습하여 클러스터링 성능을 개선하는 방법 부재

**제안 방법**:
1. 공출현 기반 의미연결망으로 전역적 단어 관계 그래프화
2. GraphMAE 자기지도학습으로 그래프 구조 학습 (레이블 불필요)
3. 멀티모달 임베딩 (Word2Vec + BERT)으로 지역/전역 정보 결합
```

**이 서술이 Introduction 1-2페이지를 채웁니다.**

---

### 🎯 3.2. Research Questions 명시

논문에는 명확한 RQ가 필요합니다.

```markdown
## Research Questions

**RQ1**: 의미연결망 기반 GraphMAE 학습이 전통적 임베딩 방법 대비
       텍스트 클러스터링 성능을 향상시키는가?

**RQ2**: 멀티모달 임베딩 (Word2Vec + BERT)이 단일 임베딩 방법보다
       효과적인가?

**RQ3**: GraphMAE의 주요 하이퍼파라미터 (mask rate, 임베딩 차원, epoch)가
       클러스터링 성능에 미치는 영향은 무엇인가?

**RQ4**: 제안 방법이 어떤 유형의 텍스트 데이터에서 효과적인가?
```

---

### 🎯 3.3. 기여도 명확화 (Contributions)

```markdown
## Contributions

1. **새로운 프레임워크 제안**
   - GraphMAE를 텍스트 클러스터링에 최초 적용한 GRACE 프레임워크

2. **멀티모달 통합 전략**
   - Word2Vec (지역 문맥) + BERT (의미 표현) 결합 방법 제시

3. **포괄적 실험 분석**
   - 4개 baseline과의 비교 실험
   - 3개 하이퍼파라미터에 대한 ablation study
   - 통계적 유의성 검증

4. **재현 가능한 오픈소스 구현**
   - 모듈화된 파이프라인 공개 (GitHub)
```

**논문 심사위원이 가장 주목하는 부분입니다.**

---

### 🎯 3.4. 데이터셋 선택 및 정당화

#### 현재 문제
```python
csv_path = '/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv'
```
→ "그냥 있어서 썼다"는 인상

---

#### 논문에 작성해야 할 내용

```markdown
## Datasets

### 4.1. Reddit Comments Dataset (RC-2019-05)

**선택 이유**:
1. **규모**: 500만 개 댓글 (실험에는 1,000-10,000개 샘플링)
2. **다양성**: 다양한 subreddit 포함 (정치, 기술, 일상 등)
3. **특성**: 짧은 텍스트 (평균 87단어), 비공식적 언어
4. **공개 데이터**: 재현 가능성 보장

**전처리**:
- 최소 길이: 10단어 이상
- 특수문자 제거, 소문자화
- Stopwords 제거 (NLTK 기본 + 도메인 특화)

**분할**:
- 문서 수: 1,000 / 5,000 / 10,000 (규모별 성능 분석)
- Top-N 단어: 500 / 1,000 (그래프 크기 조절)

### 4.2. (추가 데이터셋 - 선택사항)

**20 Newsgroups** (비교용)
- 공식 벤치마크, 긴 문서
- GRACE가 짧은 텍스트에 특화됨을 보이기 위한 대조군

**Twitter Sentiment140** (선택)
- 극도로 짧은 텍스트 (140자)
- 극한 상황에서의 성능 검증
```

---

### 🎯 3.5. Baseline 선택 및 정당화

```markdown
## Baselines

### 5.1. 전통적 방법

1. **TF-IDF + K-means**
   - 가장 기본적인 텍스트 클러스터링 방법
   - 단어 빈도 기반, 해석 가능성 높음

2. **Word2Vec + K-means**
   - 단어 임베딩 기반
   - 의미 유사도 반영

3. **BERT + K-means**
   - 최신 언어 모델
   - 문맥 의존 표현

### 5.2. Ablation Baselines

4. **GRACE w/o GraphMAE** (멀티모달 임베딩만)
   - GraphMAE 효과 검증

5. **GRACE w/ Word2Vec only**
   - 멀티모달 효과 검증

6. **GRACE w/ BERT only**
   - 멀티모달 효과 검증

### 5.3. 비교 불가능한 방법 (논문에 언급)

- **TextGCN**: 지도학습 방법, 레이블 필요 → 비교 제외
- **LDA**: 토픽 모델링, 클러스터링과 목적 상이 → 참고만
```

**심사위원이 묻는 질문 대비**: "왜 XXX 방법과 비교 안 했나요?"

---

### 🎯 3.6. 평가 지표 선택 및 정당화

#### 현재 구현
```python
eval_metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi']
```

---

#### 논문 작성 시

```markdown
## Evaluation Metrics

### 6.1. 내재적 지표 (Intrinsic Metrics)

클러스터 자체의 품질 평가 (ground truth 불필요)

1. **Silhouette Score** [-1, 1]
   - 공식: s = (b - a) / max(a, b)
   - 해석: 클러스터 내 응집도 vs 클러스터 간 분리도
   - 높을수록 좋음

2. **Davies-Bouldin Index** [0, ∞)
   - 공식: DB = (1/k) Σ max(R_ij)
   - 해석: 클러스터 간 유사도 평균
   - 낮을수록 좋음

3. **Calinski-Harabasz Score** [0, ∞)
   - 공식: CH = (tr(B_k) / tr(W_k)) × ((n-k)/(k-1))
   - 해석: 클러스터 간 분산 / 클러스터 내 분산
   - 높을수록 좋음

### 6.2. 주제 일관성 (Topic Coherence)

4. **Normalized PMI (NPMI)** [-1, 1]
   - 클러스터 내 상위 단어들의 공출현 기반 일관성
   - 높을수록 해석 가능성 높음

   PMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i)P(w_j)))
   NPMI = PMI / -log(P(w_i, w_j))

### 6.3. 외재적 지표 (선택사항, 레이블 있는 경우)

5. **Adjusted Rand Index (ARI)**
6. **Normalized Mutual Information (NMI)**

→ 20 Newsgroups 등 레이블 있는 데이터셋에만 적용

### 6.4. 정성적 평가

7. **클러스터 대표 단어 분석**
   - 각 클러스터의 상위 10-20개 단어 제시
   - 주제 해석 가능성 논의

8. **사례 연구 (Case Study)**
   - 특정 클러스터의 실제 문서 예시
   - GRACE vs Baseline 차이 분석
```

---

### 🎯 3.7. 하이퍼파라미터 설정 및 탐색 범위

```markdown
## Hyperparameters

### 7.1. 고정 파라미터 (모든 실험 동일)

- 전처리: stopwords 제거 활성화
- 최소 문장 길이: 10 단어
- 랜덤 시드: 42 (재현성)
- K-means 초기화: 10회 반복 (n_init=10)

### 7.2. GRACE 기본 설정 (Default)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `num_documents` | 1,000 | 사용 문서 수 |
| `top_n_words` | 500 | 그래프 노드 수 |
| `embed_size` | 64 | 임베딩 차원 (w2v 32 + bert 32) |
| `graphmae_epochs` | 100 | GraphMAE 학습 에포크 |
| `graphmae_lr` | 0.001 | Learning rate |
| `mask_rate` | 0.75 | GraphMAE 마스킹 비율 |
| `num_clusters` | Auto | Elbow Method 자동 탐색 |

### 7.3. Ablation Study 탐색 범위 (RQ3)

**Mask Rate** (GraphMAE 핵심 파라미터)
- 범위: [0.3, 0.5, 0.75, 0.9]
- 가설: 0.75가 최적 (논문 기본값)

**Embedding Dimension**
- 범위: [32, 64, 128, 256]
- 가설: 64 이상에서 포화

**GraphMAE Epochs**
- 범위: [50, 100, 200, 300]
- 가설: 100 이상에서 수렴

**클러스터 수 (K)**
- 자동 탐색 (Elbow) vs 고정 [5, 10, 15, 20]
```

---

### 🎯 3.8. 실험 재현성 보장

```markdown
## Reproducibility

모든 실험은 다음 조건에서 재현 가능:

### 소프트웨어 환경
- Python 3.9.12
- PyTorch 2.0.1
- DGL 1.1.0
- scikit-learn 1.3.0
- transformers 4.30.2

### 하드웨어 환경
- CPU: Intel Xeon / AMD Ryzen 이상
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 32GB 이상

### 랜덤 시드 설정
```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
```

### 코드 공개
- GitHub: https://github.com/[your-repo]/GRACE
- 라이센스: MIT
```

---

### 🎯 3.9. 통계적 유의성 검증

```markdown
## Statistical Significance

### 실험 반복
- 각 설정당 **5회 반복** (다른 랜덤 초기화)
- 결과: 평균 ± 표준편차 (mean ± std)

### 유의성 검정
- **Paired t-test** (정규분포 가정)
  - H0: GRACE와 Baseline 성능 차이 없음
  - H1: GRACE가 유의미하게 우수
  - 유의수준: α = 0.05

- **Wilcoxon signed-rank test** (비모수 대안)
  - 분포 가정 불필요
  - 소규모 반복 실험에 적합

### 결과 표기
- p < 0.05: * (유의미)
- p < 0.01: ** (매우 유의미)
- p < 0.001: *** (극도로 유의미)

### 예시 표
| Method | Silhouette ↑ | Davies-Bouldin ↓ | p-value |
|--------|--------------|-------------------|---------|
| TF-IDF | 0.45 ± 0.03  | 1.23 ± 0.08       | -       |
| GRACE  | **0.58 ± 0.02** | **0.87 ± 0.05** | **0.003** |
```

---

### 🎯 3.10. 정성적 분석 (필수!)

숫자만으로는 설득력 부족. 실제 결과 예시 필요.

```markdown
## Qualitative Analysis

### 클러스터 예시 (Reddit RC-2019-05, k=10)

#### Cluster 0: 정치/선거 관련
**Top words**: election, vote, campaign, president, poll, debate, senate
**GRACE 특징**: 'campaign'과 'donate'를 같은 클러스터로 묶음 (의미 관계)
**TF-IDF 문제**: 빈도만 보고 'poll'을 통계 클러스터로 분류

#### Cluster 3: 게임/엔터테인먼트
**Top words**: game, play, level, character, boss, quest, update
**GRACE 특징**: 'nerf', 'buff' 같은 게임 은어 정확히 분류
**Word2Vec 문제**: 문맥 부족으로 'level'을 교육 클러스터와 혼동

### 사례 연구

**문서 예시**:
> "Just hit level 50! The new update nerfed my main character but the boss fights are more balanced now."

- **GRACE**: Cluster 3 (게임) ✓
- **TF-IDF**: Cluster 7 (기타) ✗ → 'level', 'character' 빈도 낮음
- **BERT only**: Cluster 3 (게임) ✓ but 신뢰도 낮음 (0.6 vs GRACE 0.9)

### 시각화

**그림 1**: t-SNE로 본 클러스터 분포 (GRACE vs TF-IDF)
- GRACE: 명확한 경계, 높은 응집도
- TF-IDF: 클러스터 겹침, 경계 모호

**그림 2**: 워드클라우드 (Cluster 0 비교)
- GRACE: 일관된 주제 (정치 용어들)
- Baseline: 혼재된 단어들
```

---

## 4. 논문 구조 상세 가이드

### 표준 석사 논문 구조 (40-60 페이지)

```markdown
# 논문 제목 (국문)
그래프 자기지도학습 기반 단문 텍스트 클러스터링 프레임워크

# 논문 제목 (영문)
GRACE: GRAph-based Clustering with Enhanced embeddings for Short Text

---

## Abstract (국문 초록) - 1페이지
- 연구 배경 (2-3문장)
- 문제점 및 Research Gap (2-3문장)
- 제안 방법 (3-4문장)
- 실험 결과 요약 (2-3문장)
- 기여도 (1-2문장)

## Abstract (영문) - 1페이지
(동일 내용 영문)

---

## 1. Introduction (서론) - 4-5페이지

### 1.1. 연구 배경 및 동기 (1.5페이지)
- 텍스트 클러스터링의 중요성
- 소셜 미디어 데이터의 특수성
- 기존 방법의 한계

### 1.2. 연구 목표 및 범위 (1페이지)
- Research Questions (RQ1-4)
- 연구 범위 (비지도 클러스터링, 영어 텍스트)

### 1.3. 기여도 (1페이지)
- Contribution 1-4 상세 설명

### 1.4. 논문 구성 (0.5페이지)
- 각 장 간단 소개

---

## 2. Related Work (관련 연구) - 6-8페이지

### 2.1. 텍스트 클러스터링 (2페이지)
- 전통적 방법 (TF-IDF, K-means)
- 임베딩 기반 방법 (Word2Vec, Doc2Vec)
- 딥러닝 방법 (BERT, Sentence-BERT)

### 2.2. 그래프 기반 텍스트 분석 (2페이지)
- TextGCN, BertGCN (지도학습)
- 의미연결망 연구
- 한계: 레이블 필요

### 2.3. 그래프 자기지도학습 (2페이지)
- GNN Pretraining 개요
- GraphMAE (ICML 2022) 상세 설명
- 텍스트 도메인 적용 사례 부족

### 2.4. 멀티모달 임베딩 (1페이지)
- 다중 특성 결합 연구
- 앙상블 vs Concatenation

### 2.5. 본 연구의 차별점 (1페이지)
- 표로 비교 (방법별 특성)

| Method | Graph | Self-supervised | Multimodal | Unsupervised |
|--------|-------|-----------------|------------|--------------|
| TF-IDF | ✗ | ✗ | ✗ | ✓ |
| TextGCN | ✓ | ✗ | ✗ | ✗ |
| BERT | ✗ | ✓ | ✗ | ✓ |
| **GRACE** | ✓ | ✓ | ✓ | ✓ |

---

## 3. Methodology (제안 방법) - 10-12페이지

### 3.1. 전체 프레임워크 (1페이지)
- GRACE 파이프라인 다이어그램
- 6단계 설명

### 3.2. 텍스트 전처리 (1페이지)
- Tokenization
- Stopwords 제거
- POS tagging

### 3.3. 의미연결망 구축 (2페이지)
- 공출현 기반 그래프 정의
  - G = (V, E), V = words, E = co-occurrence
- 엣지 가중치 계산 (PMI, NPMI)
- 그래프 통계 (노드 수, 엣지 수, 밀도)

### 3.4. 멀티모달 노드 임베딩 (2페이지)
- Word2Vec (Skip-gram)
  - 파라미터: window=5, dim=32
- BERT (DistilBERT)
  - [CLS] 토큰 사용, dim=32
- Concatenation
  - h_i = [h_w2v; h_bert] ∈ R^64

### 3.5. GraphMAE 자기지도학습 (3페이지)
- 마스킹 전략
  - 랜덤 노드 75% 마스킹
- 인코더-디코더 구조
  - GCN 2층 인코더
  - GCN 2층 디코더
- 손실 함수
  - L = ||X_masked - X_reconstructed||^2
- 임베딩 추출
  - Z = Encoder(G, H)

### 3.6. 클러스터링 (1페이지)
- K-means (sklearn)
- Elbow Method로 k 자동 결정

### 3.7. 계산 복잡도 (1페이지)
- 시간 복잡도 분석
- 공간 복잡도 분석

---

## 4. Experimental Setup (실험 설정) - 6-8페이지

### 4.1. 데이터셋 (2페이지)
- Reddit RC-2019-05 상세 설명
- 전처리 통계
- 샘플 예시

### 4.2. Baselines (2페이지)
- 6개 baseline 상세 설명
- 하이퍼파라미터 설정

### 4.3. 평가 지표 (1.5페이지)
- 4개 지표 공식 및 해석

### 4.4. 구현 세부사항 (1.5페이지)
- 소프트웨어/하드웨어 환경
- 재현성 보장 방법
- 실험 시간

---

## 5. Results and Discussion (결과 및 분석) - 10-12페이지

### 5.1. RQ1: GRACE vs Baselines (3페이지)

**표 1**: 전체 방법 비교 (1,000 documents)

| Method | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | NPMI ↑ |
|--------|--------------|-------------------|----------------------|--------|
| TF-IDF + KMeans | 0.45 ± 0.03 | 1.23 ± 0.08 | 245 ± 18 | 0.32 ± 0.05 |
| Word2Vec + KMeans | 0.52 ± 0.02 | 1.05 ± 0.06 | 312 ± 22 | 0.41 ± 0.04 |
| BERT + KMeans | 0.54 ± 0.03 | 0.98 ± 0.07 | 338 ± 25 | 0.43 ± 0.03 |
| **GRACE** | **0.62 ± 0.02*** | **0.78 ± 0.04*** | **421 ± 19*** | **0.56 ± 0.03*** |

*p < 0.01 (paired t-test vs best baseline)

**분석**:
- GRACE가 모든 지표에서 최고 성능
- Silhouette 15% 개선 (0.54 → 0.62)
- 통계적으로 유의미 (p < 0.01)

---

### 5.2. RQ2: Ablation Study (3페이지)

**표 2**: 컴포넌트별 기여도

| Configuration | Silhouette ↑ | 개선율 |
|---------------|--------------|--------|
| Word2Vec only | 0.52 | baseline |
| BERT only | 0.54 | +3.8% |
| W2V + BERT (no GraphMAE) | 0.56 | +7.7% |
| **W2V + BERT + GraphMAE** | **0.62** | **+19.2%** |

**분석**:
- 멀티모달: +7.7% 개선
- GraphMAE: +10.7% 추가 개선
- 시너지 효과 확인

---

### 5.3. RQ3: 하이퍼파라미터 영향 (3페이지)

**그림 3**: Mask Rate에 따른 성능 변화
- 0.75에서 최고 성능
- 0.9에서 과도한 마스킹으로 성능 저하

**그림 4**: 임베딩 차원에 따른 성능
- 64 이상에서 포화
- 256은 과적합 경향

---

### 5.4. 정성적 분석 (2페이지)
- 클러스터 예시 3개
- t-SNE 시각화
- 워드클라우드

---

### 5.5. 실행 시간 분석 (1페이지)

**표 3**: 방법별 실행 시간 (1,000 documents)

| Method | Time (seconds) |
|--------|----------------|
| TF-IDF | 2.3 |
| Word2Vec | 15.7 |
| BERT | 45.2 |
| GRACE | 67.8 |

**분석**: GRACE는 느리지만 오프라인 사전학습 가능

---

## 6. Conclusion (결론) - 2-3페이지

### 6.1. 연구 요약 (1페이지)
- 문제, 방법, 결과 요약

### 6.2. 기여도 재강조 (0.5페이지)
- 학술적 기여
- 실용적 기여

### 6.3. 한계점 (0.5페이지)
- 영어만 실험
- 레이블 데이터 미활용
- 계산 비용

### 6.4. 향후 연구 방향 (1페이지)
- 다국어 확장
- 준지도학습 버전
- 실시간 클러스터링
- 하이퍼그래프 확장

---

## References (참고문헌) - 3-4페이지
- 40-60개 논문 인용 권장

---

## Appendix (부록) - 선택사항
- 전체 클러스터 결과
- 추가 실험
- 코드 스니펫
```

---

## 5. 실험 설계 체크리스트

### ✅ 실험 전 확인사항

```markdown
## Before Running Experiments

### 데이터 준비
- [ ] Reddit 데이터 다운로드 및 검증
- [ ] 전처리 파이프라인 테스트
- [ ] 샘플 크기별 데이터 분할 (1k, 5k, 10k)
- [ ] 통계 정보 저장 (단어 수, 문장 수 등)

### 코드 완성도
- [ ] BaselineService 구현
- [ ] ExperimentService 구현
- [ ] VisualizationService 구현
- [ ] 모든 서비스 랜덤 시드 설정
- [ ] 단위 테스트 작성

### 실험 설정
- [ ] 하이퍼파라미터 그리드 정의
- [ ] 실행 스크립트 작성
- [ ] 로그 저장 경로 설정
- [ ] GPU 메모리 체크

### 재현성
- [ ] requirements.txt 작성
- [ ] 환경 변수 문서화
- [ ] 실행 명령어 문서화
- [ ] Git 커밋 (실험 시작 시점)
```

---

### ✅ 실험 중 체크리스트

```markdown
## During Experiments

### 진행 상황 추적
- [ ] 실험 로그 자동 저장
- [ ] 중간 체크포인트 저장
- [ ] 메모리 사용량 모니터링
- [ ] 예상 완료 시간 계산

### 품질 관리
- [ ] NaN/Inf 값 체크
- [ ] 비정상 결과 플래그
- [ ] 실패한 실험 재실행 스크립트
- [ ] 백업 주기적 실행
```

---

### ✅ 실험 후 체크리스트

```markdown
## After Experiments

### 결과 검증
- [ ] 모든 실험 완료 확인
- [ ] 결과 무결성 체크
- [ ] 이상치(outlier) 분석
- [ ] 재현 테스트 (랜덤 실험 1개 재실행)

### 분석 및 시각화
- [ ] 결과 테이블 생성 (LaTeX)
- [ ] 그래프/차트 생성 (고해상도)
- [ ] 통계 검정 수행
- [ ] 정성적 분석 작성

### 문서화
- [ ] 실험 노트 정리
- [ ] 결과 요약 문서
- [ ] 발견한 인사이트 기록
- [ ] 논문 초안 작성
```

---

## 6. 타임라인 제안

### 📅 3개월 계획 (석사 논문 기준)

#### **Month 1: 코드 완성 및 Baseline 실험**

| 주차 | 작업 | 산출물 |
|------|------|--------|
| Week 1 | BaselineService 구현, 재현성 보장 | TF-IDF, Word2Vec, BERT baselines |
| Week 2 | ExperimentService, 통계 검증 | Ablation 자동화 스크립트 |
| Week 3 | VisualizationService, 정성 분석 | t-SNE, 워드클라우드 |
| Week 4 | Baseline 실험 실행 | Baseline 결과 테이블 |

**마일스톤**: GRACE vs Baselines 비교 완료

---

#### **Month 2: Ablation Study 및 분석**

| 주차 | 작업 | 산출물 |
|------|------|--------|
| Week 5 | Mask rate ablation (0.3-0.9) | 그림 3 |
| Week 6 | 임베딩 차원 ablation (32-256) | 그림 4 |
| Week 7 | Epoch ablation, 클러스터 수 분석 | 표 4-5 |
| Week 8 | 통계 검정, 정성적 분석 | 사례 연구 3개 |

**마일스톤**: 모든 RQ 답변 완료

---

#### **Month 3: 논문 작성**

| 주차 | 작업 | 산출물 |
|------|------|--------|
| Week 9 | Introduction + Related Work | 초안 10페이지 |
| Week 10 | Methodology + Experiments | 초안 15페이지 |
| Week 11 | Results + Discussion + Conclusion | 초안 완성 (40페이지) |
| Week 12 | 교정, 그림/표 정리, 제출 | 최종본 |

**마일스톤**: 논문 제출

---

### 📅 단축 버전: 6주 계획 (급할 때)

| 주차 | 핵심 작업 |
|------|----------|
| Week 1-2 | 코드 완성 (Baseline + 재현성) |
| Week 3 | Baseline 실험 + RQ1 |
| Week 4 | Ablation Study + RQ2-3 |
| Week 5 | 분석 + 시각화 |
| Week 6 | 논문 작성 (집중) |

---

## 7. 논문 심사 대비 예상 질문

### 🎓 심사위원이 물어볼 것들

```markdown
## 예상 질문 및 답변 준비

### Q1: "왜 GraphMAE인가? GCN이나 GAT는 안 되나요?"
**답변 준비**:
- GraphMAE는 자기지도학습 → 레이블 불필요
- GCN/GAT는 지도학습 설계 → 클러스터링에 부적합
- 실험: GraphMAE vs GCN (unsupervised) 비교 추가 가능

---

### Q2: "BERT가 이미 문맥을 보는데, GraphMAE가 추가로 뭘 학습하나요?"
**답변 준비**:
- BERT: 단일 문서 내 지역 문맥
- GraphMAE: 전체 말뭉치의 전역 단어 관계
- 예시: "bank" → BERT는 문맥에 따라 다르지만, GraphMAE는
  "finance", "loan"과의 전역적 연결 학습

---

### Q3: "왜 K-means만 쓰나요? DBSCAN은?"
**답변 준비**:
- K-means: 가장 보편적, 비교 공정성
- DBSCAN: 밀도 기반, 클러스터 수 자동 결정
- 추가 실험: DBSCAN, Hierarchical 결과 Appendix에 포함

---

### Q4: "통계적으로 유의미한가요?"
**답변 준비**:
- 5회 반복 실험, paired t-test
- p < 0.01 수준에서 유의미
- 표에 p-value 명시

---

### Q5: "다른 언어(한국어)에서도 되나요?"
**답변 준비**:
- 현재 연구: 영어에 집중
- 한계점에 명시
- Future Work: 다국어 BERT (mBERT) 활용 계획

---

### Q6: "실제로 사용할 수 있나요? 너무 느린 것 아닌가요?"
**답변 준비**:
- GraphMAE 학습: 1회만 (사전학습)
- 추론 시: 빠름 (forward pass만)
- 비교: BERT fine-tuning보다 빠름
```

---

## 8. 추가 팁

### 📝 논문 작성 Best Practices

```markdown
## Writing Tips

### 문체
- 수동태 위주 (학술 논문 표준)
  - ❌ "We propose GRACE"
  - ✅ "GRACE is proposed"

- 명확하고 간결하게
  - ❌ "The experimental results clearly demonstrate that..."
  - ✅ "GRACE outperforms baselines by 15%"

### 그림/표
- 모든 그림/표에 캡션 필수
- 본문에서 참조 필수 ("Fig. 1 shows...")
- 고해상도 (최소 300 DPI)
- 컬러는 흑백 인쇄 시에도 구분 가능하게

### 수식
- 중요한 수식만 번호 매기기
- 변수 설명 필수
  - "where h_i ∈ R^d is the embedding of node i"

### 인용
- 주장에는 반드시 인용
  - ❌ "Text clustering is important"
  - ✅ "Text clustering is important [1, 2]"
- 최신 논문 우선 (5년 이내)
- 고전 논문도 포함 (K-means 등)

### 체크리스트
- [ ] 모든 약어 첫 등장 시 풀어쓰기
  - "GRACE (GRAph-based Clustering with Enhanced embeddings)"
- [ ] 일관된 용어 사용
  - "embedding" vs "representation" 혼용 금지
- [ ] 맞춤법/문법 검사 (Grammarly)
- [ ] 동료 리뷰 (다른 대학원생에게 읽어달라고)
```

---

## 9. 최종 체크리스트

### ✅ 제출 전 최종 확인

```markdown
## Final Checklist

### 내용
- [ ] 모든 RQ에 답변
- [ ] Contribution 명확히 제시
- [ ] Baseline과 비교
- [ ] 통계적 유의성 검증
- [ ] 정성적 분석 포함
- [ ] 한계점 솔직히 기술
- [ ] Future Work 구체적 제시

### 형식
- [ ] 초록 (국문/영문)
- [ ] 목차
- [ ] 그림/표 목록
- [ ] 참고문헌 형식 통일 (IEEE, ACM 등)
- [ ] 페이지 번호
- [ ] 헤더/푸터

### 기술
- [ ] 코드 GitHub 업로드
- [ ] README 작성 (실행 방법)
- [ ] requirements.txt
- [ ] 예제 실행 스크립트
- [ ] 데이터셋 다운로드 링크

### 윤리
- [ ] 표절 검사 (Turnitin 등)
- [ ] 데이터 사용 권한 확인
- [ ] 오픈소스 라이센스 명시
- [ ] 저자 기여도 명시 (단독 저자인 경우 생략)
```

---

## 요약

### 코드 수정 사항
당신이 제시한 계획에 **모두 포함됨**:
1. 전통 클러스터링 (DBSCAN, Hierarchical)
2. Baseline (TF-IDF, Word2Vec only 등)
3. Ablation Study 자동화
4. 재현성 (랜덤 시드)
5. 통계 검증
6. 시각화

**작업량**: 2-3주

---

### 논문 작성 시 핵심
**가장 중요**:
1. 연구 문제 명확화 (Problem-driven)
2. Baseline 비교 (효과 증명)
3. 통계 검증 (신뢰성)
4. 정성 분석 (설득력)

**작업량**: 3-4주 (실험 1주 + 작성 2-3주)

---

### 총 소요 시간
- **여유롭게**: 3개월
- **빡빡하게**: 6주
- **최소**: 4주 (비추천)

---

**이 문서를 논문 작성 시 체크리스트로 활용하세요!**
