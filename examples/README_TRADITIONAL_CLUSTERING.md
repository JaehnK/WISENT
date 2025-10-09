# Traditional Graph Clustering 사용법

GRACE와 전통적 그래프 클러스터링 방법을 비교하기 위한 가이드입니다.

## 개요

`TraditionalGraphClusteringService`는 그래프 구조만을 사용하는 전통적인 커뮤니티 탐지 알고리즘을 제공합니다:

1. **Louvain**: Modularity 최적화 기반 (빠름)
2. **Leiden**: Louvain 개선판 (더 정확함)
3. **Girvan-Newman**: Edge betweenness 기반 (느리지만 계층적)

## 설치 요구사항

```bash
# 기본 패키지
pip install networkx scikit-learn numpy

# Louvain
pip install python-louvain

# Leiden
pip install leidenalg python-igraph
```

## 기본 사용법

### 1. 독립적으로 사용

```python
from core.services.GRACE import TraditionalGraphClusteringService
from core.services.Document import DocumentService
from core.services.Graph import GraphService

# 1. 데이터 전처리
doc_service = DocumentService()
doc_service.load_from_csv('data.csv', text_column='text', num_documents=1000)
doc_service.process_documents()

# 2. 의미연결망 구축
graph_service = GraphService(doc_service)
word_graph = graph_service.build_complete_graph(
    top_n=500,
    exclude_stopwords=True
)

# 3. 클러스터링 수행
traditional_clustering = TraditionalGraphClusteringService(random_state=42)

# Louvain
labels, metrics = traditional_clustering.louvain_clustering(word_graph)
print(f"Clusters: {metrics['num_clusters']}")
print(f"Modularity: {metrics['modularity']:.4f}")

# Leiden
labels, metrics = traditional_clustering.leiden_clustering(word_graph)

# Girvan-Newman (작은 그래프에만 권장)
labels, metrics = traditional_clustering.girvan_newman_clustering(
    word_graph,
    num_clusters=10  # 또는 None (자동)
)
```

### 2. GRACE와 비교

```python
from core.services.GRACE import GRACEPipeline, GRACEConfig, TraditionalGraphClusteringService

# GRACE 실행
config = GRACEConfig(
    csv_path='data.csv',
    num_documents=1000,
    top_n_words=500
)
pipeline = GRACEPipeline(config)
grace_results = pipeline.run()

# 같은 WordGraph로 전통적 클러스터링
word_graph = pipeline.word_graph
traditional = TraditionalGraphClusteringService(random_state=42)
louvain_labels, louvain_metrics = traditional.louvain_clustering(word_graph)

# 비교
print("GRACE Silhouette:", grace_results['metrics']['silhouette'])
print("Louvain Clusters:", louvain_metrics['num_clusters'])
print("Louvain Modularity:", louvain_metrics['modularity'])
```

### 3. 전체 비교 실험 실행

```bash
python examples/compare_grace_with_traditional.py
```

이 스크립트는:
- GRACE (GraphMAE)
- Louvain
- Leiden
- Girvan-Newman (작은 그래프)

을 모두 실행하고 비교 결과를 CSV/JSON으로 저장합니다.

## 결과 분석

### 클러스터 분포 확인

```python
distribution = traditional.get_cluster_distribution()
print(distribution)
# {0: 120, 1: 85, 2: 95, ...}
```

### 클러스터별 상위 단어

```python
cluster_words = traditional.get_cluster_words(word_graph, cluster_id=0, top_n=10)
for word, freq in cluster_words:
    print(f"{word}: {freq}")
```

### 모든 클러스터 단어

```python
all_clusters = traditional.get_all_cluster_words(word_graph, top_n=10)
for cluster_id, words in all_clusters.items():
    print(f"Cluster {cluster_id}:")
    print([w for w, _ in words])
```

### 그래프 통계

```python
stats = traditional.compute_graph_statistics()
print(f"Density: {stats['density']:.4f}")
print(f"Average Degree: {stats['average_degree']:.2f}")
print(f"Connected: {stats['is_connected']}")
```

### 결과 저장

```python
traditional.save_clustering_results(
    word_graph,
    output_path='./results/louvain_results.json',
    include_words=True
)
```

## 논문 작성 시 활용

### Research Question

> **RQ1**: 전통적 그래프 클러스터링(구조만 활용)과 GRACE(구조 + GraphMAE 학습)의 성능 차이는?

### 비교 항목

1. **클러스터 품질 지표**
   - Silhouette Score (높을수록 좋음)
   - Davies-Bouldin Index (낮을수록 좋음)
   - Calinski-Harabasz Score (높을수록 좋음)

2. **주제 일관성**
   - NPMI (Normalized PMI)

3. **그래프 특화 지표**
   - Modularity (커뮤니티 구조 품질)

4. **계산 시간**
   - 각 방법의 실행 시간 비교

### 예상 결과 (논문)

| Method | Silhouette ↑ | Davies-Bouldin ↓ | Modularity ↑ | NPMI ↑ |
|--------|--------------|-------------------|--------------|--------|
| Louvain | 0.45 | 1.23 | **0.65** | 0.32 |
| Leiden | 0.47 | 1.18 | **0.67** | 0.35 |
| Girvan-Newman | 0.42 | 1.35 | 0.58 | 0.28 |
| **GRACE** | **0.62** | **0.78** | N/A | **0.56** |

**분석**:
- 전통적 방법은 높은 Modularity를 보이지만, 클러스터 품질은 낮음
- GRACE는 GraphMAE 학습을 통해 의미적 클러스터 생성
- Modularity는 그래프 구조만 평가하므로 GRACE에는 적합하지 않음

## 주의사항

### Girvan-Newman 계산 비용

- 시간 복잡도: O(m²n) 또는 O(n³)
- 권장 그래프 크기: 노드 < 300
- 큰 그래프에서는 스킵 권장

```python
if word_graph.num_nodes <= 300:
    gn_labels, gn_metrics = traditional.girvan_newman_clustering(word_graph)
else:
    print("Graph too large, skipping Girvan-Newman")
```

### 패키지 설치 실패 시

#### Leiden/igraph 설치 오류

```bash
# Ubuntu/Debian
sudo apt-get install libigraph-dev

# macOS
brew install igraph

# 그 후 다시 시도
pip install python-igraph leidenalg
```

#### Louvain 설치 오류

```bash
pip install python-louvain --no-cache-dir
```

## 추가 자료

- [Louvain 논문](https://arxiv.org/abs/0803.0476)
- [Leiden 논문](https://www.nature.com/articles/s41598-019-41695-z)
- [Girvan-Newman 논문](https://arxiv.org/abs/cond-mat/0112110)
- [NetworkX Community Detection](https://networkx.org/documentation/stable/reference/algorithms/community.html)
