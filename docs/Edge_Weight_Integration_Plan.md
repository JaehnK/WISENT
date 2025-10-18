# Edge Weight 통합 구현 계획서

> **목적**: GraphMAE2 모델에 공출현 빈도(edge weight) 정보를 통합하여 모델 성능 향상
> **날짜**: 2025-10-17
> **변경 범위**: GAT → GCN 전환 + Edge Weight 활성화

---

## 📋 Executive Summary

### 현재 문제점
- **GraphMAE2 모델이 edge weight를 사용하지 않음**
- WordGraph는 공출현 빈도를 `edge_attr`에 저장하지만, DGL 그래프 변환 시 제외됨
- GAT는 attention mechanism만 사용하고 edge weight를 직접 활용하지 않음

### 해결 방안
1. **Encoder/Decoder를 GAT → GCN으로 변경**
2. **GCN의 주석 처리된 edge weight 코드 활성화**
3. **DGL 그래프 변환 시 edge weight 포함**
4. **Edge weight normalization 추가**

### 예상 효과
- ✅ 공출현 빈도 정보 활용 → 의미적 관계 강도 반영
- ✅ 모델 표현력 향상 → 클러스터링 성능 개선
- ✅ 논문의 이론적 정당성 강화

---

## 🎯 구현 계획

### Phase 1: GCN Edge Weight 활성화

#### 파일: `core/GraphMAE2/models/gcn.py`

**현재 상태** (Line 126-157):
```python
def forward(self, graph, feat):
    with graph.local_scope():
        aggregate_fn = fn.copy_src('h', 'm')
        # if edge_weight is not None:
        #     assert edge_weight.shape[0] == graph.number_of_edges()
        #     graph.edata['_edge_weight'] = edge_weight
        #     aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
```

**변경 후**:
```python
def forward(self, graph, feat):
    with graph.local_scope():
        # Edge weight 지원 추가
        if 'weight' in graph.edata:
            aggregate_fn = fn.u_mul_e('h', 'weight', 'm')
        else:
            aggregate_fn = fn.copy_src('h', 'm')

        # ... 나머지 코드 동일
```

**변경 사유**:
- DGL 그래프의 edge data에 'weight' 키가 있으면 자동으로 사용
- Backward compatibility 유지 (weight 없으면 기존 방식)

---

### Phase 2: DGL 그래프 변환 시 Edge Weight 추가

#### 파일: `core/services/Graph/GraphService.py`

**현재 상태** (Line 148-189):
```python
def wordgraph_to_dgl(self, word_graph: 'WordGraph', node_features: Optional[torch.Tensor] = None):
    # ...
    # 엣지 가중치는 self-loop 추가 후 크기가 맞지 않으므로 설정하지 않음
    # GraphMAE는 엣지 가중치를 사용하지 않음

    return dgl_graph
```

**변경 후**:
```python
def wordgraph_to_dgl(self, word_graph: 'WordGraph', node_features: Optional[torch.Tensor] = None):
    # ... 기존 코드 ...

    # Self-loop 추가
    dgl_graph = dgl.add_self_loop(dgl_graph)

    # 노드 특성 설정
    if node_features is not None:
        dgl_graph.ndata['feat'] = node_features
    else:
        freq_features = torch.tensor([[word.freq] for word in word_graph.words], dtype=torch.float32)
        dgl_graph.ndata['feat'] = freq_features

    # ========== Edge Weight 추가 (새로운 부분) ==========
    if word_graph.edge_attr is not None:
        # 원본 엣지 가중치 추출
        original_edge_weights = word_graph.edge_attr.squeeze()  # [num_edges]

        # Self-loop 가중치 (노드 자신과의 연결은 최대값으로 설정)
        num_self_loops = word_graph.num_nodes
        self_loop_weights = torch.ones(num_self_loops, dtype=torch.float32)

        # 원본 가중치 + self-loop 가중치 결합
        all_edge_weights = torch.cat([original_edge_weights, self_loop_weights])

        # 정규화 (중요!)
        all_edge_weights = self._normalize_edge_weights(all_edge_weights)

        # DGL 그래프에 설정
        dgl_graph.edata['weight'] = all_edge_weights

    return dgl_graph

def _normalize_edge_weights(self, weights: torch.Tensor, method: str = 'minmax') -> torch.Tensor:
    """
    Edge weight 정규화

    Args:
        weights: 원본 가중치 [num_edges]
        method: 정규화 방법 ('minmax', 'log', 'standard')

    Returns:
        정규화된 가중치 [num_edges]
    """
    if method == 'minmax':
        # Min-Max scaling to [0, 1]
        min_val = weights.min()
        max_val = weights.max()
        if max_val - min_val > 0:
            return (weights - min_val) / (max_val - min_val)
        else:
            return weights

    elif method == 'log':
        # Log scaling (공출현 빈도는 power-law 분포 경향)
        return torch.log(weights + 1.0)

    elif method == 'standard':
        # Standardization (mean=0, std=1)
        mean = weights.mean()
        std = weights.std()
        if std > 0:
            return (weights - mean) / std
        else:
            return weights - mean

    else:
        return weights
```

**변경 사유**:
- Self-loop 추가 후에도 edge weight 차원 일치
- 정규화로 학습 안정성 확보
- 여러 정규화 방법 지원 (실험 가능)

---

### Phase 3: Config 변경 (Encoder/Decoder를 GCN으로)

#### 파일: `core/services/GraphMAE/GraphMAEConfig.py`

**현재 상태** (Line 33-34):
```python
encoder_type: str = "gat"
decoder_type: str = "gat"
```

**변경 후**:
```python
encoder_type: str = "gcn"
decoder_type: str = "gcn"
```

**추가 파라미터**:
```python
# Edge weight 관련 설정
edge_weight_normalization: str = "minmax"  # "minmax", "log", "standard", "none"
use_edge_weight: bool = True  # Edge weight 사용 여부 (ablation study용)
```

---

### Phase 4: Edge Weight Normalization 전략 결정

#### 옵션 비교

| 방법 | 수식 | 장점 | 단점 | 권장 |
|------|------|------|------|------|
| **Min-Max** | `(w - min) / (max - min)` | 해석 용이 [0,1] 범위 | Outlier에 민감 | ⭐ 기본 |
| **Log** | `log(w + 1)` | Power-law 분포 완화 | 0 근처 압축 | 공출현 빈도 특화 |
| **Standard** | `(w - μ) / σ` | 통계적 정규화 | 음수 값 가능 | GNN에는 부적합 |
| **None** | `w` | 정보 손실 없음 | 학습 불안정 | 비권장 |

**권장 전략**:
1. **기본값**: Min-Max (해석 용이, 안정적)
2. **Ablation Study**: Log scaling도 실험하여 비교

---

### Phase 5: Backward Compatibility 보장

#### GraphService에 옵션 추가

```python
def wordgraph_to_dgl(
    self,
    word_graph: 'WordGraph',
    node_features: Optional[torch.Tensor] = None,
    use_edge_weight: bool = True,  # 새로운 파라미터
    edge_weight_norm: str = 'minmax'  # 새로운 파라미터
):
    """
    WordGraph를 DGL 그래프로 변환

    Args:
        word_graph: 변환할 WordGraph 객체
        node_features: 노드 특성 텐서
        use_edge_weight: Edge weight 사용 여부 (기본 True)
        edge_weight_norm: 정규화 방법 ('minmax', 'log', 'standard', 'none')
    """
    # ... 구현 ...
```

**변경 사유**:
- 기존 코드와의 호환성 유지
- Ablation study 지원 (edge weight 유무 비교)

---

## 🧪 검증 계획

### 1. Unit Test: Edge Weight 로딩 확인

```python
# tests/test_edge_weight_integration.py
def test_dgl_graph_has_edge_weights():
    """DGL 그래프에 edge weight가 올바르게 추가되는지 확인"""
    word_graph = create_test_word_graph()
    dgl_graph = graph_service.wordgraph_to_dgl(word_graph, use_edge_weight=True)

    assert 'weight' in dgl_graph.edata
    assert dgl_graph.edata['weight'].shape[0] == dgl_graph.num_edges()
    assert torch.all(dgl_graph.edata['weight'] >= 0)
    assert torch.all(dgl_graph.edata['weight'] <= 1)  # Min-Max normalized
```

### 2. Integration Test: GCN Forward Pass

```python
def test_gcn_uses_edge_weights():
    """GCN이 edge weight를 실제로 사용하는지 확인"""
    dgl_graph = create_test_graph_with_weights()
    gcn_model = GCN(in_dim=128, num_hidden=64, out_dim=64, num_layers=2, ...)

    # Edge weight 있을 때
    output_with_weight = gcn_model(dgl_graph, node_features)

    # Edge weight 제거
    dgl_graph.edata.pop('weight')
    output_without_weight = gcn_model(dgl_graph, node_features)

    # 결과가 달라야 함
    assert not torch.allclose(output_with_weight, output_without_weight)
```

### 3. End-to-End Test: GraphMAE 학습

```python
def test_graphmae_with_edge_weights():
    """전체 파이프라인에서 edge weight가 작동하는지 확인"""
    config = GraphMAEConfig(
        encoder_type='gcn',
        decoder_type='gcn',
        use_edge_weight=True,
        edge_weight_normalization='minmax'
    )

    # 학습 실행
    embeddings = graphmae_service.pretrain_and_extract(word_graph, embed_size=64)

    # 임베딩이 생성되었는지 확인
    assert embeddings.shape == (word_graph.num_nodes, 64)
    assert not torch.isnan(embeddings).any()
```

---

## 📊 성능 비교 실험 설계

### Ablation Study 확장

기존 ablation study에 다음 실험 추가:

#### 실험 6: Edge Weight Ablation

```python
# AblationService.py에 추가
def run_edge_weight_ablation(self, num_runs: int = 5) -> Dict[str, Any]:
    """
    Edge weight 사용 유무에 따른 성능 비교

    실험 조건:
    1. GCN without edge weight (baseline)
    2. GCN with edge weight (Min-Max)
    3. GCN with edge weight (Log scaling)

    고정 변수:
    - encoder_type: gcn
    - decoder_type: gcn
    - embed_size: 64
    - mask_rate: 0.3
    - epochs: 1000
    """
    results = {}

    # 1. Without edge weight
    config_no_weight = GRACEConfig(
        encoder_type='gcn',
        decoder_type='gcn',
        use_edge_weight=False
    )
    results['no_weight'] = self._run_multiple_experiments(config_no_weight, num_runs)

    # 2. With edge weight (Min-Max)
    config_minmax = GRACEConfig(
        encoder_type='gcn',
        decoder_type='gcn',
        use_edge_weight=True,
        edge_weight_normalization='minmax'
    )
    results['minmax'] = self._run_multiple_experiments(config_minmax, num_runs)

    # 3. With edge weight (Log)
    config_log = GRACEConfig(
        encoder_type='gcn',
        decoder_type='gcn',
        use_edge_weight=True,
        edge_weight_normalization='log'
    )
    results['log'] = self._run_multiple_experiments(config_log, num_runs)

    # 통계적 검정
    stats = self._statistical_comparison(
        baseline=results['no_weight'],
        treatments={
            'minmax': results['minmax'],
            'log': results['log']
        }
    )

    return {
        'results': results,
        'statistics': stats
    }
```

### 예상 결과

| Configuration | Silhouette | Davies-Bouldin | NPMI | Improvement |
|---------------|------------|----------------|------|-------------|
| GCN (no weight) | 0.42 ± 0.03 | 1.25 ± 0.08 | 0.31 ± 0.02 | Baseline |
| GCN (Min-Max) | **0.48 ± 0.02** | **1.12 ± 0.06** | **0.36 ± 0.02** | **+14.3%** |
| GCN (Log) | 0.46 ± 0.03 | 1.18 ± 0.07 | 0.34 ± 0.03 | +9.5% |

**해석**:
- Edge weight 사용 시 모든 메트릭에서 유의미한 개선
- Min-Max normalization이 Log scaling보다 약간 우수
- NPMI 개선이 가장 큼 (공출현 정보가 직접 반영되므로)

---

## 🔄 롤백 계획

만약 edge weight 통합이 성능을 저하시키는 경우:

### 옵션 1: Config 변경으로 즉시 롤백
```python
# GraphMAEConfig.py
use_edge_weight: bool = False  # 즉시 비활성화
encoder_type: str = "gat"      # GAT로 복귀
decoder_type: str = "gat"
```

### 옵션 2: Git Revert
```bash
git revert <commit_hash>  # Edge weight 통합 커밋 되돌리기
```

### 실패 기준
다음 중 하나라도 발생 시 롤백 고려:
1. **학습 불안정**: Loss가 NaN이 되거나 발산
2. **성능 저하**: 모든 메트릭에서 5% 이상 하락
3. **통계적 유의성 없음**: p-value > 0.05

---

## 📝 구현 체크리스트

### Phase 1: Core Implementation
- [ ] `gcn.py`: Edge weight 지원 코드 작성 및 주석 해제
- [ ] `GraphService.py`: `_normalize_edge_weights()` 메서드 추가
- [ ] `GraphService.py`: `wordgraph_to_dgl()` 수정 (edge weight 추가)
- [ ] `GraphMAEConfig.py`: `encoder_type`, `decoder_type` 변경
- [ ] `GraphMAEConfig.py`: `use_edge_weight`, `edge_weight_normalization` 파라미터 추가

### Phase 2: Testing
- [ ] Unit test: `test_dgl_graph_has_edge_weights()`
- [ ] Unit test: `test_edge_weight_normalization()`
- [ ] Integration test: `test_gcn_uses_edge_weights()`
- [ ] End-to-end test: `test_graphmae_with_edge_weights()`

### Phase 3: Ablation Study
- [ ] `AblationService.py`: `run_edge_weight_ablation()` 추가
- [ ] 실험 실행 및 결과 수집
- [ ] 통계적 유의성 검정 (t-test, Cohen's d)

### Phase 4: Documentation
- [ ] 코드 주석 업데이트 (docstring)
- [ ] `GraphMAE2_Implementation_Report.md` 업데이트
- [ ] 논문 초안에 edge weight 정당화 섹션 추가

### Phase 5: Validation
- [ ] Baseline (GAT without edge weight) 재실행
- [ ] New approach (GCN with edge weight) 실행
- [ ] 성능 비교 및 분석
- [ ] 롤백 여부 결정

---

## 🎯 성공 기준

### 최소 성공 기준 (Must-Have)
1. ✅ GCN이 edge weight를 성공적으로 로드하고 사용
2. ✅ 학습이 안정적으로 수렴 (Loss < 1.0, no NaN)
3. ✅ 최소 하나의 메트릭에서 5% 이상 개선

### 이상적 성공 기준 (Should-Have)
1. ✅ 모든 메트릭(Silhouette, Davies-Bouldin, NPMI)에서 개선
2. ✅ 통계적으로 유의미한 개선 (p < 0.05, Cohen's d > 0.5)
3. ✅ NPMI에서 10% 이상 개선 (공출현 정보 활용 효과)

### 추가 가치 (Nice-to-Have)
1. ✅ 전통적 클러스터링(Louvain, Leiden) 대비 우위 확보
2. ✅ Computational cost 증가 < 20%
3. ✅ 다양한 정규화 방법 간 trade-off 분석

---

## 📅 예상 일정

| Phase | 작업 | 예상 시간 | 담당 |
|-------|------|----------|------|
| **Day 1** | Core Implementation | 4h | Claude + 사용자 |
| **Day 2** | Testing & Debugging | 3h | Claude + 사용자 |
| **Day 3** | Ablation Study 실행 | 6h | 자동 실행 |
| **Day 4** | 결과 분석 및 문서화 | 2h | 사용자 |
| **Day 5** | 검증 및 롤백 결정 | 1h | 사용자 |
| **Total** | | **16h** | |

---

## 🚨 리스크 및 대응 방안

### Risk 1: 학습 불안정 (Loss 발산)
**원인**: Edge weight 범위가 너무 큼
**대응**:
- Min-Max normalization 적용
- Learning rate 감소 (0.001 → 0.0005)
- Gradient clipping 추가

### Risk 2: 성능 저하
**원인**: Edge weight noise가 신호보다 큼
**대응**:
- Log scaling 시도 (power-law 완화)
- Edge pruning (낮은 가중치 제거)
- GAT와 GCN 앙상블 고려

### Risk 3: 계산 비용 증가
**원인**: Edge weight 처리 overhead
**대응**:
- Sparse matrix 최적화 활용
- Batch size 조정
- Early stopping 적용

---

## 📚 참고 문헌

### 이론적 배경
1. **GCN 원논문**: Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR.
2. **GraphMAE2**: Hou et al. (2022). "GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner." WWW.
3. **TextGCN**: Yao et al. (2019). "Graph Convolutional Networks for Text Classification." AAAI.

### 구현 참고
1. DGL Documentation: [Edge Weight in Message Passing](https://docs.dgl.ai/guide/message-passing.html#edge-weight-normalization)
2. PyTorch Geometric: [GCNConv with edge_weight](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)

---

## 💡 추가 개선 아이디어 (Future Work)

### 1. Learnable Edge Weight
```python
# Edge weight를 학습 가능한 파라미터로
edge_weight_mlp = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
learned_weights = edge_weight_mlp(original_weights)
```

### 2. Attention + Edge Weight 결합
```python
# GAT의 attention score와 edge weight를 결합
final_score = alpha * attention_score + (1 - alpha) * edge_weight
```

### 3. Edge Feature (Multi-dimensional)
```python
# 공출현 빈도 외에 추가 엣지 특성 사용
edge_features = [co_occurrence, pmi, cosine_similarity]  # 3차원
```

---

**작성자**: Claude (Anthropic)
**작성일**: 2025-10-17
**버전**: 1.0
**검토 필요**: 사용자 승인 후 구현 시작
