"""
Ablation Study Service

GRACE 컴포넌트별 기여도 분석
- GraphMAE 유/무
- Multimodal embedding (Word2Vec + BERT) vs 단일 임베딩
- 하이퍼파라미터 영향 (mask rate, embedding dimension, epochs)
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import json

from ..GRACE.GRACEConfig import GRACEConfig
from ..GRACE.GRACEPipeline import GRACEPipeline
from ..GRACE.ClusteringService import ClusteringService
from ..Metric import MetricsService
from ..Document.DocumentService import DocumentService
from ..Graph.GraphService import GraphService
from ..Graph.NodeFeatureHandler import NodeFeatureHandler
from entities import WordGraph, NodeFeatureType


class AblationService:
    """
    Ablation Study를 위한 서비스 클래스

    GRACE의 각 컴포넌트가 성능에 미치는 영향을 분석:
    1. Embedding Method Ablation: Word2Vec only, BERT only, Multimodal
    2. GraphMAE Ablation: GraphMAE 유/무
    3. Hyperparameter Ablation: mask_rate, embed_size, epochs
    """

    def __init__(self, base_config: GRACEConfig, random_state: int = 42):
        """
        Args:
            base_config: 기본 GRACE 설정 (baseline)
            random_state: 재현성을 위한 랜덤 시드
        """
        self.base_config = base_config
        self.random_state = random_state

        # 공유 데이터 (모든 ablation 실험에서 동일하게 사용)
        self.doc_service: Optional[DocumentService] = None
        self.word_graph: Optional[WordGraph] = None

        # 결과 저장
        self.ablation_results: Dict[str, Any] = {}

    # ============================================================
    # 1. 공통 전처리 (한 번만 실행)
    # ============================================================

    def prepare_shared_data(self) -> None:
        """
        모든 ablation 실험에서 공유할 데이터 준비
        - Documents 로드 및 전처리
        - Word Graph 구축
        """
        print("📁 공유 데이터 준비 중...")

        # DocumentService 초기화
        pipeline = GRACEPipeline(self.base_config)
        pipeline.load_and_preprocess()
        pipeline.build_semantic_network()

        self.doc_service = pipeline.doc_service
        self.word_graph = pipeline.word_graph

        print(f"  ✅ 문서: {pipeline.doc_service.get_sentence_count()}개")
        print(f"  ✅ 그래프: {self.word_graph.num_nodes} nodes, {self.word_graph.num_edges} edges")

    # ============================================================
    # 2. Embedding Method Ablation
    # ============================================================

    def run_embedding_ablation(self) -> Dict[str, Dict[str, float]]:
        """
        임베딩 방법별 성능 비교

        실험:
        - Word2Vec only
        - BERT only
        - Multimodal (Word2Vec + BERT)

        Returns:
            {embedding_method: {metric: value}}
        """
        if self.doc_service is None or self.word_graph is None:
            raise RuntimeError("prepare_shared_data()를 먼저 호출하세요.")

        print("\n" + "=" * 80)
        print("🔬 Ablation Study 1: Embedding Method")
        print("=" * 80)

        results = {}
        embedding_methods = [
            ('w2v', "Word2Vec only"),
            ('bert', "BERT only"),
            ('concat', "Multimodal (Word2Vec + BERT)")
        ]

        for method, desc in embedding_methods:
            print(f"\n[{desc}]")
            config = self._create_config_variant(embedding_method=method)
            metrics = self._run_experiment(config, use_graphmae=True)
            results[method] = metrics
            self._print_metrics(metrics)

        self.ablation_results['embedding_method'] = results
        return results

    # ============================================================
    # 3. GraphMAE Ablation
    # ============================================================

    def run_graphmae_ablation(self) -> Dict[str, Dict[str, float]]:
        """
        GraphMAE 유/무에 따른 성능 비교

        실험:
        - Without GraphMAE (멀티모달 임베딩만 사용)
        - With GraphMAE (GRACE 전체 파이프라인)

        Returns:
            {graphmae_status: {metric: value}}
        """
        if self.doc_service is None or self.word_graph is None:
            raise RuntimeError("prepare_shared_data()를 먼저 호출하세요.")

        print("\n" + "=" * 80)
        print("🔬 Ablation Study 2: GraphMAE Effect")
        print("=" * 80)

        results = {}

        # Without GraphMAE
        print(f"\n[Without GraphMAE]")
        config = self._create_config_variant()
        metrics_without = self._run_experiment(config, use_graphmae=False)
        results['without_graphmae'] = metrics_without
        self._print_metrics(metrics_without)

        # With GraphMAE
        print(f"\n[With GraphMAE]")
        metrics_with = self._run_experiment(config, use_graphmae=True)
        results['with_graphmae'] = metrics_with
        self._print_metrics(metrics_with)

        # 개선율 계산
        improvement = self._calculate_improvement(metrics_without, metrics_with)
        print(f"\n📈 Improvement:")
        for metric, pct in improvement.items():
            print(f"  {metric}: {pct:+.2f}%")

        results['improvement'] = improvement
        self.ablation_results['graphmae'] = results
        return results

    # ============================================================
    # 4. Hyperparameter Ablation
    # ============================================================

    def run_mask_rate_ablation(
        self,
        mask_rates: Optional[List[float]] = None,
        num_runs: int = 1
    ) -> Dict[float, Dict[str, float]]:
        """
        GraphMAE mask rate에 따른 성능 변화

        Args:
            mask_rates: 테스트할 mask rate 리스트 (기본: [0.3, 0.5, 0.75, 0.9])
            num_runs: 각 mask rate당 반복 실행 횟수 (기본: 1)

        Returns:
            {mask_rate: {metric: value}} 또는
            {mask_rate: {metric_mean: value, metric_std: value}} (num_runs > 1)
        """
        if self.doc_service is None or self.word_graph is None:
            raise RuntimeError("prepare_shared_data()를 먼저 호출하세요.")

        if mask_rates is None:
            mask_rates = [0.3, 0.5, 0.75, 0.9]

        print("\n" + "=" * 80)
        print("🔬 Ablation Study 3: Mask Rate")
        if num_runs > 1:
            print(f"   (각 mask rate당 {num_runs}회 반복 실행)")
        print("=" * 80)

        results = {}

        for mask_rate in mask_rates:
            print(f"\n[Mask Rate = {mask_rate}]")
            config = self._create_config_variant(mask_rate=mask_rate)

            if num_runs == 1:
                # 단일 실행
                metrics = self._run_experiment(config, use_graphmae=True)
                results[mask_rate] = metrics
                self._print_metrics(metrics)
            else:
                # 다중 실행
                all_runs = []
                for run_idx in range(num_runs):
                    print(f"  Run {run_idx + 1}/{num_runs}...", end=" ")

                    # 매 실행마다 다른 시드 사용 (재현성 유지하면서 변동성 측정)
                    config_variant = self._create_config_variant(mask_rate=mask_rate)
                    metrics = self._run_experiment(config_variant, use_graphmae=True)
                    all_runs.append(metrics)
                    print(f"Silhouette={metrics['silhouette']:.4f}")

                # 평균과 표준편차 계산
                aggregated = self._aggregate_multiple_runs(all_runs)
                results[mask_rate] = aggregated

                print(f"\n  📊 집계 결과:")
                self._print_aggregated_metrics(aggregated)

        self.ablation_results['mask_rate'] = results
        return results

    def run_embed_size_ablation(
        self,
        embed_sizes: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        임베딩 차원에 따른 성능 변화

        Args:
            embed_sizes: 테스트할 임베딩 차원 리스트 (기본: [32, 64, 128, 256])

        Returns:
            {embed_size: {metric: value}}
        """
        if self.doc_service is None or self.word_graph is None:
            raise RuntimeError("prepare_shared_data()를 먼저 호출하세요.")

        if embed_sizes is None:
            embed_sizes = [32, 64, 128, 256]

        print("\n" + "=" * 80)
        print("🔬 Ablation Study 4: Embedding Dimension")
        print("=" * 80)

        results = {}

        for embed_size in embed_sizes:
            print(f"\n[Embed Size = {embed_size}]")
            config = self._create_config_variant(
                embed_size=embed_size,
                w2v_dim=embed_size // 2,
                bert_dim=embed_size // 2
            )
            metrics = self._run_experiment(config, use_graphmae=True)
            results[embed_size] = metrics
            self._print_metrics(metrics)

        self.ablation_results['embed_size'] = results
        return results

    def run_epochs_ablation(
        self,
        epochs_list: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        GraphMAE 학습 epoch에 따른 성능 변화

        Args:
            epochs_list: 테스트할 epoch 리스트 (기본: [50, 100, 200, 300])

        Returns:
            {epochs: {metric: value}}
        """
        if self.doc_service is None or self.word_graph is None:
            raise RuntimeError("prepare_shared_data()를 먼저 호출하세요.")

        if epochs_list is None:
            epochs_list = [50, 100, 200, 300]

        print("\n" + "=" * 80)
        print("🔬 Ablation Study 5: GraphMAE Epochs")
        print("=" * 80)

        results = {}

        for epochs in epochs_list:
            print(f"\n[Epochs = {epochs}]")
            config = self._create_config_variant(graphmae_epochs=epochs)
            metrics = self._run_experiment(config, use_graphmae=True)
            results[epochs] = metrics
            self._print_metrics(metrics)

        self.ablation_results['epochs'] = results
        return results

    # ============================================================
    # 5. 전체 Ablation Study 실행
    # ============================================================

    def run_full_ablation_study(
        self,
        include_embedding: bool = True,
        include_graphmae: bool = True,
        include_mask_rate: bool = True,
        include_embed_size: bool = False,
        include_epochs: bool = False
    ) -> Dict[str, Any]:
        """
        전체 Ablation Study 실행

        Args:
            include_embedding: 임베딩 방법 ablation 실행 여부
            include_graphmae: GraphMAE ablation 실행 여부
            include_mask_rate: Mask rate ablation 실행 여부
            include_embed_size: 임베딩 차원 ablation 실행 여부
            include_epochs: Epochs ablation 실행 여부

        Returns:
            전체 ablation 결과
        """
        print("\n" + "=" * 80)
        print("🚀 GRACE Ablation Study 시작")
        print("=" * 80)

        start_time = datetime.now()

        # 공유 데이터 준비
        self.prepare_shared_data()

        # 각 ablation study 실행
        if include_embedding:
            self.run_embedding_ablation()

        if include_graphmae:
            self.run_graphmae_ablation()

        if include_mask_rate:
            self.run_mask_rate_ablation()

        if include_embed_size:
            self.run_embed_size_ablation()

        if include_epochs:
            self.run_epochs_ablation()

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print("\n" + "=" * 80)
        print(f"✅ Ablation Study 완료 (소요시간: {elapsed:.2f}초)")
        print("=" * 80)

        return self.ablation_results

    # ============================================================
    # 6. 결과 저장 및 시각화
    # ============================================================

    def save_results(self, output_dir: str = "./ablation_output") -> None:
        """
        Ablation 결과 저장

        Args:
            output_dir: 저장 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 저장
        json_path = output_path / f"ablation_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # numpy/torch 타입을 Python 네이티브 타입으로 변환
            serializable_results = self._make_serializable(self.ablation_results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 결과 저장: {json_path}")

    # ============================================================
    # 헬퍼 메서드
    # ============================================================

    def _create_config_variant(self, **kwargs) -> GRACEConfig:
        """
        base_config를 복사하고 특정 파라미터만 변경한 설정 생성

        Args:
            **kwargs: 변경할 파라미터

        Returns:
            변형된 GRACEConfig
        """
        import copy
        config = copy.deepcopy(self.base_config)

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # 결과 저장 비활성화 (ablation 결과만 저장)
        config.save_results = False
        config.save_embeddings = False
        config.save_graph_viz = False
        config.verbose = False

        return config

    def _run_experiment(
        self,
        config: GRACEConfig,
        use_graphmae: bool = True
    ) -> Dict[str, float]:
        """
        단일 실험 실행 및 메트릭 반환

        Args:
            config: 실험 설정
            use_graphmae: GraphMAE 사용 여부

        Returns:
            {metric_name: value}
        """
        # NodeFeatureHandler로 임베딩 계산
        node_feature_handler = NodeFeatureHandler(self.doc_service)
        node_features = node_feature_handler.calculate_embeddings(
            self.word_graph.words,
            method=config.embedding_method,
            embed_size=config.embed_size
        )

        # GraphMAE 사용 여부에 따라 분기
        if use_graphmae:
            # GraphMAE로 임베딩 학습
            from ..GraphMAE import GraphMAEService, GraphMAEConfig

            embed_size = node_features.shape[1]
            mae_config = GraphMAEConfig.create_default(embed_size)
            mae_config.max_epochs = config.graphmae_epochs
            mae_config.learning_rate = config.graphmae_lr
            mae_config.mask_rate = config.mask_rate
            mae_config.device = config.graphmae_device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            graph_service = GraphService(self.doc_service)
            mae_service = GraphMAEService(graph_service, mae_config)

            # WordGraph에 노드 특성 설정
            self.word_graph.set_node_features_custom(node_features, NodeFeatureType.CUSTOM)

            # DGL 그래프 변환
            dgl_graph = graph_service.wordgraph_to_dgl(self.word_graph, node_features)

            # 모델 학습
            mae_service.model = mae_service.create_mae_model(embed_size)
            device = torch.device(mae_config.device)
            mae_service.model.to(device)
            dgl_graph = dgl_graph.to(device)

            optimizer = torch.optim.Adam(
                mae_service.model.parameters(),
                lr=mae_config.learning_rate,
                weight_decay=mae_config.weight_decay
            )

            mae_service.model.train()
            for epoch in range(mae_config.max_epochs):
                optimizer.zero_grad()
                x = dgl_graph.ndata['feat']
                loss = mae_service.model(dgl_graph, x, epoch=epoch)
                loss.backward()
                optimizer.step()

            # 임베딩 추출
            mae_service.model.eval()
            with torch.no_grad():
                embeddings = mae_service.model.embed(dgl_graph, dgl_graph.ndata['feat'])
            embeddings = embeddings.cpu()
        else:
            # GraphMAE 없이 멀티모달 임베딩만 사용
            embeddings = node_features

        # 클러스터링
        clustering_service = ClusteringService(random_state=self.random_state)

        if config.num_clusters is not None:
            labels = clustering_service.kmeans_clustering(embeddings, n_clusters=config.num_clusters)
        else:
            labels, best_k, _, _ = clustering_service.auto_clustering_elbow(
                embeddings,
                min_clusters=config.min_clusters,
                max_clusters=config.max_clusters
            )

        # 평가 지표 계산
        metrics_service = MetricsService()
        metrics = metrics_service.calculate_metrics(
            embeddings,
            labels,
            ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        )

        return metrics

    def _calculate_improvement(
        self,
        baseline: Dict[str, float],
        improved: Dict[str, float]
    ) -> Dict[str, float]:
        """
        두 결과 간 개선율 계산

        Args:
            baseline: 기준 결과
            improved: 개선된 결과

        Returns:
            {metric: improvement_percentage}
        """
        improvement = {}

        for metric in baseline.keys():
            if metric not in improved:
                continue

            base_val = baseline[metric]
            improved_val = improved[metric]

            # davies_bouldin은 낮을수록 좋으므로 부호 반전
            if metric == 'davies_bouldin':
                pct = ((base_val - improved_val) / base_val) * 100
            else:
                pct = ((improved_val - base_val) / base_val) * 100

            improvement[metric] = pct

        return improvement

    def _print_metrics(self, metrics: Dict[str, float]) -> None:
        """메트릭 출력"""
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    def _aggregate_multiple_runs(self, runs: List[Dict[str, float]]) -> Dict[str, float]:
        """
        여러 번 실행한 결과를 집계 (평균 ± 표준편차)

        Args:
            runs: [{metric: value}, {metric: value}, ...]

        Returns:
            {metric_mean: value, metric_std: value, ...}
        """
        if not runs:
            return {}

        # 모든 메트릭 이름 추출
        metric_names = list(runs[0].keys())
        aggregated = {}

        for metric in metric_names:
            values = [run[metric] for run in runs]
            aggregated[f"{metric}_mean"] = float(np.mean(values))
            aggregated[f"{metric}_std"] = float(np.std(values))
            aggregated[f"{metric}_min"] = float(np.min(values))
            aggregated[f"{metric}_max"] = float(np.max(values))

        # 실행 횟수 추가
        aggregated['num_runs'] = len(runs)

        return aggregated

    def _print_aggregated_metrics(self, aggregated: Dict[str, float]) -> None:
        """집계된 메트릭 출력 (평균 ± 표준편차)"""
        metric_names = set()
        for key in aggregated.keys():
            if key.endswith('_mean'):
                metric_names.add(key.replace('_mean', ''))

        for metric in sorted(metric_names):
            mean = aggregated.get(f"{metric}_mean", 0)
            std = aggregated.get(f"{metric}_std", 0)
            min_val = aggregated.get(f"{metric}_min", 0)
            max_val = aggregated.get(f"{metric}_max", 0)

            print(f"     {metric:20s}: {mean:.4f} ± {std:.4f}  (range: [{min_val:.4f}, {max_val:.4f}])")

    def _make_serializable(self, obj: Any) -> Any:
        """
        JSON 직렬화를 위해 numpy/torch 타입을 Python 네이티브 타입으로 변환

        Args:
            obj: 변환할 객체

        Returns:
            직렬화 가능한 객체
        """
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return obj
