"""
GRACE (GRAph-based Clustering with Enhanced embeddings) Pipeline

전체 파이프라인:
1. 데이터 전처리 (DocumentService)
2. 의미연결망 구축 (GraphService)
3. 멀티모달 임베딩 (Word2Vec + BERT)
4. GraphMAE 자기지도학습
5. 클러스터링 (K-means, DBSCAN 등)
6. 평가 및 결과 분석
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

# 서비스 import
from ..Document.DocumentService import DocumentService
from ..Graph.GraphService import GraphService
from ..Graph.NodeFeatureHandler import NodeFeatureHandler
from ..GraphMAE import GraphMAEService, GraphMAEConfig
from entities import Word, WordGraph, NodeFeatureType

from .GRACEConfig import GRACEConfig


class GRACEPipeline:
    """GRACE 파이프라인 오케스트레이터"""

    def __init__(self, config: GRACEConfig):
        """
        Args:
            config: GRACE 파이프라인 설정
        """
        self.config = config

        # 서비스 인스턴스
        self.doc_service: Optional[DocumentService] = None
        self.graph_service: Optional[GraphService] = None
        self.node_feature_handler: Optional[NodeFeatureHandler] = None

        # 중간 결과 저장
        self.documents: Optional[List[str]] = None
        self.word_graph: Optional[WordGraph] = None
        self.node_features: Optional[torch.Tensor] = None
        self.graphmae_embeddings: Optional[torch.Tensor] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_results: Optional[Dict[str, Any]] = None

        # 메타데이터
        self.pipeline_start_time: Optional[datetime] = None
        self.pipeline_end_time: Optional[datetime] = None

    # ============================================================
    # 메인 실행 메서드
    # ============================================================

    def run(self) -> Dict[str, Any]:
        """
        전체 파이프라인 실행

        Returns:
            최종 결과 딕셔너리
        """
        self.pipeline_start_time = datetime.now()
        self._log("🚀 GRACE 파이프라인 시작")
        self._log("=" * 80)

        try:
            # 1. 데이터 로딩 및 전처리
            self._log("\n[1/6] 데이터 로딩 및 전처리")
            self.load_and_preprocess()

            # 2. 의미연결망 구축
            self._log("\n[2/6] 의미연결망 구축")
            self.build_semantic_network()

            # 3. 멀티모달 노드 특성 계산
            self._log("\n[3/6] 멀티모달 임베딩 계산")
            self.compute_node_features()

            # 4. GraphMAE 자기지도학습
            self._log("\n[4/6] GraphMAE 자기지도학습")
            self.train_graphmae()

            # 5. 클러스터링
            self._log("\n[5/6] 클러스터링 수행")
            self.perform_clustering()

            # 6. 평가 및 결과 저장
            self._log("\n[6/6] 평가 및 결과 저장")
            results = self.evaluate_and_save()

            self.pipeline_end_time = datetime.now()
            elapsed = (self.pipeline_end_time - self.pipeline_start_time).total_seconds()

            self._log("\n" + "=" * 80)
            self._log(f"✅ GRACE 파이프라인 완료 (소요시간: {elapsed:.2f}초)")
            self._log("=" * 80)

            return results

        except Exception as e:
            self._log(f"\n❌ 파이프라인 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    # ============================================================
    # 1. 데이터 로딩 및 전처리
    # ============================================================

    def load_and_preprocess(self) -> None:
        """데이터 로딩 및 전처리"""
        # CSV 로드
        self._log(f"📁 데이터셋 로드: {self.config.csv_path}")

        if not os.path.exists(self.config.csv_path):
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {self.config.csv_path}")

        df = pd.read_csv(self.config.csv_path)
        self._log(f"  전체 데이터: {len(df)} 행")

        if self.config.text_column not in df.columns:
            raise ValueError(f"CSV에 '{self.config.text_column}' 컬럼이 없습니다.")

        # 문서 추출
        self.documents = df[self.config.text_column].dropna().head(
            self.config.num_documents
        ).tolist()
        self._log(f"  로드된 문서: {len(self.documents)}개")

        # DocumentService 초기화 및 전처리
        self.doc_service = DocumentService()
        self.doc_service.create_sentence_list(documents=self.documents)
        self._log(f"  전처리 완료: {self.doc_service.get_sentence_count()}개 문장")
        self._log(f"  추출된 단어: {len(self.doc_service.words_list)}개")

    # ============================================================
    # 2. 의미연결망 구축
    # ============================================================

    def build_semantic_network(self) -> None:
        """공출현 기반 의미연결망 구축"""
        if self.doc_service is None:
            raise RuntimeError("DocumentService가 초기화되지 않았습니다. load_and_preprocess()를 먼저 호출하세요.")

        # GraphService 초기화
        self.graph_service = GraphService(self.doc_service)

        # 공출현 그래프 생성
        self.word_graph = self.graph_service.build_complete_graph(
            top_n=self.config.top_n_words,
            exclude_stopwords=self.config.exclude_stopwords,
            max_length=self.config.max_sentence_length
        )

        self._log(f"  노드 수: {self.word_graph.num_nodes}")
        self._log(f"  엣지 수: {self.word_graph.num_edges}")

    # ============================================================
    # 3. 멀티모달 노드 특성 계산
    # ============================================================

    def compute_node_features(self) -> None:
        """멀티모달 임베딩 계산 (Word2Vec + BERT)"""
        if self.word_graph is None or self.graph_service is None:
            raise RuntimeError("WordGraph가 생성되지 않았습니다. build_semantic_network()를 먼저 호출하세요.")

        self.node_feature_handler = NodeFeatureHandler(self.doc_service)

        method_desc = {
            'concat': f"Word2Vec({self.config.w2v_dim}) + BERT({self.config.bert_dim})",
            'w2v': f"Word2Vec({self.config.embed_size})",
            'bert': f"BERT({self.config.embed_size})"
        }
        self._log(f"  임베딩 방법: {method_desc[self.config.embedding_method]}")

        self.node_features = self.node_feature_handler.calculate_embeddings(
            self.word_graph.words,
            method=self.config.embedding_method,
            embed_size=self.config.embed_size
        )

        self._log(f"  노드 특성 형태: {self.node_features.shape}")

        # WordGraph에 설정
        self.word_graph.set_node_features_custom(self.node_features, NodeFeatureType.CUSTOM)

    # ============================================================
    # 4. GraphMAE 자기지도학습
    # ============================================================

    def train_graphmae(self) -> None:
        """GraphMAE 사전학습 및 임베딩 추출"""
        if self.word_graph is None or self.graph_service is None:
            raise RuntimeError("WordGraph가 생성되지 않았습니다.")

        if self.word_graph.node_features is None:
            raise RuntimeError("노드 특성이 계산되지 않았습니다. compute_node_features()를 먼저 호출하세요.")

        embed_size = self.word_graph.node_features.shape[1]

        # GraphMAE 설정
        mae_config = GraphMAEConfig.create_default(embed_size)
        mae_config.max_epochs = self.config.graphmae_epochs
        mae_config.learning_rate = self.config.graphmae_lr
        mae_config.weight_decay = self.config.graphmae_weight_decay
        mae_config.mask_rate = self.config.mask_rate
        mae_config.device = self.config.graphmae_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._log(f"  학습 설정: {mae_config.max_epochs} epochs, device: {mae_config.device}")
        if torch.cuda.is_available() and mae_config.device == "cuda":
            self._log(f"  GPU: {torch.cuda.get_device_name(0)}")

        # GraphMAE 서비스 초기화
        mae_service = GraphMAEService(self.graph_service, mae_config)

        # DGL 그래프로 변환
        dgl_graph = self.graph_service.wordgraph_to_dgl(
            self.word_graph, self.word_graph.node_features
        )

        # 모델 생성 및 학습
        mae_service.model = mae_service.create_mae_model(embed_size)
        device_obj = torch.device(mae_config.device)
        mae_service.model.to(device_obj)
        dgl_graph = dgl_graph.to(device_obj)

        optimizer = torch.optim.Adam(
            mae_service.model.parameters(),
            lr=mae_config.learning_rate,
            weight_decay=mae_config.weight_decay
        )

        # 학습 루프
        mae_service.model.train()
        for epoch in range(mae_config.max_epochs):
            optimizer.zero_grad()
            x = dgl_graph.ndata['feat']
            loss = mae_service.model(dgl_graph, x, epoch=epoch)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.config.log_interval == 0:
                self._log(f"  Epoch {epoch + 1}/{mae_config.max_epochs}, Loss: {loss.item():.4f}")

        # 임베딩 추출
        mae_service.model.eval()
        with torch.no_grad():
            self.graphmae_embeddings = mae_service.model.embed(dgl_graph, dgl_graph.ndata['feat'])

        self.graphmae_embeddings = self.graphmae_embeddings.cpu()
        self._log(f"  GraphMAE 임베딩 추출 완료: {self.graphmae_embeddings.shape}")

    # ============================================================
    # 5. 클러스터링
    # ============================================================

    def perform_clustering(self) -> None:
        """클러스터링 수행"""
        if self.graphmae_embeddings is None:
            raise RuntimeError("GraphMAE 임베딩이 없습니다. train_graphmae()를 먼저 호출하세요.")

        # TODO: ClusteringService 구현 후 연동
        # 현재는 기본 K-means만 구현
        from sklearn.cluster import KMeans

        embeddings_np = self.graphmae_embeddings.numpy()

        if self.config.num_clusters is not None:
            # 지정된 클러스터 수로 실행
            n_clusters = self.config.num_clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = kmeans.fit_predict(embeddings_np)
            self._log(f"  K-means 완료: {n_clusters}개 클러스터")
        else:
            # Elbow Method로 최적 클러스터 수 탐색
            self._log(f"  Elbow Method로 최적 클러스터 수 탐색 중 ({self.config.min_clusters}-{self.config.max_clusters})...")

            inertias = []
            silhouette_scores = []
            k_range = range(self.config.min_clusters, self.config.max_clusters + 1)

            from sklearn.metrics import silhouette_score

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings_np)
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(embeddings_np, labels))

            # Elbow point 찾기 (inertia 감소율이 급격히 줄어드는 지점)
            best_k = self._find_elbow_point(list(k_range), inertias)

            # 최적 k로 최종 클러스터링
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            self.cluster_labels = kmeans.fit_predict(embeddings_np)

            self._log(f"  Elbow Point: k={best_k}")
            self._log(f"  최적 클러스터 수: {best_k} (Silhouette: {silhouette_scores[best_k - self.config.min_clusters]:.4f})")

            # Elbow curve 시각화 저장
            if self.config.save_graph_viz:
                self._save_elbow_curve(list(k_range), inertias, silhouette_scores, best_k)

        # 클러스터별 단어 분포
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        self._log(f"  클러스터별 단어 수: {dict(zip(unique, counts))}")

    # ============================================================
    # 6. 평가 및 결과 저장
    # ============================================================

    def evaluate_and_save(self) -> Dict[str, Any]:
        """평가 지표 계산 및 결과 저장"""
        if self.cluster_labels is None or self.graphmae_embeddings is None:
            raise RuntimeError("클러스터링이 완료되지 않았습니다.")

        embeddings_np = self.graphmae_embeddings.numpy()
        unique_clusters, counts = np.unique(self.cluster_labels, return_counts=True)
        results = {
            'config': self.config.__dict__,
            'graph_stats': self.word_graph.get_graph_stats(),
            'num_clusters': int(len(unique_clusters)),
            'cluster_distribution': {int(k): int(v) for k, v in zip(unique_clusters, counts)},
            'metrics': {},
            'clusters': self._build_cluster_info()
        }

        # 평가 지표 계산
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        if 'silhouette' in self.config.eval_metrics:
            results['metrics']['silhouette'] = float(silhouette_score(embeddings_np, self.cluster_labels))
            self._log(f"  Silhouette Score: {results['metrics']['silhouette']:.4f}")

        if 'davies_bouldin' in self.config.eval_metrics:
            results['metrics']['davies_bouldin'] = float(davies_bouldin_score(embeddings_np, self.cluster_labels))
            self._log(f"  Davies-Bouldin Score: {results['metrics']['davies_bouldin']:.4f}")

        if 'calinski_harabasz' in self.config.eval_metrics:
            results['metrics']['calinski_harabasz'] = float(calinski_harabasz_score(embeddings_np, self.cluster_labels))
            self._log(f"  Calinski-Harabasz Score: {results['metrics']['calinski_harabasz']:.4f}")

        # TODO: NPMI 계산 (공출현 정보 필요)
        if 'npmi' in self.config.eval_metrics:
            # results['metrics']['npmi'] = self._calculate_npmi()
            pass

        # 결과 저장
        if self.config.save_results:
            self._save_results(results)

        self.cluster_results = results
        return results

    # ============================================================
    # 유틸리티 메서드
    # ============================================================

    def _build_cluster_info(self) -> Dict[int, List[str]]:
        """클러스터별 단어 정보 구성"""
        cluster_info = {}
        for cluster_id in np.unique(self.cluster_labels):
            indices = np.where(self.cluster_labels == cluster_id)[0]
            words = [self.word_graph.words[i].content for i in indices]
            cluster_info[int(cluster_id)] = words
        return cluster_info

    def _save_results(self, results: Dict[str, Any]) -> None:
        """결과를 디스크에 저장"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 결과 저장
        json_path = output_dir / f"grace_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self._log(f"  결과 저장: {json_path}")

        # 임베딩 저장
        if self.config.save_embeddings:
            embed_path = output_dir / f"embeddings_{timestamp}.pt"
            torch.save({
                'graphmae_embeddings': self.graphmae_embeddings,
                'node_features': self.node_features,
                'cluster_labels': self.cluster_labels,
                'word_list': [w.content for w in self.word_graph.words]
            }, embed_path)
            self._log(f"  임베딩 저장: {embed_path}")

    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """
        Elbow Method로 최적 클러스터 수 찾기

        각도 변화율을 계산하여 가장 급격하게 꺾이는 지점 탐색
        """
        if len(k_values) < 3:
            return k_values[0]

        # 정규화된 inertia 계산
        inertias_norm = np.array(inertias)
        inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min())

        # 2차 미분 (곡률) 계산
        second_derivative = np.diff(inertias_norm, 2)

        # 곡률이 최대인 지점 = elbow point
        elbow_idx = np.argmax(second_derivative) + 1  # diff로 인한 인덱스 조정

        return k_values[elbow_idx]

    def _save_elbow_curve(self, k_values: List[int], inertias: List[float],
                          silhouette_scores: List[float], best_k: int) -> None:
        """Elbow curve 시각화 저장"""
        import matplotlib.pyplot as plt

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Inertia plot (Elbow curve)
        ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Elbow Point (k={best_k})')
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Silhouette score plot
        ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Selected k={best_k}')
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = output_dir / f"elbow_curve_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self._log(f"  Elbow curve 저장: {save_path}")

    def _log(self, message: str) -> None:
        """로그 출력"""
        if self.config.verbose:
            print(message)

    # ============================================================
    # 외부 접근 메서드
    # ============================================================

    def get_cluster_words(self, cluster_id: int) -> List[str]:
        """특정 클러스터의 단어들 반환"""
        if self.cluster_labels is None:
            raise RuntimeError("클러스터링이 완료되지 않았습니다.")

        indices = np.where(self.cluster_labels == cluster_id)[0]
        return [self.word_graph.words[i].content for i in indices]

    def get_word_cluster(self, word: str) -> Optional[int]:
        """특정 단어가 속한 클러스터 ID 반환"""
        if self.cluster_labels is None or self.word_graph is None:
            raise RuntimeError("클러스터링이 완료되지 않았습니다.")

        node_id = self.word_graph.get_node_id_by_word(word)
        if node_id is None:
            return None

        return int(self.cluster_labels[node_id])

    def export_cluster_csv(self, output_path: str) -> None:
        """클러스터 결과를 CSV로 저장"""
        if self.cluster_labels is None or self.word_graph is None:
            raise RuntimeError("클러스터링이 완료되지 않았습니다.")

        data = []
        for i, word in enumerate(self.word_graph.words):
            data.append({
                'word': word.content,
                'cluster': int(self.cluster_labels[i]),
                'frequency': word.freq,
                'pos': word.dominant_pos
            })

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        self._log(f"  클러스터 CSV 저장: {output_path}")
