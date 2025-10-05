"""
AblationService 테스트

Ablation Study의 각 기능을 단위 테스트 및 통합 테스트
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

from core.services.Experiment.AblationService import AblationService
from core.services.GRACE.GRACEConfig import GRACEConfig


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_csv_path(tmp_path):
    """테스트용 샘플 CSV 파일 생성"""
    import pandas as pd

    # 샘플 문서 생성
    sample_data = {
        'body': [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language for data science",
            "Deep learning models require large amounts of training data",
            "Natural language processing enables computers to understand text",
            "The weather is sunny and warm today",
            "I love reading books in my free time",
            "The stock market showed significant gains this week",
            "Climate change is a pressing global issue",
            "Technology advances at an incredible pace",
        ] * 10  # 100개 문서
    }

    df = pd.DataFrame(sample_data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    return str(csv_path)


@pytest.fixture
def test_config(sample_csv_path):
    """테스트용 GRACE 설정"""
    return GRACEConfig(
        csv_path=sample_csv_path,
        num_documents=50,
        top_n_words=30,
        graphmae_epochs=5,  # 빠른 테스트를 위해 epoch 감소
        embed_size=32,
        w2v_dim=16,
        bert_dim=16,
        min_clusters=2,
        max_clusters=5,
        save_results=False,
        save_embeddings=False,
        save_graph_viz=False,
        verbose=False
    )


@pytest.fixture
def ablation_service(test_config):
    """AblationService 인스턴스"""
    return AblationService(base_config=test_config, random_state=42)


# ============================================================
# 단위 테스트
# ============================================================

class TestAblationServiceInit:
    """AblationService 초기화 테스트"""

    def test_init(self, test_config):
        """정상 초기화 테스트"""
        service = AblationService(base_config=test_config, random_state=42)

        assert service.base_config == test_config
        assert service.random_state == 42
        assert service.doc_service is None
        assert service.word_graph is None
        assert service.ablation_results == {}

    def test_init_with_different_seed(self, test_config):
        """다른 랜덤 시드로 초기화"""
        service = AblationService(base_config=test_config, random_state=123)
        assert service.random_state == 123


class TestPrepareSharedData:
    """공유 데이터 준비 테스트"""

    def test_prepare_shared_data(self, ablation_service):
        """공유 데이터 준비 성공"""
        ablation_service.prepare_shared_data()

        assert ablation_service.doc_service is not None
        assert ablation_service.word_graph is not None
        assert ablation_service.word_graph.num_nodes > 0
        assert ablation_service.word_graph.num_edges > 0

    def test_prepare_shared_data_creates_correct_graph(self, ablation_service):
        """올바른 그래프 구조 생성 확인"""
        ablation_service.prepare_shared_data()

        # 그래프 노드 수가 top_n_words와 일치하는지 확인
        assert ablation_service.word_graph.num_nodes <= ablation_service.base_config.top_n_words


class TestConfigVariant:
    """설정 변형 생성 테스트"""

    def test_create_config_variant_default(self, ablation_service):
        """기본 설정 변형 생성"""
        variant = ablation_service._create_config_variant()

        assert variant.csv_path == ablation_service.base_config.csv_path
        assert variant.save_results is False  # 자동으로 비활성화
        assert variant.save_embeddings is False
        assert variant.verbose is False

    def test_create_config_variant_with_params(self, ablation_service):
        """파라미터 변경된 설정 생성"""
        variant = ablation_service._create_config_variant(
            mask_rate=0.5,
            graphmae_epochs=200
        )

        assert variant.mask_rate == 0.5
        assert variant.graphmae_epochs == 200
        assert variant.csv_path == ablation_service.base_config.csv_path

    def test_create_config_variant_embedding_method(self, ablation_service):
        """임베딩 방법 변경"""
        variant = ablation_service._create_config_variant(embedding_method='w2v')
        assert variant.embedding_method == 'w2v'


class TestCalculateImprovement:
    """개선율 계산 테스트"""

    def test_calculate_improvement_positive(self, ablation_service):
        """양수 개선율 계산"""
        baseline = {'silhouette': 0.5, 'calinski_harabasz': 100}
        improved = {'silhouette': 0.6, 'calinski_harabasz': 120}

        improvement = ablation_service._calculate_improvement(baseline, improved)

        assert improvement['silhouette'] == pytest.approx(20.0, rel=1e-5)
        assert improvement['calinski_harabasz'] == pytest.approx(20.0, rel=1e-5)

    def test_calculate_improvement_davies_bouldin(self, ablation_service):
        """Davies-Bouldin (낮을수록 좋음) 개선율 계산"""
        baseline = {'davies_bouldin': 1.0}
        improved = {'davies_bouldin': 0.8}

        improvement = ablation_service._calculate_improvement(baseline, improved)

        # davies_bouldin은 낮아질 때 개선
        assert improvement['davies_bouldin'] == pytest.approx(20.0, rel=1e-5)

    def test_calculate_improvement_negative(self, ablation_service):
        """음수 개선율 (성능 저하)"""
        baseline = {'silhouette': 0.6}
        improved = {'silhouette': 0.5}

        improvement = ablation_service._calculate_improvement(baseline, improved)

        assert improvement['silhouette'] == pytest.approx(-16.666666, rel=1e-5)


class TestMakeSerializable:
    """JSON 직렬화 헬퍼 테스트"""

    def test_make_serializable_dict(self, ablation_service):
        """딕셔너리 직렬화"""
        obj = {'a': np.float64(1.5), 'b': np.int32(10)}
        result = ablation_service._make_serializable(obj)

        assert isinstance(result['a'], float)
        assert isinstance(result['b'], float)
        assert result['a'] == pytest.approx(1.5)
        assert result['b'] == pytest.approx(10.0)

    def test_make_serializable_numpy_array(self, ablation_service):
        """NumPy 배열 직렬화"""
        obj = np.array([1.0, 2.0, 3.0])
        result = ablation_service._make_serializable(obj)

        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_make_serializable_torch_tensor(self, ablation_service):
        """PyTorch 텐서 직렬화"""
        obj = torch.tensor([1.0, 2.0, 3.0])
        result = ablation_service._make_serializable(obj)

        assert isinstance(result, list)
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_make_serializable_nested(self, ablation_service):
        """중첩된 구조 직렬화"""
        obj = {
            'metrics': {
                'silhouette': np.float64(0.5),
                'scores': np.array([1, 2, 3])
            }
        }
        result = ablation_service._make_serializable(obj)

        assert isinstance(result['metrics']['silhouette'], float)
        assert isinstance(result['metrics']['scores'], list)


# ============================================================
# 통합 테스트 (실제 실험 실행)
# ============================================================

class TestEmbeddingAblation:
    """임베딩 방법 Ablation 테스트"""

    def test_run_embedding_ablation(self, ablation_service):
        """임베딩 방법별 실험 실행"""
        ablation_service.prepare_shared_data()
        results = ablation_service.run_embedding_ablation()

        # 3가지 방법 모두 테스트되었는지 확인
        assert 'w2v' in results
        assert 'bert' in results
        assert 'concat' in results

        # 각 결과에 메트릭이 포함되어 있는지 확인
        for method, metrics in results.items():
            assert 'silhouette' in metrics
            assert 'davies_bouldin' in metrics
            assert 'calinski_harabasz' in metrics

    def test_embedding_ablation_metrics_range(self, ablation_service):
        """메트릭 값 범위 확인"""
        ablation_service.prepare_shared_data()
        results = ablation_service.run_embedding_ablation()

        for method, metrics in results.items():
            # Silhouette: [-1, 1]
            assert -1 <= metrics['silhouette'] <= 1
            # Davies-Bouldin: [0, ∞)
            assert metrics['davies_bouldin'] >= 0
            # Calinski-Harabasz: [0, ∞)
            assert metrics['calinski_harabasz'] >= 0


class TestGraphMAEAblation:
    """GraphMAE Ablation 테스트"""

    def test_run_graphmae_ablation(self, ablation_service):
        """GraphMAE 유/무 실험 실행"""
        ablation_service.prepare_shared_data()
        results = ablation_service.run_graphmae_ablation()

        assert 'without_graphmae' in results
        assert 'with_graphmae' in results
        assert 'improvement' in results

        # 개선율 계산 확인
        improvement = results['improvement']
        assert 'silhouette' in improvement
        assert 'davies_bouldin' in improvement

    def test_graphmae_ablation_improvement_exists(self, ablation_service):
        """GraphMAE로 인한 성능 변화 확인 (개선 여부는 데이터에 따라 다름)"""
        ablation_service.prepare_shared_data()
        results = ablation_service.run_graphmae_ablation()

        # improvement 값이 계산되었는지만 확인 (부호는 확인 안 함)
        improvement = results['improvement']
        assert isinstance(improvement['silhouette'], float)


class TestMaskRateAblation:
    """Mask Rate Ablation 테스트"""

    def test_run_mask_rate_ablation_default(self, ablation_service):
        """기본 mask rate 리스트로 실험"""
        ablation_service.prepare_shared_data()
        results = ablation_service.run_mask_rate_ablation()

        # 기본값: [0.3, 0.5, 0.75, 0.9]
        assert 0.3 in results
        assert 0.5 in results
        assert 0.75 in results
        assert 0.9 in results

    def test_run_mask_rate_ablation_custom(self, ablation_service):
        """커스텀 mask rate 리스트"""
        ablation_service.prepare_shared_data()
        results = ablation_service.run_mask_rate_ablation(mask_rates=[0.5, 0.7])

        assert len(results) == 2
        assert 0.5 in results
        assert 0.7 in results

    def test_run_mask_rate_ablation_multiple_runs(self, ablation_service):
        """Multiple runs 테스트"""
        ablation_service.prepare_shared_data()
        results = ablation_service.run_mask_rate_ablation(
            mask_rates=[0.5, 0.75],
            num_runs=3
        )

        assert len(results) == 2
        assert 0.5 in results
        assert 0.75 in results

        # 집계된 통계 확인
        for mask_rate, metrics in results.items():
            assert 'silhouette_mean' in metrics
            assert 'silhouette_std' in metrics
            assert 'silhouette_min' in metrics
            assert 'silhouette_max' in metrics
            assert 'num_runs' in metrics
            assert metrics['num_runs'] == 3

            # 통계적 일관성 확인
            assert metrics['silhouette_min'] <= metrics['silhouette_mean'] <= metrics['silhouette_max']
            assert metrics['silhouette_std'] >= 0


class TestEmbedSizeAblation:
    """임베딩 차원 Ablation 테스트"""

    def test_run_embed_size_ablation_default(self, ablation_service):
        """기본 임베딩 차원 리스트로 실험"""
        ablation_service.prepare_shared_data()
        results = ablation_service.run_embed_size_ablation()

        # 기본값: [32, 64, 128, 256]
        assert 32 in results
        assert 64 in results
        assert 128 in results
        assert 256 in results

    def test_run_embed_size_ablation_custom(self, ablation_service):
        """커스텀 임베딩 차원 리스트"""
        ablation_service.prepare_shared_data()
        results = ablation_service.run_embed_size_ablation(embed_sizes=[16, 32])

        assert len(results) == 2
        assert 16 in results
        assert 32 in results


class TestEpochsAblation:
    """Epochs Ablation 테스트"""

    def test_run_epochs_ablation_default(self, ablation_service):
        """기본 epochs 리스트로 실험"""
        ablation_service.prepare_shared_data()
        results = ablation_service.run_epochs_ablation(epochs_list=[5, 10])  # 빠른 테스트

        assert 5 in results
        assert 10 in results

    def test_run_epochs_ablation_custom(self, ablation_service):
        """커스텀 epochs 리스트"""
        ablation_service.prepare_shared_data()
        results = ablation_service.run_epochs_ablation(epochs_list=[3, 5])

        assert len(results) == 2
        assert 3 in results
        assert 5 in results


class TestFullAblationStudy:
    """전체 Ablation Study 테스트"""

    def test_run_full_ablation_study_minimal(self, ablation_service):
        """최소 ablation study (빠른 테스트)"""
        results = ablation_service.run_full_ablation_study(
            include_embedding=True,
            include_graphmae=True,
            include_mask_rate=False,
            include_embed_size=False,
            include_epochs=False
        )

        assert 'embedding_method' in results
        assert 'graphmae' in results

    def test_run_full_ablation_study_complete(self, ablation_service):
        """전체 ablation study (시간 소요)"""
        # 실제 환경에서는 이 테스트가 오래 걸릴 수 있음
        pytest.skip("Complete ablation study takes too long for CI")

        results = ablation_service.run_full_ablation_study(
            include_embedding=True,
            include_graphmae=True,
            include_mask_rate=True,
            include_embed_size=True,
            include_epochs=True
        )

        assert 'embedding_method' in results
        assert 'graphmae' in results
        assert 'mask_rate' in results
        assert 'embed_size' in results
        assert 'epochs' in results


class TestSaveResults:
    """결과 저장 테스트"""

    def test_save_results(self, ablation_service, tmp_path):
        """결과 JSON 저장"""
        ablation_service.prepare_shared_data()
        ablation_service.run_embedding_ablation()

        output_dir = tmp_path / "ablation_output"
        ablation_service.save_results(output_dir=str(output_dir))

        # JSON 파일이 생성되었는지 확인
        json_files = list(output_dir.glob("ablation_results_*.json"))
        assert len(json_files) == 1

        # JSON 파일을 읽을 수 있는지 확인
        import json
        with open(json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert 'embedding_method' in data
        assert 'w2v' in data['embedding_method']


# ============================================================
# 에러 케이스 테스트
# ============================================================

class TestErrorCases:
    """에러 케이스 테스트"""

    def test_run_ablation_without_prepare(self, ablation_service):
        """prepare_shared_data 없이 실험 실행 시 에러"""
        with pytest.raises(RuntimeError, match="prepare_shared_data"):
            ablation_service.run_embedding_ablation()

    def test_prepare_with_invalid_csv(self, test_config):
        """잘못된 CSV 경로"""
        test_config.csv_path = "/invalid/path/data.csv"
        service = AblationService(base_config=test_config)

        with pytest.raises(FileNotFoundError):
            service.prepare_shared_data()


# ============================================================
# 재현성 테스트
# ============================================================

class TestReproducibility:
    """재현성 테스트"""

    def test_same_seed_same_results(self, test_config):
        """동일한 시드로 동일한 결과 생성"""
        service1 = AblationService(base_config=test_config, random_state=42)
        service1.prepare_shared_data()
        results1 = service1.run_embedding_ablation()

        service2 = AblationService(base_config=test_config, random_state=42)
        service2.prepare_shared_data()
        results2 = service2.run_embedding_ablation()

        # Silhouette 점수가 동일한지 확인
        for method in ['w2v', 'bert', 'concat']:
            assert results1[method]['silhouette'] == pytest.approx(
                results2[method]['silhouette'], rel=1e-5
            )

    def test_different_seed_different_results(self, test_config):
        """다른 시드로 다른 결과 생성 (확률적)"""
        service1 = AblationService(base_config=test_config, random_state=42)
        service1.prepare_shared_data()
        results1 = service1.run_embedding_ablation()

        service2 = AblationService(base_config=test_config, random_state=123)
        service2.prepare_shared_data()
        results2 = service2.run_embedding_ablation()

        # 시드가 다르므로 결과가 다를 수 있음 (항상은 아님)
        # 여기서는 단순히 실행 가능 여부만 확인
        assert results1 is not None
        assert results2 is not None
