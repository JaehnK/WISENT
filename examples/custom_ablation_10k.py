"""
Custom Ablation Study - 10k Documents

설정:
- 문서 수: 10,000
- GraphMAE Epochs: 100
- Word2Vec Epochs: 10
- Mask Rates: [0.3, 0.5, 0.75]
- 나머지: 기본값 고정
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'core'))

from core.services.Experiment.AblationService import AblationService
from core.services.GRACE.GRACEConfig import GRACEConfig


def main():
    """10k documents, 100 epochs ablation study"""

    print("\n" + "=" * 80)
    print("🔬 Custom Ablation Study - 10k Documents")
    print("=" * 80)
    print("\n📋 실험 설정:")
    print("-" * 80)
    print(f"  {'문서 수 (num_documents)':<35}: 10,000")
    print(f"  {'상위 단어 수 (top_n_words)':<35}: 1,000")
    print(f"  {'최소 문장 길이 (max_sentence_length)':<35}: -1 (제한 없음)")
    print(f"  {'Stopwords 제거 (exclude_stopwords)':<35}: True")
    print()
    print(f"  {'임베딩 방법 (embedding_method)':<35}: 'concat' (Word2Vec + BERT)")
    print(f"  {'총 임베딩 차원 (embed_size)':<35}: 64")
    print(f"  {'  - Word2Vec 차원 (w2v_dim)':<35}: 32")
    print(f"  {'  - BERT 차원 (bert_dim)':<35}: 32")
    print()
    print(f"  {'GraphMAE Epochs (graphmae_epochs)':<35}: 100")
    print(f"  {'GraphMAE Learning Rate (graphmae_lr)':<35}: 0.001")
    print(f"  {'GraphMAE Weight Decay':<35}: 0.0")
    print(f"  {'Mask Rates (실험 대상)':<35}: [0.3, 0.5, 0.75]")
    print()
    print(f"  {'클러스터링 방법 (clustering_method)':<35}: 'kmeans'")
    print(f"  {'클러스터 수 (num_clusters)':<35}: None (Elbow Method 자동)")
    print(f"  {'최소 클러스터 수 (min_clusters)':<35}: 5")
    print(f"  {'최대 클러스터 수 (max_clusters)':<35}: 20")
    print()
    print(f"  {'평가 지표 (eval_metrics)':<35}: ['silhouette', 'davies_bouldin',")
    print(f"  {'':<35}  'calinski_harabasz', 'npmi']")
    print()
    print(f"  {'랜덤 시드 (random_state)':<35}: 42")
    print(f"  {'Device':<35}: cuda (if available) else cpu")
    print("-" * 80)

    # 설정 생성
    config = GRACEConfig(
        # === 데이터 로딩 ===
        csv_path='/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv',
        num_documents=10000,
        text_column='body',

        # === 그래프 구축 ===
        top_n_words=1000,
        exclude_stopwords=True,
        max_sentence_length=-1,  # 제한 없음

        # === 임베딩 설정 ===
        embedding_method='concat',  # Word2Vec + BERT
        embed_size=64,
        w2v_dim=32,
        bert_dim=32,

        # === GraphMAE 설정 ===
        graphmae_epochs=100,
        graphmae_lr=0.001,
        graphmae_weight_decay=0.0,
        graphmae_device=None,  # 자동 선택 (CUDA 우선)
        mask_rate=0.75,  # 기본값 (ablation에서 변경됨)

        # === 클러스터링 설정 ===
        clustering_method='kmeans',
        num_clusters=None,  # Elbow Method 자동 탐색
        min_clusters=5,
        max_clusters=20,

        # === 평가 설정 ===
        eval_metrics=['silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi'],

        # === 출력 설정 ===
        save_results=False,
        output_dir='./ablation_output',
        save_graph_viz=False,
        save_embeddings=False,

        # === 디버그 설정 ===
        verbose=True,
        log_interval=10
    )

    # Word2Vec은 별도 설정이 없으므로 기본값 사용
    # 기본값: window=5, min_count=1, workers=4, sg=1 (Skip-gram)
    # epochs는 Word2VecService 내부에서 기본 5로 설정되어 있음
    # 만약 10 epochs를 명시적으로 설정하려면 Word2VecService 코드 수정 필요

    # print(f"\n⚠️  참고: Word2Vec Epochs")
    # print(f"    현재 Word2VecService는 epochs 파라미터를 GRACEConfig에서 받지 않습니다.")
    # print(f"    Word2Vec은 기본 5 epochs로 학습됩니다.")
    # print(f"    10 epochs로 변경하려면 core/services/Word2Vec/Word2VecService.py 수정 필요")

    print(f"\n🚀 Ablation Study 시작...")
    print("=" * 80)

    # AblationService 초기화
    ablation_service = AblationService(base_config=config, random_state=42)

    # 공유 데이터 준비
    print("\n📁 공유 데이터 준비 중...")
    ablation_service.prepare_shared_data()

    # Mask Rate Ablation 실행
    print("\n" + "=" * 80)
    print("🔬 Mask Rate Ablation Study")
    print("=" * 80)
    print(f"  테스트할 Mask Rates: [0.3, 0.5, 0.75]")
    print(f"  각 mask rate마다 100 epochs 학습")
    print("=" * 80)

    mask_results = ablation_service.run_mask_rate_ablation(
        mask_rates=[0.3, 0.5, 0.75]
    )

    # 결과 요약 출력
    print("\n" + "=" * 80)
    print("📊 결과 요약")
    print("=" * 80)

    best_mask = None
    best_score = -1

    print(f"\n{'Mask Rate':<12} {'Silhouette':<12} {'Davies-Bouldin':<18} {'Calinski-Harabasz':<20}")
    print("-" * 70)

    for mask_rate, metrics in sorted(mask_results.items()):
        sil = metrics['silhouette']
        db = metrics['davies_bouldin']
        ch = metrics['calinski_harabasz']

        print(f"{mask_rate:<12.2f} {sil:<12.4f} {db:<18.4f} {ch:<20.2f}")

        if sil > best_score:
            best_score = sil
            best_mask = mask_rate

    print("-" * 70)
    print(f"\n⭐ Best Mask Rate: {best_mask:.2f} (Silhouette = {best_score:.4f})")

    # 결과 저장
    print("\n💾 결과 저장 중...")
    ablation_service.save_results(output_dir="./ablation_output")

    print("\n" + "=" * 80)
    print("✅ Custom Ablation Study 완료!")
    print("=" * 80)
    print(f"\n결과 파일: ./ablation_output/ablation_results_*.json")
    print(f"로드된 문서: {config.num_documents:,}개")
    print(f"GraphMAE 학습: {config.graphmae_epochs} epochs")
    print(f"테스트한 Mask Rates: [0.3, 0.5, 0.75]")
    print(f"Best Mask Rate: {best_mask}")
    print()


if __name__ == '__main__':
    main()
