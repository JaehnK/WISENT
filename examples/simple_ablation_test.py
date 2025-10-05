"""
간단한 Ablation Test 예제

AblationService의 기본 사용법을 보여주는 간단한 예제
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
    """간단한 ablation test 실행"""

    print("\n" + "=" * 80)
    print("🧪 Simple Ablation Test")
    print("=" * 80)

    # 1. 설정 생성 (테스트용 작은 크기)
    config = GRACEConfig(
        csv_path='/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv',
        num_documents=50,       # 빠른 테스트를 위해 50개만
        top_n_words=30,
        graphmae_epochs=5,      # 5 epoch만
        embed_size=32,
        w2v_dim=16,
        bert_dim=16,
        min_clusters=2,
        max_clusters=5,
        verbose=True
    )

    # 2. AblationService 초기화
    ablation_service = AblationService(base_config=config, random_state=42)

    # 3. 공유 데이터 준비 (한 번만 실행)
    print("\n📁 데이터 준비 중...")
    ablation_service.prepare_shared_data()

    # 4. 개별 ablation 실험 실행

    # 4-1. Embedding Method Ablation
    print("\n" + "=" * 80)
    print("1️⃣  Embedding Method Ablation")
    print("=" * 80)
    embedding_results = ablation_service.run_embedding_ablation()

    print("\n[결과]")
    for method, metrics in embedding_results.items():
        print(f"  {method:10s}: Silhouette={metrics['silhouette']:.4f}")

    # 4-2. GraphMAE Effect
    print("\n" + "=" * 80)
    print("2️⃣  GraphMAE Effect")
    print("=" * 80)
    graphmae_results = ablation_service.run_graphmae_ablation()

    print("\n[결과]")
    print(f"  Without GraphMAE: {graphmae_results['without_graphmae']['silhouette']:.4f}")
    print(f"  With GraphMAE:    {graphmae_results['with_graphmae']['silhouette']:.4f}")
    print(f"  Improvement:      {graphmae_results['improvement']['silhouette']:+.2f}%")

    # 4-3. Mask Rate Ablation (2개만 테스트)
    print("\n" + "=" * 80)
    print("3️⃣  Mask Rate Ablation")
    print("=" * 80)
    mask_results = ablation_service.run_mask_rate_ablation(
        mask_rates=[0.5, 0.75]  # 2개만 빠르게 테스트
    )

    print("\n[결과]")
    for mask, metrics in mask_results.items():
        print(f"  mask={mask:.2f}: Silhouette={metrics['silhouette']:.4f}")

    # 5. 결과 저장
    print("\n💾 결과 저장 중...")
    ablation_service.save_results(output_dir="./ablation_output")

    print("\n" + "=" * 80)
    print("✅ Simple Ablation Test 완료!")
    print("=" * 80)
    print("\n결과 파일: ./ablation_output/ablation_results_*.json")


if __name__ == '__main__':
    main()
