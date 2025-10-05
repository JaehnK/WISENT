"""
GRACE Ablation Study 실행 스크립트

사용법:
    python examples/run_ablation_study.py

옵션:
    --full: 전체 ablation study 실행 (시간 오래 걸림)
    --quick: 빠른 테스트용 (적은 데이터, 적은 epoch)
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'core'))

import argparse
from core.services.Experiment.AblationService import AblationService
from core.services.GRACE.GRACEConfig import GRACEConfig


def run_quick_ablation(csv_path: str):
    """
    빠른 ablation study (테스트용)
    - 적은 문서 수
    - 짧은 학습
    """
    print("\n" + "=" * 80)
    print("🚀 Quick Ablation Study (테스트용)")
    print("=" * 80)

    config = GRACEConfig(
        csv_path=csv_path,
        num_documents=100,         # 적은 문서
        top_n_words=50,            # 적은 단어
        graphmae_epochs=10,        # 짧은 학습
        embed_size=32,
        w2v_dim=16,
        bert_dim=16,
        min_clusters=3,
        max_clusters=8,
        save_results=False,
        save_embeddings=False,
        save_graph_viz=False,
        verbose=True
    )

    ablation_service = AblationService(base_config=config, random_state=42)

    # 빠른 ablation: 임베딩 방법과 GraphMAE만
    results = ablation_service.run_full_ablation_study(
        include_embedding=True,
        include_graphmae=True,
        include_mask_rate=False,
        include_embed_size=False,
        include_epochs=False
    )

    # 결과 저장
    ablation_service.save_results(output_dir="./ablation_output")

    print("\n" + "=" * 80)
    print("✅ Quick Ablation Study 완료!")
    print("=" * 80)

    return results


def run_standard_ablation(csv_path: str):
    """
    표준 ablation study
    - 중간 크기 데이터
    - 임베딩 방법, GraphMAE, Mask Rate 실험
    """
    print("\n" + "=" * 80)
    print("🚀 Standard Ablation Study")
    print("=" * 80)

    config = GRACEConfig(
        csv_path=csv_path,
        num_documents=1000,
        top_n_words=500,
        graphmae_epochs=100,
        embed_size=64,
        w2v_dim=32,
        bert_dim=32,
        min_clusters=5,
        max_clusters=15,
        save_results=False,
        save_embeddings=False,
        save_graph_viz=False,
        verbose=True
    )

    ablation_service = AblationService(base_config=config, random_state=42)

    # 표준 ablation: 임베딩, GraphMAE, Mask Rate
    results = ablation_service.run_full_ablation_study(
        include_embedding=True,
        include_graphmae=True,
        include_mask_rate=True,
        include_embed_size=False,  # 시간 많이 걸림
        include_epochs=False       # 시간 많이 걸림
    )

    # 결과 저장
    ablation_service.save_results(output_dir="./ablation_output")

    print("\n" + "=" * 80)
    print("✅ Standard Ablation Study 완료!")
    print("=" * 80)

    return results


def run_full_ablation(csv_path: str):
    """
    전체 ablation study (논문용)
    - 모든 ablation 실험 수행
    - 시간 오래 걸림 (수 시간)
    """
    print("\n" + "=" * 80)
    print("🚀 Full Ablation Study (논문용)")
    print("=" * 80)
    print("⚠️  경고: 이 작업은 수 시간이 걸릴 수 있습니다!")
    print("=" * 80)

    config = GRACEConfig(
        csv_path=csv_path,
        num_documents=5000,        # 많은 문서
        top_n_words=1000,          # 많은 단어
        graphmae_epochs=200,       # 긴 학습
        embed_size=128,
        w2v_dim=64,
        bert_dim=64,
        min_clusters=5,
        max_clusters=20,
        save_results=False,
        save_embeddings=False,
        save_graph_viz=False,
        verbose=True
    )

    ablation_service = AblationService(base_config=config, random_state=42)

    # 전체 ablation: 모든 실험
    results = ablation_service.run_full_ablation_study(
        include_embedding=True,
        include_graphmae=True,
        include_mask_rate=True,
        include_embed_size=True,
        include_epochs=True
    )

    # 결과 저장
    ablation_service.save_results(output_dir="./ablation_output")

    print("\n" + "=" * 80)
    print("✅ Full Ablation Study 완료!")
    print("=" * 80)

    return results


def run_custom_ablation(csv_path: str):
    """
    커스텀 ablation study
    - 사용자가 원하는 실험만 선택
    """
    print("\n" + "=" * 80)
    print("🔧 Custom Ablation Study")
    print("=" * 80)

    config = GRACEConfig(
        csv_path=csv_path,
        num_documents=1000,
        top_n_words=500,
        graphmae_epochs=100,
        embed_size=64,
        w2v_dim=32,
        bert_dim=32,
        verbose=True
    )

    ablation_service = AblationService(base_config=config, random_state=42)

    # 공유 데이터 준비
    ablation_service.prepare_shared_data()

    # 개별 ablation 실험 선택 실행
    print("\n1️⃣  Embedding Method Ablation")
    embedding_results = ablation_service.run_embedding_ablation()

    print("\n2️⃣  GraphMAE Effect Ablation")
    graphmae_results = ablation_service.run_graphmae_ablation()

    print("\n3️⃣  Mask Rate Ablation")
    mask_results = ablation_service.run_mask_rate_ablation(
        mask_rates=[0.5, 0.75, 0.9]  # 커스텀 mask rate
    )

    # 결과 저장
    ablation_service.save_results(output_dir="./ablation_output")

    print("\n" + "=" * 80)
    print("✅ Custom Ablation Study 완료!")
    print("=" * 80)

    return {
        'embedding': embedding_results,
        'graphmae': graphmae_results,
        'mask_rate': mask_results
    }


def print_summary(results: dict):
    """결과 요약 출력"""
    print("\n" + "=" * 80)
    print("📊 결과 요약")
    print("=" * 80)

    # Embedding Method 결과
    if 'embedding_method' in results:
        print("\n[Embedding Method Ablation]")
        for method, metrics in results['embedding_method'].items():
            print(f"  {method:10s}: Silhouette={metrics['silhouette']:.4f}, "
                  f"Davies-Bouldin={metrics['davies_bouldin']:.4f}")

    # GraphMAE 결과
    if 'graphmae' in results:
        print("\n[GraphMAE Effect]")
        gm = results['graphmae']
        if 'without_graphmae' in gm and 'with_graphmae' in gm:
            without = gm['without_graphmae']
            with_gm = gm['with_graphmae']
            improvement = gm.get('improvement', {})

            print(f"  Without: Silhouette={without['silhouette']:.4f}")
            print(f"  With:    Silhouette={with_gm['silhouette']:.4f}")
            if 'silhouette' in improvement:
                print(f"  Improvement: {improvement['silhouette']:+.2f}%")

    # Mask Rate 결과
    if 'mask_rate' in results:
        print("\n[Mask Rate Ablation]")
        best_mask = None
        best_score = -1
        for mask, metrics in results['mask_rate'].items():
            score = metrics['silhouette']
            print(f"  {mask:.2f}: Silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_mask = mask
        print(f"  ⭐ Best: mask_rate={best_mask:.2f}")

    # Embed Size 결과
    if 'embed_size' in results:
        print("\n[Embedding Dimension Ablation]")
        for size, metrics in results['embed_size'].items():
            print(f"  {size:4d}: Silhouette={metrics['silhouette']:.4f}")

    # Epochs 결과
    if 'epochs' in results:
        print("\n[Training Epochs Ablation]")
        for epochs, metrics in results['epochs'].items():
            print(f"  {epochs:4d}: Silhouette={metrics['silhouette']:.4f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='GRACE Ablation Study 실행',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 빠른 테스트 (기본)
  python examples/run_ablation_study.py --quick

  # 표준 실험
  python examples/run_ablation_study.py --standard

  # 전체 실험 (논문용)
  python examples/run_ablation_study.py --full

  # 커스텀 실험
  python examples/run_ablation_study.py --custom

  # CSV 경로 지정
  python examples/run_ablation_study.py --quick --csv /path/to/data.csv
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='빠른 테스트용 (적은 데이터, 짧은 학습)'
    )
    parser.add_argument(
        '--standard',
        action='store_true',
        help='표준 ablation study (중간 크기)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='전체 ablation study (논문용, 시간 오래 걸림)'
    )
    parser.add_argument(
        '--custom',
        action='store_true',
        help='커스텀 ablation study'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv',
        help='CSV 파일 경로 (기본: kaggle_RC_2019-05.csv)'
    )

    args = parser.parse_args()

    # CSV 파일 존재 확인
    if not os.path.exists(args.csv):
        print(f"❌ 오류: CSV 파일을 찾을 수 없습니다: {args.csv}")
        sys.exit(1)

    # 실험 모드 선택
    if args.full:
        results = run_full_ablation(args.csv)
    elif args.standard:
        results = run_standard_ablation(args.csv)
    elif args.custom:
        results = run_custom_ablation(args.csv)
    else:  # 기본: quick
        results = run_quick_ablation(args.csv)

    # 결과 요약 출력
    print_summary(results)

    print("\n💾 결과가 ./ablation_output 디렉토리에 저장되었습니다.")


if __name__ == '__main__':
    main()
