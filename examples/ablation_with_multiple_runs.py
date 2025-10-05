"""
Ablation Study with Multiple Runs (재현성 검증)

각 mask rate를 여러 번 반복 실행하여 평균 ± 표준편차 계산
논문 작성 시 통계적 유의성을 보여주기 위한 실험
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'core'))

from core.services.Experiment.AblationService import AblationService
from core.services.GRACE.GRACEConfig import GRACEConfig
import json


def main():
    """Multiple runs ablation study"""

    print("\n" + "=" * 80)
    print("🔬 Mask Rate Ablation Study with Multiple Runs")
    print("=" * 80)

    # 실험 설정
    NUM_RUNS = 5  # 각 mask rate당 5회 반복
    MASK_RATES = [0.3, 0.5, 0.75]

    print("\n📋 실험 설정:")
    print("-" * 80)
    print(f"  문서 수: 10,000")
    print(f"  GraphMAE Epochs: 100")
    print(f"  Mask Rates: {MASK_RATES}")
    print(f"  반복 횟수 (각 mask rate): {NUM_RUNS}")
    print(f"  총 실험 횟수: {len(MASK_RATES)} × {NUM_RUNS} = {len(MASK_RATES) * NUM_RUNS}")
    print("-" * 80)

    # 설정 생성
    config = GRACEConfig(
        csv_path='/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv',
        num_documents=10000,
        top_n_words=1000,
        graphmae_epochs=100,
        embed_size=64,
        w2v_dim=32,
        bert_dim=32,
        min_clusters=5,
        max_clusters=20,
        save_results=False,
        save_embeddings=False,
        save_graph_viz=False,
        verbose=True
    )

    # AblationService 초기화
    ablation_service = AblationService(base_config=config, random_state=42)

    # 공유 데이터 준비
    print("\n📁 공유 데이터 준비 중...")
    ablation_service.prepare_shared_data()

    # Multiple runs로 Mask Rate Ablation 실행
    print("\n🚀 Mask Rate Ablation 시작 (multiple runs)...")
    mask_results = ablation_service.run_mask_rate_ablation(
        mask_rates=MASK_RATES,
        num_runs=NUM_RUNS
    )

    # 결과 분석 및 출력
    print("\n" + "=" * 80)
    print("📊 최종 결과 요약 (평균 ± 표준편차)")
    print("=" * 80)

    # 테이블 헤더
    print(f"\n{'Mask Rate':<12} {'Silhouette':<25} {'Davies-Bouldin':<25} {'Calinski-Harabasz':<25}")
    print("-" * 90)

    best_mask = None
    best_silhouette = -1

    for mask_rate in sorted(MASK_RATES):
        metrics = mask_results[mask_rate]

        sil_mean = metrics['silhouette_mean']
        sil_std = metrics['silhouette_std']
        db_mean = metrics['davies_bouldin_mean']
        db_std = metrics['davies_bouldin_std']
        ch_mean = metrics['calinski_harabasz_mean']
        ch_std = metrics['calinski_harabasz_std']

        print(f"{mask_rate:<12.2f} "
              f"{sil_mean:.4f} ± {sil_std:.4f}       "
              f"{db_mean:.4f} ± {db_std:.4f}       "
              f"{ch_mean:.2f} ± {ch_std:.2f}")

        if sil_mean > best_silhouette:
            best_silhouette = sil_mean
            best_mask = mask_rate

    print("-" * 90)
    print(f"\n⭐ Best Mask Rate: {best_mask:.2f} (Silhouette = {best_silhouette:.4f})")

    # 통계적 유의성 분석
    print("\n" + "=" * 80)
    print("📈 통계적 분석")
    print("=" * 80)

    print("\n[변동성 분석 (Coefficient of Variation)]")
    for mask_rate in sorted(MASK_RATES):
        metrics = mask_results[mask_rate]
        sil_mean = metrics['silhouette_mean']
        sil_std = metrics['silhouette_std']
        cv = (sil_std / sil_mean) * 100 if sil_mean != 0 else 0

        print(f"  mask={mask_rate:.2f}: CV = {cv:.2f}% ", end="")
        if cv < 5:
            print("(매우 안정적 ✅)")
        elif cv < 10:
            print("(안정적 ✓)")
        else:
            print("(변동 있음 ⚠️)")

    print("\n[신뢰구간 (95% CI, 정규분포 가정)]")
    import numpy as np
    z_score = 1.96  # 95% CI

    for mask_rate in sorted(MASK_RATES):
        metrics = mask_results[mask_rate]
        sil_mean = metrics['silhouette_mean']
        sil_std = metrics['silhouette_std']
        margin = z_score * (sil_std / np.sqrt(NUM_RUNS))
        ci_lower = sil_mean - margin
        ci_upper = sil_mean + margin

        print(f"  mask={mask_rate:.2f}: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # 논문용 LaTeX 표 생성
    print("\n" + "=" * 80)
    print("📝 논문용 LaTeX 표")
    print("=" * 80)

    print("""
\\begin{table}[h]
\\centering
\\caption{Mask Rate Ablation Results (Mean ± Std, n=%d)}
\\label{tab:mask_rate_ablation}
\\begin{tabular}{lccc}
\\toprule
Mask Rate & Silhouette $\\uparrow$ & Davies-Bouldin $\\downarrow$ & Calinski-Harabasz $\\uparrow$ \\\\
\\midrule""" % NUM_RUNS)

    for mask_rate in sorted(MASK_RATES):
        metrics = mask_results[mask_rate]
        sil_mean = metrics['silhouette_mean']
        sil_std = metrics['silhouette_std']
        db_mean = metrics['davies_bouldin_mean']
        db_std = metrics['davies_bouldin_std']
        ch_mean = metrics['calinski_harabasz_mean']
        ch_std = metrics['calinski_harabasz_std']

        # Best 표시
        if mask_rate == best_mask:
            print(f"{mask_rate:.2f} & \\textbf{{{sil_mean:.4f} $\\pm$ {sil_std:.4f}}} & "
                  f"{db_mean:.4f} $\\pm$ {db_std:.4f} & "
                  f"{ch_mean:.2f} $\\pm$ {ch_std:.2f} \\\\")
        else:
            print(f"{mask_rate:.2f} & {sil_mean:.4f} $\\pm$ {sil_std:.4f} & "
                  f"{db_mean:.4f} $\\pm$ {db_std:.4f} & "
                  f"{ch_mean:.2f} $\\pm$ {ch_std:.2f} \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    # 결과 저장
    print("\n💾 결과 저장 중...")
    ablation_service.save_results(output_dir="./ablation_output")

    # 추가로 요약 통계 저장
    summary = {
        'experiment_config': {
            'num_documents': 10000,
            'graphmae_epochs': 100,
            'mask_rates': MASK_RATES,
            'num_runs': NUM_RUNS
        },
        'results': mask_results,
        'best_mask_rate': best_mask,
        'best_silhouette': best_silhouette
    }

    summary_path = Path("./ablation_output") / "summary_statistics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  요약 통계: {summary_path}")

    print("\n" + "=" * 80)
    print("✅ Multiple Runs Ablation Study 완료!")
    print("=" * 80)
    print(f"\n결과:")
    print(f"  - 전체 결과: ./ablation_output/ablation_results_*.json")
    print(f"  - 요약 통계: ./ablation_output/summary_statistics.json")
    print(f"  - Best Mask Rate: {best_mask}")
    print(f"  - Runs per setting: {NUM_RUNS}")
    print()


if __name__ == '__main__':
    main()
