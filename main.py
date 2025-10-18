#!/usr/bin/env python3
"""
GRACE Main Script with Edge Weight Integration

GraphMAE2 + GCN + Edge Weight를 사용한 전체 파이프라인 실행

Usage:
    python main.py --mode train --config config.json
    python main.py --mode evaluate --model model.pkl
    python main.py --mode ablation --output results/
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')


# PYTHONPATH 설정
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "core"))

from colorama import init, Fore, Style
from core.services.GRACE import GRACEPipeline, GRACEConfig

# Colorama 초기화
init(autoreset=True)


def print_banner():
    """GRACE 배너 출력"""
    banner = f"""
{Fore.CYAN}{'=' * 80}
{Fore.GREEN}  GRACE: GRAph-based Clustering with Enhanced embeddings
{Fore.YELLOW}  Version: 2.0 (Edge Weight Integration)
{Fore.MAGENTA}  GraphMAE2 + GCN + Edge Weight (Min-Max Normalized)
{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}
"""
    print(banner)


def create_default_config() -> GRACEConfig:
    """기본 설정 생성"""

    config = GRACEConfig(
        # 데이터
        csv_path="data/reddit_mental_health_cleaned.csv",
        num_documents=10000,
        text_column='body',

        # 그래프
        top_n_words=500,
        exclude_stopwords=True,

        # 임베딩
        embedding_method='concat',  # bert + word2vec
        embed_size=128,
        w2v_dim=64,
        bert_dim=64,

        # GraphMAE
        graphmae_epochs=1000,
        graphmae_lr=0.001,
        graphmae_device=None,  # 자동 감지
        mask_rate=0.3,
        encoder_type='gcn',
        decoder_type='gcn',

        # 클러스터링
        clustering_method='kmeans',
        num_clusters=None,  # Elbow method로 자동 결정
        min_clusters=3,
        max_clusters=20,

        # 평가
        eval_metrics=['silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi'],

        # 출력
        save_results=True,
        output_dir="results/grace_gcn_edge_weight",
        save_graph_viz=True,
        save_embeddings=True,

        # 디버그
        verbose=True,
        log_interval=10
    )

    return config


def run_training(config: GRACEConfig, verbose: bool = True):
    """전체 파이프라인 학습 실행"""

    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.GREEN}Starting GRACE Pipeline")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

    # 파이프라인 초기화 및 실행
    pipeline = GRACEPipeline(config)
    results = pipeline.run()

    # 결과 요약 출력
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.GREEN}Results Summary")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

    # 메트릭 출력
    metrics = results.get('metrics', {})
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {Fore.CYAN}{metric_name:20s}{Style.RESET_ALL}: {value:.4f}")

    # 클러스터 분포
    if 'cluster_info' in results:
        print(f"\n{Fore.YELLOW}Cluster Distribution:{Style.RESET_ALL}")
        cluster_info = results['cluster_info']
        for cluster_id, info in cluster_info.items():
            count = info.get('count', 0)
            print(f"  Cluster {cluster_id:2d}: {count:4d} words")

    # 상위 단어 출력 (각 클러스터별)
    if 'cluster_top_words' in results:
        print(f"\n{Fore.YELLOW}Top Words per Cluster:{Style.RESET_ALL}")
        for cluster_id, top_words in list(results['cluster_top_words'].items())[:5]:
            words = ', '.join(top_words[:10])
            print(f"\n  {Fore.CYAN}Cluster {cluster_id}{Style.RESET_ALL}:")
            print(f"    {words}")

    print(f"\n{Fore.GREEN}✓ Training completed successfully!{Style.RESET_ALL}\n")
    print(f"  Results saved to: {config.output_dir}")

    return pipeline, results


def run_traditional_comparison(config: GRACEConfig):
    """전통적 그래프 클러스터링과 GRACE 비교"""

    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.GREEN}Comparison: GRACE vs Traditional Graph Clustering")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

    print(f"{Fore.YELLOW}Note: Traditional comparison is a separate analysis mode.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Running full GRACE pipeline...{Style.RESET_ALL}\n")

    # GRACE 파이프라인 실행
    pipeline = GRACEPipeline(config)
    results = pipeline.run()

    print(f"\n{Fore.GREEN}✓ Comparison completed!{Style.RESET_ALL}")
    print(f"  Results saved to: {config.output_dir}")

    return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="GRACE: GRAph-based Clustering with Enhanced embeddings (Edge Weight Integration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 학습 실행
  python main.py --mode train

  # 설정 파일로 학습
  python main.py --mode train --config configs/my_config.json

  # GAT vs GCN+EdgeWeight 비교
  python main.py --mode compare --max-docs 5000 --epochs 500

  # 전통적 모델과 비교 (Louvain, Leiden, etc.)
  python main.py --mode compare-traditional --max-docs 10000

  # GPU 사용
  python main.py --mode train --device cuda

  # 소규모 테스트
  python main.py --mode train --max-docs 1000 --epochs 100

  # Ablation study는 별도 스크립트 사용
  python ablation_main.py --all
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'compare', 'compare-traditional'],
        help='실행 모드 (기본값: train)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='설정 파일 경로 (JSON)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/reddit_mental_health_cleaned.csv',
        help='데이터 파일 경로'
    )

    parser.add_argument(
        '--max-docs',
        type=int,
        default=10000,
        help='최대 문서 수'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='GraphMAE 학습 에폭'
    )

    parser.add_argument(
        '--embed-size',
        type=int,
        default=128,
        help='임베딩 차원'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='디바이스 (기본값: 자동 감지)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/grace_gcn_edge_weight',
        help='출력 디렉토리'
    )

    parser.add_argument(
        '--encoder-type',
        type=str,
        default='gat',
        choices=['gat', 'tsgat', 'gcn', 'mlp', 'linear'],
        help='GraphMAE encoder type (기본값: gcn)'
    )

    parser.add_argument(
        '--decoder-type',
        type=str,
        default='gat',
        choices=['gat', 'tsgat', 'gcn', 'mlp', 'linear'],
        help='GraphMAE decoder type (기본값: gcn)'
    )

    parser.add_argument(
        '--mask-rate',
        type=float,
        default=0.3,
        help='GraphMAE mask rate (기본값: 0.3)'
    )

    parser.add_argument(
        '--no-graphmae',
        action='store_true',
        help='GraphMAE 사용하지 않음'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='상세 출력'
    )

    args = parser.parse_args()

    # 배너 출력
    print_banner()

    # 설정 생성
    if args.config:
        print(f"{Fore.YELLOW}Loading config from: {args.config}{Style.RESET_ALL}")
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        # TODO: config_dict로부터 GRACEConfig 생성
        config = create_default_config()
    else:
        config = create_default_config()

    # CLI 인자로 설정 덮어쓰기
    config.csv_path = args.data
    config.num_documents = args.max_docs
    config.embed_size = args.embed_size
    config.output_dir = args.output

    # GraphMAE 관련 설정
    if not args.no_graphmae:
        config.graphmae_epochs = args.epochs
        config.graphmae_device = args.device
        config.encoder_type = args.encoder_type
        config.decoder_type = args.decoder_type
        config.mask_rate = args.mask_rate

    # 설정 출력
    print(f"\n{Fore.CYAN}Configuration:{Style.RESET_ALL}")
    print(f"  Data: {config.csv_path}")
    print(f"  Max docs: {config.num_documents}")
    print(f"  Embedding method: {config.embedding_method}")
    print(f"  Embed size: {config.embed_size}")
    print(f"  GraphMAE epochs: {config.graphmae_epochs}")
    print(f"  Device: {config.graphmae_device if config.graphmae_device else 'auto'}")
    print(f"  Encoder type: {config.encoder_type}")
    print(f"  Decoder type: {config.decoder_type}")
    print(f"  Mask rate: {config.mask_rate}")
    print(f"  Output: {config.output_dir}")

    # 모드별 실행
    try:
        if args.mode == 'train':
            run_training(config, verbose=args.verbose)

        elif args.mode == 'compare':
            # 비교 모드 (현재는 단순히 학습 실행)
            print(f"{Fore.YELLOW}Running comparison mode...{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Note: For detailed comparisons, consider implementing specific comparison logic{Style.RESET_ALL}\n")
            run_training(config, verbose=args.verbose)

        elif args.mode == 'compare-traditional':
            # 전통적 그래프 클러스터링과 비교
            run_traditional_comparison(config)

    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Interrupted by user.{Style.RESET_ALL}")
        sys.exit(1)

    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
