"""
GRACE (GRAph-based Clustering with Enhanced embeddings) 테스트 스크립트

사용 예시:
python test_grace.py --documents 500 --clusters 10
python test_grace.py --documents 1000 --epochs 100 --auto-cluster
"""

import argparse
from services.GRACE import GRACEPipeline, GRACEConfig


def parse_args():
    parser = argparse.ArgumentParser(description='GRACE 파이프라인 실행')

    # 데이터 설정
    parser.add_argument('--csv', type=str,
                        default='/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv',
                        help='CSV 파일 경로')
    parser.add_argument('--documents', type=int, default=500,
                        help='사용할 문서 수')
    parser.add_argument('--top-words', type=int, default=500,
                        help='상위 단어 수 (노드)')

    # 임베딩 설정
    parser.add_argument('--embed-method', type=str, default='concat',
                        choices=['concat', 'w2v', 'bert'],
                        help='임베딩 방법')
    parser.add_argument('--embed-size', type=int, default=64,
                        help='임베딩 차원')

    # GraphMAE 설정
    parser.add_argument('--epochs', type=int, default=100,
                        help='GraphMAE 학습 에포크')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--mask-rate', type=float, default=0.75,
                        help='GraphMAE mask rate')

    # 클러스터링 설정
    parser.add_argument('--clusters', type=int, default=None,
                        help='클러스터 수 (지정하지 않으면 자동 탐색)')
    parser.add_argument('--auto-cluster', action='store_true',
                        help='최적 클러스터 수 자동 탐색')
    parser.add_argument('--min-clusters', type=int, default=3,
                        help='최소 클러스터 수 (자동 탐색 시)')
    parser.add_argument('--max-clusters', type=int, default=20,
                        help='최대 클러스터 수 (자동 탐색 시)')

    # 출력 설정
    parser.add_argument('--output-dir', type=str, default='./grace_output',
                        help='결과 저장 디렉토리')
    parser.add_argument('--no-save', action='store_true',
                        help='결과 저장 안 함')
    parser.add_argument('--quiet', action='store_true',
                        help='로그 출력 안 함')

    return parser.parse_args()


def main():
    args = parse_args()

    # 설정 생성
    config = GRACEConfig(
        # 데이터
        csv_path=args.csv,
        num_documents=args.documents,
        top_n_words=args.top_words,

        # 임베딩
        embedding_method=args.embed_method,
        embed_size=args.embed_size,
        w2v_dim=32 if args.embed_method == 'concat' else args.embed_size,
        bert_dim=32 if args.embed_method == 'concat' else args.embed_size,

        # GraphMAE
        graphmae_epochs=args.epochs,
        graphmae_lr=args.lr,
        mask_rate=args.mask_rate,

        # 클러스터링
        num_clusters=None if args.auto_cluster else args.clusters,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,

        # 출력
        output_dir=args.output_dir,
        save_results=not args.no_save,
        verbose=not args.quiet
    )

    # 파이프라인 실행
    pipeline = GRACEPipeline(config)
    results = pipeline.run()

    # 결과 요약 출력
    print("\n" + "=" * 80)
    print("📊 GRACE 결과 요약")
    print("=" * 80)
    print(f"클러스터 수: {results['num_clusters']}")
    print(f"\n평가 지표:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")

    print(f"\n클러스터별 단어 수:")
    for cluster_id, count in results['cluster_distribution'].items():
        print(f"  Cluster {cluster_id}: {count}개")

    # 각 클러스터의 상위 단어 출력
    print(f"\n클러스터별 대표 단어 (상위 10개):")
    for cluster_id in sorted(results['clusters'].keys()):
        words = results['clusters'][cluster_id][:10]
        print(f"  Cluster {cluster_id}: {', '.join(words)}")

    # CSV 저장
    if not args.no_save:
        csv_path = f"{args.output_dir}/clusters.csv"
        pipeline.export_cluster_csv(csv_path)
        print(f"\n💾 클러스터 결과 CSV 저장: {csv_path}")


if __name__ == "__main__":
    main()
