"""
Traditional Graph Clustering 간단 테스트

GRACE 없이 전통적 그래프 클러스터링만 빠르게 테스트합니다.
"""

import sys
from pathlib import Path
import pandas as pd

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'core'))

from services.GRACE import TraditionalGraphClusteringService
from services.Document import DocumentService
from services.Graph import GraphService


def main():
    print("=" * 80)
    print("🔬 Traditional Graph Clustering 테스트")
    print("=" * 80)
    
    # 설정
    csv_path = '/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv'
    num_documents = 10000
    top_n_words = 500
    text_column = 'body'
    
    print(f"\n📊 설정:")
    print(f"  - 문서 수: {num_documents}")
    print(f"  - 노드 수: {top_n_words}")
    
    # Step 1: 데이터 전처리
    print("\n[1/4] 데이터 로딩 및 전처리...")
    
    # CSV 로드 (run_grace_visualization.py 패턴)
    df = pd.read_csv(csv_path)
    print(f"  전체 데이터: {len(df)} 행")
    
    if text_column not in df.columns:
        raise ValueError(f"CSV에 '{text_column}' 컬럼이 없습니다. 사용 가능한 컬럼: {df.columns.tolist()}")
    
    # 문서 추출
    documents = df[text_column].dropna().head(num_documents).tolist()
    print(f"  로드된 문서: {len(documents)}개")
    
    # DocumentService 초기화 및 전처리
    doc_service = DocumentService()
    doc_service.create_sentence_list(documents=documents)
    print(f"✅ 전처리 완료")
    print(f"  - 문장: {doc_service.get_sentence_count()}개")
    print(f"  - 단어: {len(doc_service.words_list) if doc_service.words_list else 0}개")
    
    # Step 2: 의미연결망 구축
    print("\n[2/4] 의미연결망 구축...")
    graph_service = GraphService(doc_service)
    word_graph = graph_service.build_complete_graph(
        top_n=top_n_words,
        exclude_stopwords=True
    )
    print(f"✅ 그래프 생성 완료:")
    print(f"   - 노드: {word_graph.num_nodes}")
    print(f"   - 엣지: {word_graph.num_edges}")
    
    # Step 3: 전통적 클러스터링
    print("\n[3/4] 클러스터링 수행...")
    traditional = TraditionalGraphClusteringService(random_state=42)
    
    # Louvain
    print("\n🔵 Louvain:")
    louvain_labels, louvain_metrics = traditional.louvain_clustering(word_graph)
    print(f"   클러스터 수: {louvain_metrics['num_clusters']}")
    print(f"   Modularity: {louvain_metrics['modularity']:.4f}")
    
    # 클러스터 분포
    distribution = traditional.get_cluster_distribution()
    print(f"   클러스터 분포: {distribution}")
    
    # Leiden (설치되어 있으면)
    print("\n🟢 Leiden:")
    try:
        leiden_labels, leiden_metrics = traditional.leiden_clustering(word_graph)
        print(f"   클러스터 수: {leiden_metrics['num_clusters']}")
        print(f"   Modularity: {leiden_metrics['modularity']:.4f}")
    except ImportError as e:
        print(f"   ⚠️  스킵 (패키지 없음): {e}")
    
    # Step 4: 클러스터 단어 출력
    print("\n[4/4] 클러스터 분석...")
    print("\n상위 3개 클러스터의 대표 단어:")
    
    # Louvain 결과 사용
    traditional.cluster_labels = louvain_labels
    traditional.num_clusters = louvain_metrics['num_clusters']
    
    num_clusters_to_show = min(3, traditional.num_clusters)
    
    for cluster_id in range(num_clusters_to_show):
        words = traditional.get_cluster_words(word_graph, cluster_id, top_n=10)
        print(f"\n📌 Cluster {cluster_id} (크기: {distribution.get(cluster_id, 0)}):")
        print(f"   {', '.join([w for w, _ in words])}")
    
    # 그래프 통계
    print("\n📊 그래프 통계:")
    stats = traditional.compute_graph_statistics()
    print(f"   - 밀도: {stats['density']:.4f}")
    print(f"   - 평균 차수: {stats['average_degree']:.2f}")
    print(f"   - 연결됨: {stats['is_connected']}")
    if not stats['is_connected']:
        print(f"   - 연결 성분 수: {stats['num_connected_components']}")
    
    # 결과 저장
    print("\n💾 결과 저장...")
    output_dir = Path('./traditional_clustering_output')
    output_dir.mkdir(exist_ok=True)
    
    traditional.save_clustering_results(
        word_graph,
        output_path=str(output_dir / 'louvain_results.json'),
        include_words=True
    )
    
    print("\n" + "=" * 80)
    print("✅ 테스트 완료!")
    print("=" * 80)
    print(f"\n💡 다음 단계:")
    print(f"   1. GRACE와 비교: python examples/compare_grace_with_traditional.py")
    print(f"   2. 결과 확인: {output_dir}/louvain_results.json")


if __name__ == '__main__':
    main()
