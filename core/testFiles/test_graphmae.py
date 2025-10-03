"""
GraphMAE 파이프라인 스크립트 - 논문 흐름

파이프라인:
1. 데이터 로드 (kaggle_RC_2019-05.csv)
2. 문서 전처리 (DocumentService)
3. 공출현 그래프 생성 (상위 500개 단어)
4. 노드 특성 계산: Word2Vec(32) + BERT(32) = 64차원 concat
5. GraphMAE 사전학습 (그래프 구조 + 노드 특성)
6. 학습된 임베딩 추출
7. 결과 분석 및 시각화

다음 단계: 추출된 임베딩으로 단어 클러스터링 (K-Means, DBSCAN 등)
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
from typing import List

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 필요한 모듈들 import
from services.Document.DocumentService import DocumentService
from services.Graph.GraphService import GraphService
from services.Graph.NodeFeatureHandler import NodeFeatureHandler
from services.GraphMAE import GraphMAEService, GraphMAEConfig
from entities import Word, WordGraph, NodeFeatureType


def load_dataset(csv_path: str, num_documents: int = 1000) -> List[str]:
    """
    CSV 데이터셋 로드

    Args:
        csv_path: CSV 파일 경로
        num_documents: 로드할 문서 수

    Returns:
        문서 리스트
    """
    print(f"📁 데이터셋 로드 중: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✓ 전체 데이터 크기: {len(df)} 행")

    # 'body' 컬럼에서 문서 추출
    if 'body' not in df.columns:
        raise ValueError("CSV 파일에 'body' 컬럼이 없습니다.")

    # NaN 값 제거 및 문서 선택
    documents = df['body'].dropna().head(num_documents).tolist()
    print(f"✓ 로드된 문서 수: {len(documents)}")

    return documents


def create_graph_service(documents: List[str]) -> GraphService:
    """
    DocumentService와 GraphService 초기화

    Args:
        documents: 문서 리스트

    Returns:
        초기화된 GraphService
    """
    print("\n🔧 서비스 초기화 중...")

    # DocumentService 초기화
    doc_service = DocumentService()
    print("✓ DocumentService 초기화 완료")

    # 문서 처리
    doc_service.create_sentence_list(documents=documents)
    print(f"✓ 문장 처리 완료 - {doc_service.get_sentence_count()}개 문장")

    # 단어 처리 (문장 처리 시 자동 생성됨)
    print(f"✓ 단어 처리 완료 - {len(doc_service.words_list)}개 단어")

    # GraphService 초기화
    graph_service = GraphService(doc_service)
    print("✓ GraphService 초기화 완료")

    return graph_service


def build_cooccurrence_graph(graph_service: GraphService, top_n: int = 500) -> WordGraph:
    """
    공출현 그래프 생성

    Args:
        graph_service: GraphService 인스턴스
        top_n: 상위 단어 수

    Returns:
        공출현 그래프
    """
    print(f"\n📊 공출현 그래프 생성 중 (상위 {top_n}개 단어)...")

    # 공출현 그래프 생성
    word_graph = graph_service.build_complete_graph(
        top_n=top_n,
        exclude_stopwords=True,
        max_length=-1
    )

    print(f"✓ 그래프 생성 완료")
    print(f"  - 노드 수: {word_graph.num_nodes}")
    print(f"  - 엣지 수: {word_graph.num_edges}")

    return word_graph


def set_node_features(graph_service: GraphService, word_graph: WordGraph,
                      method: str = 'concat', embed_size: int = 64) -> torch.Tensor:
    """
    노드 특성 계산 및 설정

    Args:
        graph_service: GraphService 인스턴스
        word_graph: 특성을 설정할 WordGraph
        method: 임베딩 방법 ('concat', 'w2v', 'bert')
        embed_size: 총 임베딩 크기

    Returns:
        계산된 노드 특성 텐서
    """
    print(f"\n🎯 노드 특성 계산 중 (method={method}, embed_size={embed_size})...")

    # NodeFeatureHandler 사용
    node_handler = NodeFeatureHandler(graph_service.doc_service)

    # 임베딩 계산
    if method == 'concat':
        print("  - Word2Vec + BERT concatenation")
    elif method == 'w2v':
        print("  - Word2Vec only")
    elif method == 'bert':
        print("  - BERT only")

    node_features = node_handler.calculate_embeddings(
        word_graph.words,
        method=method,
        embed_size=embed_size
    )

    print(f"✓ 노드 특성 계산 완료: {node_features.shape}")

    # WordGraph에 특성 설정
    word_graph.set_node_features_custom(node_features, NodeFeatureType.CUSTOM)
    print("✓ WordGraph에 노드 특성 설정 완료")

    return node_features


def train_graphmae(graph_service: GraphService, word_graph: WordGraph,
                   epochs: int = 10, device: str = None) -> torch.Tensor:
    """
    GraphMAE 사전학습 및 임베딩 추출

    Args:
        graph_service: GraphService 인스턴스
        word_graph: 학습할 WordGraph (노드 특성이 이미 설정되어 있어야 함)
        epochs: 학습 에포크 수
        device: 학습 디바이스 ('cuda' 또는 'cpu', None이면 자동 선택)

    Returns:
        학습된 GraphMAE 임베딩 [num_nodes, embed_size]
    """
    print(f"\n🚀 GraphMAE 사전학습 시작...")

    # 노드 특성이 설정되어 있는지 확인
    if word_graph.node_features is None:
        raise ValueError("WordGraph에 노드 특성이 설정되지 않았습니다. set_node_features()를 먼저 호출하세요.")

    embed_size = word_graph.node_features.shape[1]
    print(f"  - 입력 차원: {embed_size}")
    print(f"  - 노드 수: {word_graph.num_nodes}")
    print(f"  - 엣지 수: {word_graph.num_edges}")

    # GraphMAE 설정
    config = GraphMAEConfig.create_default(embed_size)
    config.max_epochs = epochs
    config.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✓ 학습 설정: {config.max_epochs} epochs, device: {config.device}")
    if torch.cuda.is_available() and config.device == "cuda":
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")

    # GraphMAE 서비스 초기화
    mae_service = GraphMAEService(graph_service, config)

    # DGL 그래프로 변환 (이미 설정된 노드 특성 사용)
    dgl_graph = graph_service.wordgraph_to_dgl(word_graph, word_graph.node_features)

    # 모델 생성 및 학습
    mae_service.model = mae_service.create_mae_model(embed_size)
    device_obj = torch.device(config.device)
    mae_service.model.to(device_obj)
    dgl_graph = dgl_graph.to(device_obj)

    optimizer = torch.optim.Adam(
        mae_service.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 학습 루프
    mae_service.model.train()
    for epoch in range(config.max_epochs):
        optimizer.zero_grad()
        x = dgl_graph.ndata['feat']
        loss = mae_service.model(dgl_graph, x, epoch=epoch)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, config.max_epochs // 10) == 0:
            print(f"  Epoch {epoch + 1}/{config.max_epochs}, Loss: {loss.item():.4f}")

    # 임베딩 추출
    mae_service.model.eval()
    with torch.no_grad():
        embeddings = mae_service.model.embed(dgl_graph, dgl_graph.ndata['feat'])

    print(f"✓ GraphMAE 학습 완료: {embeddings.shape}")
    return embeddings.cpu()


def analyze_results(original_features: torch.Tensor, mae_embeddings: torch.Tensor,
                   word_graph: WordGraph) -> None:
    """
    결과 분석 및 출력

    Args:
        original_features: 원본 특성
        mae_embeddings: GraphMAE 임베딩
        word_graph: WordGraph 객체
    """
    print(f"\n📈 결과 분석...")

    print(f"원본 특성 형태: {original_features.shape}")
    print(f"GraphMAE 임베딩 형태: {mae_embeddings.shape}")

    # 코사인 유사도 계산 (첫 10개 단어)
    from torch.nn.functional import cosine_similarity

    print(f"\n상위 10개 단어의 원본 vs GraphMAE 코사인 유사도:")
    print("-" * 50)

    for i in range(min(10, len(word_graph.words))):
        word = word_graph.words[i]
        orig_vec = original_features[i].unsqueeze(0)
        mae_vec = mae_embeddings[i].unsqueeze(0)

        similarity = cosine_similarity(orig_vec, mae_vec).item()
        print(f"{word.content:<15} {similarity:.4f}")

    # 임베딩 통계
    print(f"\nGraphMAE 임베딩 통계:")
    print(f"  평균: {mae_embeddings.mean().item():.4f}")
    print(f"  표준편차: {mae_embeddings.std().item():.4f}")
    print(f"  최솟값: {mae_embeddings.min().item():.4f}")
    print(f"  최댓값: {mae_embeddings.max().item():.4f}")


def main():
    """
    메인 실행 함수 - 논문 흐름에 맞춘 파이프라인

    파이프라인:
    1. 데이터 로드
    2. 문서 전처리 (DocumentService)
    3. 공출현 그래프 생성 (상위 N개 단어)
    4. 노드 특성 계산 (Word2Vec + BERT concat)
    5. GraphMAE 사전학습
    6. 학습된 임베딩 추출
    7. (다음 단계: 클러스터링)
    """
    print("🎉 GraphMAE 파이프라인 시작!")
    print("=" * 80)

    try:
        # ===== 1. 데이터셋 로드 =====
        csv_path = "/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv"
        documents = load_dataset(csv_path, num_documents=500)

        # ===== 2. 문서 전처리 및 서비스 초기화 =====
        graph_service = create_graph_service(documents)

        # ===== 3. 공출현 그래프 생성 =====
        top_n = 500
        word_graph = build_cooccurrence_graph(graph_service, top_n=top_n)

        # ===== 4. 노드 특성 계산 (Word2Vec + BERT concat) =====
        embed_size = 64
        node_features = set_node_features(
            graph_service,
            word_graph,
            method='concat',  # Word2Vec + BERT
            embed_size=embed_size
        )

        # ===== 5. GraphMAE 사전학습 =====
        epochs = 10  # 테스트용 짧은 학습
        graphmae_embeddings = train_graphmae(
            graph_service,
            word_graph,
            epochs=epochs,
            device=None  # 자동 선택 (CUDA 우선)
        )

        # ===== 6. 결과 분석 =====
        print(f"\n📊 최종 결과:")
        print(f"  - 원본 노드 특성: {node_features.shape}")
        print(f"  - GraphMAE 임베딩: {graphmae_embeddings.shape}")

        analyze_results(node_features, graphmae_embeddings, word_graph)

        print("\n" + "=" * 80)
        print("🎊 GraphMAE 파이프라인 완료!")
        print("=" * 80)
        print("\n💡 다음 단계: 클러스터링 (K-Means, DBSCAN 등)")

    except Exception as e:
        print(f"\n❌ 파이프라인 실패: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()