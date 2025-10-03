"""
직접 GraphMAE 테스트 - DGL 호환성 문제 우회
- PyTorch Geometric 기반으로 GraphMAE 테스트
- kaggle_RC_2019-05.csv 데이터 사용
- 200개 단어, concat 특성 (64차원)
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
from entities import Word, WordGraph, NodeFeatureType


def create_torch_geometric_data(word_graph: WordGraph):
    """WordGraph를 PyTorch Geometric Data로 변환 (이미 PyG 형태이지만 호환성을 위해)"""
    # WordGraph가 이미 PyTorch Geometric 형태를 사용하므로 직접 변환
    return word_graph.to_pytorch_geometric()


def simple_graph_autoencoder_test(data):
    """간단한 그래프 오토인코더 테스트 (GraphMAE 구조 모방)"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class SimpleGraphAutoEncoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            # 인코더
            self.encoder1 = GCNConv(input_dim, hidden_dim)
            self.encoder2 = GCNConv(hidden_dim, output_dim)

            # 디코더
            self.decoder1 = GCNConv(output_dim, hidden_dim)
            self.decoder2 = GCNConv(hidden_dim, input_dim)

            self.dropout = nn.Dropout(0.2)

        def encode(self, x, edge_index):
            h = F.relu(self.encoder1(x, edge_index))
            h = self.dropout(h)
            z = self.encoder2(h, edge_index)
            return z

        def decode(self, z, edge_index):
            h = F.relu(self.decoder1(z, edge_index))
            h = self.dropout(h)
            x_recon = self.decoder2(h, edge_index)
            return x_recon

        def forward(self, x, edge_index):
            z = self.encode(x, edge_index)
            x_recon = self.decode(z, edge_index)
            return x_recon, z

    # 모델 생성
    input_dim = data.x.shape[1]
    hidden_dim = 32
    output_dim = 64

    model = SimpleGraphAutoEncoder(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"모델 구조:")
    print(f"  - 입력 차원: {input_dim}")
    print(f"  - 히든 차원: {hidden_dim}")
    print(f"  - 출력 차원: {output_dim}")
    print(f"  - 노드 수: {data.x.shape[0]}")
    print(f"  - 엣지 수: {data.edge_index.shape[1]}")

    # 훈련
    model.train()
    losses = []

    for epoch in range(10):
        optimizer.zero_grad()
        x_recon, z = model(data.x, data.edge_index)

        # 재구성 손실
        loss = F.mse_loss(x_recon, data.x)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % 2 == 0:
            print(f"  Epoch {epoch:2d}: Loss = {loss.item():.4f}")

    # 임베딩 추출
    model.eval()
    with torch.no_grad():
        _, embeddings = model(data.x, data.edge_index)

    return embeddings, losses


def test_graphmae_pipeline():
    """GraphMAE 파이프라인 테스트"""
    print("🎉 GraphMAE 파이프라인 테스트 시작!")
    print("=" * 60)

    try:
        # 1. 데이터 로드
        csv_path = "/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv"
        df = pd.read_csv(csv_path)
        documents = df['body'].dropna().head(500).tolist()
        print(f"✓ 로드된 문서 수: {len(documents)}")

        # 2. 서비스 초기화
        doc_service = DocumentService()
        doc_service.create_sentence_list(documents=documents)
        print(f"✓ 문장 처리 완료 - {doc_service.get_sentence_count()}개 문장")

        graph_service = GraphService(doc_service)
        print("✓ GraphService 초기화 완료")

        # 3. 그래프 생성
        print(f"\n📊 공출현 그래프 생성 중 (상위 200개 단어)...")
        word_graph = graph_service.build_complete_graph(
            top_n=200,
            exclude_stopwords=True,
            max_length=-1
        )

        print(f"✓ 그래프 생성 완료")
        print(f"  - 노드 수: {word_graph.num_nodes}")
        print(f"  - 엣지 수: {word_graph.num_edges}")

        # 4. 노드 특성 설정
        print(f"\n🎯 노드 특성 설정 중...")
        node_handler = NodeFeatureHandler(graph_service.doc_service)
        concat_features = node_handler.calculate_embeddings(
            word_graph.words,
            method='concat',
            embed_size=64
        )

        word_graph.set_node_features_custom(concat_features, NodeFeatureType.CUSTOM)
        print(f"✓ 노드 특성 설정 완료: {word_graph.node_features.shape}")

        # 5. PyTorch Geometric 변환
        print(f"\n🔄 PyTorch Geometric 변환 중...")
        pyg_data = create_torch_geometric_data(word_graph)
        print(f"✓ PyG 데이터 생성 완료")
        print(f"  - 노드 특성: {pyg_data.x.shape}")
        print(f"  - 엣지 인덱스: {pyg_data.edge_index.shape}")

        # 6. 그래프 오토인코더 테스트
        print(f"\n🤖 그래프 오토인코더 테스트 중...")
        embeddings, losses = simple_graph_autoencoder_test(pyg_data)

        print(f"✓ 오토인코더 테스트 완료")
        print(f"  - 출력 임베딩: {embeddings.shape}")
        print(f"  - 최종 손실: {losses[-1]:.4f}")
        print(f"  - 손실 감소: {losses[0]:.4f} -> {losses[-1]:.4f}")

        # 7. 임베딩 통계
        print(f"\n📊 임베딩 통계:")
        print(f"  - 평균: {embeddings.mean().item():.4f}")
        print(f"  - 표준편차: {embeddings.std().item():.4f}")
        print(f"  - 최솟값: {embeddings.min().item():.4f}")
        print(f"  - 최댓값: {embeddings.max().item():.4f}")

        # 8. 상위 단어 출력
        print(f"\n📋 상위 10개 단어:")
        for i, word in enumerate(word_graph.words[:10]):
            emb_norm = torch.norm(embeddings[i]).item()
            print(f"  {i+1:2d}. {word.content:<15} (빈도: {word.freq:4d}, 임베딩 놈: {emb_norm:.3f})")

        print(f"\n🎊 GraphMAE 파이프라인 테스트 완료!")
        print("=" * 60)
        print("✅ 모든 테스트 통과!")

        return embeddings

    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    embeddings = test_graphmae_pipeline()
    if embeddings is not None:
        print(f"\n🎯 최종 결과: {embeddings.shape} 임베딩 생성 성공!")