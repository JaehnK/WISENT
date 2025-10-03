"""
간단한 GraphMAE 모델 테스트
- DGL 호환성 문제를 우회하여 GraphMAE 모델 자체만 테스트
- 더 큰 데이터셋으로 테스트
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
from typing import List

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# GraphMAE2 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'GraphMAE2'))

# 필요한 모듈들 import
from services.Document.DocumentService import DocumentService
from services.Graph.GraphService import GraphService
from services.Graph.NodeFeatureHandler import NodeFeatureHandler
from entities import Word, WordGraph, NodeFeatureType


def create_simple_dgl_graph(num_nodes: int, node_features: torch.Tensor):
    """
    간단한 DGL 그래프 수동 생성 (import 문제 우회)
    """
    try:
        import dgl
        import torch

        # 간단한 순환 그래프 생성 (각 노드가 다음 노드와 연결)
        src = list(range(num_nodes))
        dst = [(i + 1) % num_nodes for i in range(num_nodes)]

        # 양방향 엣지 추가
        src_all = src + dst
        dst_all = dst + src

        g = dgl.graph((src_all, dst_all), num_nodes=num_nodes)
        g.ndata['feat'] = node_features

        return g

    except ImportError:
        print("DGL not available, creating mock graph structure")
        # DGL이 없으면 모의 그래프 구조 반환
        class MockGraph:
            def __init__(self, num_nodes, node_features):
                self.num_nodes_val = num_nodes
                self.ndata = {'feat': node_features}

            def num_nodes(self):
                return self.num_nodes_val

            def to(self, device):
                self.ndata['feat'] = self.ndata['feat'].to(device)
                return self

        return MockGraph(num_nodes, node_features)


def test_graphmae_model():
    """GraphMAE 모델 구조 테스트"""
    print("🚀 GraphMAE 모델 테스트 시작!")
    print("=" * 50)

    try:
        # GraphMAE 모델 import 테스트
        try:
            from models.edcoder import PreModel
            print("✓ GraphMAE PreModel import 성공")
        except ImportError as e:
            print(f"❌ GraphMAE PreModel import 실패: {e}")
            return

        # 1. 기본 설정
        embed_size = 64
        num_nodes = 100
        num_hidden = embed_size

        print(f"테스트 설정:")
        print(f"  - 노드 수: {num_nodes}")
        print(f"  - 임베딩 크기: {embed_size}")
        print(f"  - 히든 크기: {num_hidden}")

        # 2. 모의 노드 특성 생성
        node_features = torch.randn(num_nodes, embed_size, dtype=torch.float32)
        print(f"✓ 노드 특성 생성: {node_features.shape}")

        # 3. 모의 DGL 그래프 생성
        dgl_graph = create_simple_dgl_graph(num_nodes, node_features)
        print(f"✓ DGL 그래프 생성 완료")

        # 4. GraphMAE 모델 생성
        print("\n🔧 GraphMAE 모델 초기화 중...")
        model = PreModel(
            in_dim=embed_size,
            num_hidden=num_hidden,
            num_layers=2,
            num_dec_layers=1,
            num_remasking=1,
            nhead=4,
            nhead_out=1,
            activation='relu',
            feat_drop=0.2,
            attn_drop=0.1,
            negative_slope=0.2,
            residual=True,
            norm='layernorm',
            mask_rate=0.3,
            remask_rate=0.5,
            remask_method='random',
            mask_method='random',
            encoder_type='gat',
            decoder_type='gat',
            loss_fn='sce',
            drop_edge_rate=0.0,
            alpha_l=2.0,
            lam=1.0,
            delayed_ema_epoch=0,
            momentum=0.996,
            replace_rate=0.0,
            zero_init=False
        )

        print("✓ GraphMAE 모델 생성 완료")
        print(f"  - 인코더 파라미터: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad):,}")
        print(f"  - 디코더 파라미터: {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad):,}")

        # 5. 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ 디바이스: {device}")

        model = model.to(device)
        dgl_graph = dgl_graph.to(device)

        # 6. 순전파 테스트
        print("\n🧪 순전파 테스트 중...")
        model.train()

        x = dgl_graph.ndata['feat']
        print(f"  - 입력 특성: {x.shape}")

        # 훈련 모드에서 손실 계산
        try:
            loss = model(dgl_graph, x, epoch=0)
            print(f"✓ 순전파 성공! 손실: {loss.item():.4f}")
        except Exception as e:
            print(f"❌ 순전파 실패: {e}")
            return

        # 7. 임베딩 추출 테스트
        print("\n🎯 임베딩 추출 테스트 중...")
        model.eval()

        with torch.no_grad():
            embeddings = model.embed(dgl_graph, x)
            print(f"✓ 임베딩 추출 성공: {embeddings.shape}")

            # 임베딩 통계
            print(f"  - 평균: {embeddings.mean().item():.4f}")
            print(f"  - 표준편차: {embeddings.std().item():.4f}")
            print(f"  - 최솟값: {embeddings.min().item():.4f}")
            print(f"  - 최댓값: {embeddings.max().item():.4f}")

        # 8. 간단한 훈련 루프 테스트
        print("\n🏃 간단한 훈련 테스트 중...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            loss = model(dgl_graph, x, epoch=epoch)
            loss.backward()
            optimizer.step()

            print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")

        print("\n🎊 GraphMAE 모델 테스트 완료!")
        print("=" * 50)
        print("✅ 모든 테스트 통과!")

    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()


def test_larger_dataset():
    """더 큰 데이터셋으로 기본 그래프 테스트"""
    print("\n📊 더 큰 데이터셋 테스트 시작!")
    print("=" * 50)

    try:
        # 1. 더 많은 문서 로드
        csv_path = "/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv"
        df = pd.read_csv(csv_path)
        documents = df['body'].dropna().head(500).tolist()  # 500개 문서
        print(f"✓ 로드된 문서 수: {len(documents)}")

        # 2. DocumentService 초기화
        doc_service = DocumentService()
        doc_service.create_sentence_list(documents=documents)
        print(f"✓ 문장 처리 완료 - {doc_service.get_sentence_count()}개 문장")

        # 3. GraphService 초기화
        graph_service = GraphService(doc_service)

        # 4. 더 큰 그래프 생성
        word_graph = graph_service.build_complete_graph(
            top_n=200,  # 200개 단어
            exclude_stopwords=True,
            max_length=-1
        )
        print(f"✓ 그래프 생성 완료")
        print(f"  - 노드 수: {word_graph.num_nodes}")
        print(f"  - 엣지 수: {word_graph.num_edges}")

        # 5. Concat 특성 설정
        node_handler = NodeFeatureHandler(graph_service.doc_service)
        concat_features = node_handler.calculate_embeddings(
            word_graph.words,
            method='concat',
            embed_size=64
        )
        print(f"✓ Concat 특성 설정 완료: {concat_features.shape}")

        # 6. WordGraph에 특성 설정
        word_graph.set_node_features_custom(concat_features, NodeFeatureType.CUSTOM)
        print(f"✓ WordGraph 특성 설정 완료")

        # 7. 상위 연결 단어들 출력
        print(f"\n📋 상위 연결 단어들 (빈도순):")
        for i, word in enumerate(word_graph.words[:10]):
            print(f"  {i+1:2d}. {word.content:<15} (빈도: {word.freq})")

        print("\n🎊 더 큰 데이터셋 테스트 완료!")

    except Exception as e:
        print(f"\n❌ 더 큰 데이터셋 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """메인 실행 함수"""
    print("🎉 GraphMAE 통합 테스트!")
    print("=" * 60)

    # 1. GraphMAE 모델 테스트
    test_graphmae_model()

    # 2. 더 큰 데이터셋 테스트
    test_larger_dataset()


if __name__ == "__main__":
    main()