"""
기본 그래프 생성 테스트 (DGL 없이)
- kaggle_RC_2019-05.csv 데이터셋 사용
- 상위 500개 단어로 공출현 그래프 생성
- Word2Vec(32) + BERT(32) = 64차원 concat 특성
"""

import os
import sys
import pandas as pd
import torch
from typing import List

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 필요한 모듈들 import
from services.Document.DocumentService import DocumentService
from services.Graph.GraphService import GraphService
from services.Graph.NodeFeatureHandler import NodeFeatureHandler
from entities import Word, WordGraph, NodeFeatureType


def load_dataset(csv_path: str, num_documents: int = 100) -> List[str]:
    """CSV 데이터셋 로드"""
    print(f"📁 데이터셋 로드 중: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✓ 전체 데이터 크기: {len(df)} 행")

    if 'body' not in df.columns:
        raise ValueError("CSV 파일에 'body' 컬럼이 없습니다.")

    documents = df['body'].dropna().head(num_documents).tolist()
    print(f"✓ 로드된 문서 수: {len(documents)}")

    return documents


def test_basic_graph():
    """기본 그래프 생성 테스트"""
    print("🎉 기본 그래프 생성 테스트 시작!")
    print("=" * 50)

    try:
        # 1. 데이터셋 로드
        csv_path = "/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv"
        documents = load_dataset(csv_path, num_documents=100)

        # 2. DocumentService 초기화
        print("\n🔧 서비스 초기화 중...")
        doc_service = DocumentService()
        print("✓ DocumentService 초기화 완료")

        # 3. 문서 처리
        doc_service.create_sentence_list(documents=documents)
        print(f"✓ 문장 처리 완료 - {doc_service.get_sentence_count()}개 문장")

        # 단어 리스트는 문장 처리 시 자동으로 생성됨
        top_words = doc_service.get_top_words(100)
        print(f"✓ 단어 처리 완료 - 상위 {len(top_words)}개 단어")

        # 4. GraphService 초기화
        graph_service = GraphService(doc_service)
        print("✓ GraphService 초기화 완료")

        # 5. 공출현 그래프 생성
        print(f"\n📊 공출현 그래프 생성 중 (상위 50개 단어)...")
        word_graph = graph_service.build_complete_graph(
            top_n=50,  # 작은 수로 테스트
            exclude_stopwords=True,
            max_length=-1
        )

        print(f"✓ 그래프 생성 완료")
        print(f"  - 노드 수: {word_graph.num_nodes}")
        print(f"  - 엣지 수: {word_graph.num_edges}")

        # 6. NodeFeatureHandler 테스트
        print(f"\n🎯 노드 특성 테스트 중...")
        node_handler = NodeFeatureHandler(graph_service.doc_service)

        # Word2Vec 특성 테스트
        print("Word2Vec 특성 계산 중...")
        try:
            w2v_features = node_handler.calculate_embeddings(
                word_graph.words[:5],  # 처음 5개 단어만
                method='w2v',
                embed_size=32
            )
            print(f"✓ Word2Vec 특성: {w2v_features.shape}")
        except Exception as e:
            print(f"❌ Word2Vec 실패: {e}")
            return

        # BERT 특성 테스트
        print("BERT 특성 계산 중...")
        try:
            bert_features = node_handler.calculate_embeddings(
                word_graph.words[:5],  # 처음 5개 단어만
                method='bert',
                embed_size=64  # BERT는 embed_size 무시
            )
            print(f"✓ BERT 특성: {bert_features.shape}")
        except Exception as e:
            print(f"❌ BERT 실패: {e}")
            return

        # Concat 특성 테스트
        print("Concat 특성 계산 중...")
        try:
            concat_features = node_handler.calculate_embeddings(
                word_graph.words[:5],  # 처음 5개 단어만
                method='concat',
                embed_size=64
            )
            print(f"✓ Concat 특성: {concat_features.shape}")
        except Exception as e:
            print(f"❌ Concat 실패: {e}")
            return

        # 7. WordGraph에 특성 설정 테스트
        print(f"\n🔧 WordGraph 특성 설정 테스트...")
        all_concat_features = node_handler.calculate_embeddings(
            word_graph.words,
            method='concat',
            embed_size=64
        )

        word_graph.set_node_features_custom(all_concat_features, NodeFeatureType.CUSTOM)
        print(f"✓ 노드 특성 설정 완료: {word_graph.node_features.shape}")

        # 8. 상위 단어들 출력
        print(f"\n📋 상위 10개 단어:")
        for i, word in enumerate(word_graph.words[:10]):
            print(f"  {i+1:2d}. {word.content:<15} (빈도: {word.freq})")

        print("\n🎊 기본 그래프 테스트 완료!")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_basic_graph()