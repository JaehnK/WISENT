import torch
import numpy as np
from typing import List, Dict, Any
from transformers import DistilBertTokenizer, DistilBertModel
import sys
import os

# 상대 임포트와 절대 임포트 모두 지원
try:
    from ..Document.DocumentService import DocumentService
except ImportError:
    # 절대 경로로 임포트 시도
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from services.Document.DocumentService import DocumentService


class BERTService:
    """
    BERT 서비스 - 파이프라인 관리
    """

    def __init__(self, docs: DocumentService, model_path="./DistillBERT/model"):
        
        self.docs = docs
        self.model_path = model_path
        self.cache_path = model_path  # cache_path 추가
        self.model_name = 'distilbert-base-uncased'

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_path
        )
        
        self.model = DistilBertModel.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_path
        )

        self.model.eval()
    
    def get_token_embeddings(self, text: str) -> List[Dict[str, Any]]:
        """
        입력 텍스트를 토큰 단위로 분해하고 각 토큰의 임베딩을 반환
        """

        # 1. 토크나이징 (uncased로 자동 소문자 변환)
        encoded = self.tokenizer(
            text, 
            return_tensors='pt',
            add_special_tokens=True,  # [CLS], [SEP] 토큰 추가
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )

        # 2. 토큰 ID를 실제 토큰으로 변환
        input_ids = encoded['input_ids'][0]  # 배치 차원 제거
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # 3. 모델 추론 (임베딩 추출)
        with torch.no_grad():
            outputs = self.model(**encoded)
            # last_hidden_state: [batch_size, sequence_length, hidden_size]
            token_embeddings = outputs.last_hidden_state[0]  # 첫 번째 배치 선택

        # 4. 토큰과 임베딩을 함께 반환
        token_results = []
        for i, (token, embedding) in enumerate(zip(tokens, token_embeddings)):
            token_results.append({
                'position': i,
                'token': token,
                'embedding': embedding.cpu().numpy(),  # tensor를 numpy로 변환
                'embedding_shape': embedding.shape
            })
        
        return token_results

    def get_combined_embeddings(self, text: str) -> List[Dict[str, Any]]:
        """
        텍스트의 토큰 임베딩을 추출하고 서브워드를 결합하여 반환
        """
        token_results = self.get_token_embeddings(text)
        return self.combine_subword_embeddings(token_results)

    @staticmethod
    def combine_subword_embeddings(token_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        서브워드 토큰들을 원래 단어로 합치고 그 임베딩을 평균합니다.
        예: 'cr', '##inge' -> 'cringe' (평균 임베딩)
        """
        combined_results = []
        current_word_tokens = []
        current_word_embeddings_list = []

        for item in token_results:
            token = item['token']
            embedding = item['embedding']

            if token.startswith('##'):
                # 이전 토큰의 서브워드인 경우
                current_word_tokens.append(token[2:])  # '##' 제거 후 추가
                current_word_embeddings_list.append(embedding)
            else:
                # 새로운 단어의 시작 또는 특수 토큰인 경우
                if current_word_tokens:
                    # 이전에 누적된 서브워드가 있다면 합치고 임베딩 평균 계산
                    combined_token = "".join(current_word_tokens)
                    averaged_embedding = np.mean(current_word_embeddings_list, axis=0)
                    combined_results.append({
                        'token': combined_token,
                        'embedding': averaged_embedding,
                        'embedding_shape': averaged_embedding.shape  # 평균 후 shape
                    })
                    current_word_tokens = []
                    current_word_embeddings_list = []
                
                # 현재 토큰 처리
                if token in ['[CLS]', '[SEP]']:
                    combined_results.append({
                        'token': token,
                        'embedding': embedding,
                        'embedding_shape': embedding.shape
                    })
                else:
                    # 새로운 단어 (또는 서브워드의 첫 부분) 누적 시작
                    current_word_tokens.append(token)
                    current_word_embeddings_list.append(embedding)
        
        # 루프 종료 후 남아있는 서브워드 처리 (예: 문장 끝이 서브워드로 끝나는 경우)
        if current_word_tokens:
            combined_token = "".join(current_word_tokens)
            averaged_embedding = np.mean(current_word_embeddings_list, axis=0)
            combined_results.append({
                'token': combined_token,
                'embedding': averaged_embedding,
                'embedding_shape': averaged_embedding.shape
            })
        
        # 최종 결과에 position 재할당
        for i, item in enumerate(combined_results):
            item['position'] = i

        return combined_results

    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """
        문장 전체의 임베딩을 반환 ([CLS] 토큰 사용)
        """
        token_results = self.get_token_embeddings(text)
        # [CLS] 토큰의 임베딩을 문장 임베딩으로 사용
        for token_info in token_results:
            if token_info['token'] == '[CLS]':
                return token_info['embedding']
        
        # [CLS] 토큰을 찾지 못한 경우 첫 번째 토큰 반환
        return token_results[0]['embedding'] if token_results else None

    def update_at_wordtrie(self, token_result:list):
        # 특수 토큰들 필터링
        special_tokens = {'[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'}

        for result in token_result:
            token = result['token']
            # 특수 토큰은 제외하고 업데이트
            if token not in special_tokens:
                self.docs._word_service.update_word_bert_embedding(
                    word_content=token,
                    embedding=result['embedding']
                    )
        

    def train(self):
        # 여기서 독스에서 문장을 받아와서 한문장씩 get_sentence_embedding 실시
        # 실시한 결과를 받아와서 독스의 트라이에 업데이트 진행
        # 업데이트 로직은 가중치 연산을 수행함
        sentences = self.docs.Sentence_list
        count = 0
        for sentence in sentences:
            count += 1
            embed = self.get_sentence_embedding(sentence)
            token_results = self.combine_subword_embeddings(embed)
            self.update_at_wordtrie(token_results)

            if count % 10 == 0:
                print(f"BERT Embedding {count} completed")

        return None


if __name__ == "__main__":
    import pandas as pd

    print("=== BERT Service Test ===")

    # 실제 데이터 파일에서 테스트 문서 로드 (DBERT_test.py 참고)
    try:
        # CSV 파일 경로 설정 (현재 파일 기준으로 상대 경로)
        csv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'kaggle_RC_2019-05.csv')
        df = pd.read_csv(csv_path)
        test_documents = df['body'][:15].tolist()  # 처음 15개 문서 사용
        print(f"✓ CSV 파일에서 테스트 문서 {len(test_documents)}개 로드 완료")
    except Exception as e:
        print(f"CSV 파일 로드 실패: {e}")
        # 대안으로 더 많은 테스트 문서 생성
        test_documents = [
            "This is a simple test sentence.",
            "BERT embeddings are very powerful for NLP tasks.",
            "The preprocessing step is crucial for good results.",
            "Natural language processing has many applications.",
            "Machine learning models need training data.",
            "Deep learning transforms how we process text.",
            "Tokenization is an important preprocessing step.",
            "Word embeddings capture semantic meaning.",
            "Neural networks can learn complex patterns.",
            "Text classification is a common NLP task.",
            "Sentiment analysis helps understand opinions.",
            "Language models are trained on large corpora."
        ]
        print(f"대안 테스트 문서 {len(test_documents)}개 준비")

    # DocumentService 초기화
    try:
        doc_service = DocumentService()
        print("✓ DocumentService 초기화 완료")

        # 문장 처리
        doc_service.create_sentence_list(documents=test_documents)
        print(f"✓ 문장 처리 완료 - {doc_service.get_sentence_count()}개 문장")

        # BERTService 초기화
        bert_service = BERTService(doc_service)
        print("✓ BERTService 초기화 완료")

        # 테스트 1: 토큰 임베딩 추출
        test_text = "This is a preprocessing test."
        print(f"\n=== 테스트 1: 토큰 임베딩 추출 ===")
        print(f"입력 텍스트: {test_text}")

        token_results = bert_service.get_token_embeddings(test_text)
        print(f"토큰 개수: {len(token_results)}")

        print("\n토큰별 정보:")
        print("-" * 60)
        print(f"{'위치':<4} {'토큰':<15} {'임베딩 형태':<15} {'샘플'}")
        print("-" * 60)

        for result in token_results[:5]:  # 처음 5개만 출력
            embedding_sample = f"[{result['embedding'][0]:.3f}, {result['embedding'][1]:.3f}, ...]"
            print(f"{result['position']:<4} {result['token']:<15} {str(result['embedding_shape']):<15} {embedding_sample}")

        # 테스트 2: 서브워드 결합 임베딩
        print(f"\n=== 테스트 2: 서브워드 결합 임베딩 ===")
        combined_results = bert_service.get_combined_embeddings(test_text)
        print(f"결합된 토큰 개수: {len(combined_results)}")

        print("\n결합된 토큰별 정보:")
        print("-" * 60)
        print(f"{'위치':<4} {'토큰':<15} {'임베딩 형태':<15} {'샘플'}")
        print("-" * 60)

        for result in combined_results:
            embedding_sample = f"[{result['embedding'][0]:.3f}, {result['embedding'][1]:.3f}, ...]"
            print(f"{result['position']:<4} {result['token']:<15} {str(result['embedding_shape']):<15} {embedding_sample}")

        # 테스트 3: 문장 임베딩
        print(f"\n=== 테스트 3: 문장 임베딩 ===")
        sentence_embedding = bert_service.get_sentence_embedding(test_text)
        print(f"문장 임베딩 형태: {sentence_embedding.shape}")
        print(f"문장 임베딩 샘플: [{sentence_embedding[0]:.3f}, {sentence_embedding[1]:.3f}, {sentence_embedding[2]:.3f}, ...]")

        # 테스트 4: 워드트라이 업데이트 테스트
        print(f"\n=== 테스트 4: 워드트라이 업데이트 테스트 ===")
        words_before = len(doc_service.words_list) if doc_service.words_list else 0
        print("업데이트 전 단어 개수:", words_before)

        # 토큰 결과로 테스트 (특수 토큰은 메서드에서 자동 필터링)
        test_token_results = combined_results[:5]  # 처음 5개 토큰 사용

        print(f"테스트할 토큰들: {[r['token'] for r in test_token_results]}")
        bert_service.update_at_wordtrie(test_token_results)

        words_after = len(doc_service.words_list) if doc_service.words_list else 0
        print("업데이트 후 단어 개수:", words_after)
        print("✓ 워드트라이 업데이트 완료")

        print(f"\n=== 모든 테스트 완료 ===")
        print("✓ BERTService가 정상적으로 작동합니다!")

    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()