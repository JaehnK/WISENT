import torch
import numpy as np
from typing import List, Dict, Any
from transformers import DistilBertTokenizer, DistilBertModel

from core.services.Document import DocumentService


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

    def update_at_wordtrie(embed: np.ndarray):
        self.

    def train(self):
        # 여기서 독스에서 문장을 받아와서 한문장씩 get_sentence_embedding 실시
        # 실시한 결과를 받아와서 독스의 트라이에 업데이트 진행
        # 업데이트 로직은 가중치 연산을 수행함
        sentences = self.docs.Sentence_list
        count = 0
        for sentence in sentences:
            count += 1
            embed = self.get_sentence_embedding(sentence)
            self.update_at_wordtrie(embed)

            if count % 10 == 0:
                print(f"BERT Embedding {} completed")

        return None