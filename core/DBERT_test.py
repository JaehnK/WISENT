import contractions
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from services import DocumentService
from entities import Documents
# from preprocess import *

# 원하는 경로 지정
cache_path = "./DistillBERT/model"

model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(
    model_name, 
    cache_dir=cache_path
)
model = DistilBertModel.from_pretrained(
    model_name, 
    cache_dir=cache_path
)


# # 모델과 토크나이저 로드
# model_name = 'distilbert-base-uncased'
# tokenizer = DistilBertTokenizer.from_pretrained(model_name)
# model = DistilBertModel.from_pretrained(model_name)

# 평가 모드로 설정 (추론 시)
model.eval()

def get_token_embeddings(text):
    """
    입력 텍스트를 토큰 단위로 분해하고 각 토큰의 임베딩을 반환
    """
    
    # 1. 토크나이징 (uncased로 자동 소문자 변환)
    encoded = tokenizer(
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
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # 3. 모델 추론 (임베딩 추출)
    with torch.no_grad():
        outputs = model(**encoded)
        # last_hidden_state: [batch_size, sequence_length, hidden_size]
        token_embeddings = outputs.last_hidden_state[0]  # 첫 번째 배치 선택
    
    # 4. 결과 정리
    results = []
    for i, (token, embedding) in enumerate(zip(tokens, token_embeddings)):
        results.append({
            'position': i,
            'token': token,
            'embedding': embedding.numpy(),  # 768차원 벡터
            'embedding_shape': embedding.shape
        })
    
    return results

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
            current_word_tokens.append(token[2:]) # '##' 제거 후 추가
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
                    'embedding_shape': averaged_embedding.shape # 평균 후 shape
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

# 예시 사용
if __name__ == "__main__":
    # 테스트 문장들
    df = pd.read_csv("../kaggle_RC_2019-05.csv")
    text = df['body'][:15]
    
    print(text)
    # docs = Documents()
    # docs.rawdata = text
    # for sentence in docs.sentence_list:
    #     print(sentence._lemmatised)
        
    doc_serv = DocumentService()
    doc_serv.create_sentence_list(documents=text)

    for i,sentence in enumerate(text):
        print(f"\n{'='*50}")
        print(f"원본 문장: {sentence}")
        lemmatised_sentence = " ".join(doc_serv.sentence_list[i]._lemmatised)
        print(f"독스 문장: {lemmatised_sentence}")
        print(f"{'='*50}")
        
        # 토큰 임베딩 추출
        raw_token_results = get_token_embeddings(lemmatised_sentence)
        # 서브워드 합치기 및 임베딩 평균
        token_results = combine_subword_embeddings(raw_token_results)
        
        # 결과 출력
        print(f"총 토큰 수: {len(token_results)}")
        print("\n토큰별 정보:")
        print("-" * 70)
        print(f"{'위치':<4} {'토큰':<15} {'임베딩 형태':<15} {'임베딩 요약'}")
        print("-" * 70)
        
        for result in token_results:
            # 임베딩 벡터의 처음 3개 값만 표시
            embedding_summary = f"[{result['embedding'][0]:.3f}, {result['embedding'][1]:.3f}, {result['embedding'][2]:.3f}, ...]"
            print(f"{result['position']:<4} {result['token']:<15} {str(result['embedding_shape']):<15} {embedding_summary}")


## 자 재훈아 BERT의 입력값은 Contractions를 한번 진행한 값을 짚어넣자.
## 근데 이건 sentence에서 전처리할 떄 저장하고 있으면 일을 두번하지 않겠지?
## 그리고 아웃풋 값에서 특수문자와 구둣점들의 값은 다 삭제해
## 그리고 ##으로 시작하거나 끝난 것들은 합쳐
## Word에서는 가중 합으로 처리해 일단 다 더하고 마지막에 나눠

