import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import pandas as pd

from Preprocess import *

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

# 예시 사용
if __name__ == "__main__":
    # 테스트 문장들
    df = pd.read_csv("../kaggle_RC_2019-05.csv")
    text = df['body'][:15]
    
    print(text)
    docs = Docs()
    docs.rawdata = text
    # for sentence in docs.sentence_list:
    #     print(sentence._lemmatised)
        
    
    for i,sentence in enumerate(text):
        print(f"\n{'='*50}")
        print(f"원본 문장: {sentence}")
        print(f"독스 문장: {docs.sentence_list[i]._lemmatised}")
        print(f"{'='*50}")
        
        # 토큰 임베딩 추출
        token_results = get_token_embeddings(sentence)
        
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

## https://claude.ai/chat/259a318c-4710-44bf-86e7-ecd3e2f7605a