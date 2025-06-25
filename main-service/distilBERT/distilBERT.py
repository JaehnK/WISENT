
import contractions
import numpy as np

import torch
from transformers import DistilBertTokenizer, DistilBertModel

class DistilBERT:
    class _TokenResult:
        __slots__ = ['position', 'token', 'embedding']
        
        def __init__(self):
            self.position = -1
            self.token = ""
            self.embedding = None
        
        def clear(self):
            self.position = -1
            self.token = ""
            self.embedding = None
    
    def __init__(self, cache_path: str = "./model", pool_size: int = 2000):
        self.cache_path:str = cache_path
        self.model_name:str = "distilbert-base-uncased"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_tokenizer()
        self._load_model()
        
        self.inference_count = 0
        self.pool_size = pool_size 
        self._initalise_result_pool()

    def _load_tokenizer(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(
                            self.model_name,
                            cache_dir = self.cache_path)

    def _load_model(self):
        self.model = DistilBertModel.from_pretrained(
            self.model_name,
            cache_dir = self.cache_path
        )
        self.model.to(self.device)
        self.model.eval()

    def _initalise_result_pool(self):
        self.result_pool = []
        for _ in range(self.prealloced_count):
            self.result_pool.append(self._TokenResult())

    def _get_result_dict(self):
        """
            메모리 풀 관리용 내부 함수
        Returns:
            self._TokenResult: TokenResult를 리턴합니다. 없으면 새로 할당합니다.
        """
        if self._result_pool:
            result = self._result_pool.pop()
            result.clear()
            return (result)
        return (self._TokenResult())
    
    def _return_result_dict(self, result_dict):
        """
            메모리 풀 관리용 내부함수
        Args:
            result_dict (self._TokenResult): 사용한 TokenResult를 반환합니다.
        """
        if len(self._result_pool) < self._pool_size:
            self._result_pool.append(result_dict)

    def clear_cache(self):
        """캐시 정리"""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self._result_pool.clear()

    @torch.inference_mode()
    def inference(self, text:str):
        """
            DistilBERT를 인퍼런스 합니다.
            inference 모드를 기반으로 작동합니다
            Sentence 객체의 전처리한 모듈과 일치하도록 토큰을 aggregate한 이후 리턴합니다.
            1000번에 한번씩 캐시를 정리합니다.
        Args:
            text (str): 인퍼런스 하고자 하는 텍스트

        Returns:
            List: 인퍼런스가 완료된 각 단어의 임베딩 값
        """
        if text is None:
            raise ValueError("Input String Needed")
        else:
            txt = contractions.fix(text)
        
        encoded = self.tokenizer(
            txt, 
            return_tensors='pt',
            add_special_tokens=True,  # [CLS], [SEP] 토큰 추가
            padding=False,
            truncation=True,
            max_length=512,
            return_attention_mask=True
            )
        
        input_ids = encoded['input_ids'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        attention_mask = encoded['attention_mask'].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        token_embeddings = outputs.last_hidden_state[0].cpu().numpy()
        
        results = []
        for i, (token, embedding) in enumerate(zip(tokens, token_embeddings)):
            result = self._get_result_dict()
            result.position = i
            result.token = token
            result.embedding = embedding
            results.append(result)
        
        try:
            aggregated_results = self._aggregate_tokens(results)
        finally:
            # 사용한 객체들을 풀에 반환 (finally로 보장)
            for result in results:
                self._return_result_object(result)
            
        self.inference_count += 1
        
        if self.inference_count % 1000 == 0:
            self.clear_cache()
        
        return (aggregated_results)

    def _aggregate_tokens(self, results:_TokenResult):
        if not results:
            raise ValueError("DistilBert->AggregateTokens: Results Not Defined")
        
        aggregated = []
        for result in results:
            tkn = result.token
        
            if tkn in ("[CLS]", "[SEP]"):
                continue
            
            if not any(char.isalnum() for char in tkn):
                continue
                
            # 서브워드 토큰 처리
            if tkn.startswith("##") and aggregated:
                last_token = aggregated[-1]
                last_token['token'] += tkn[2:]
                
                # 서브토큰 갯수 추적
                if 'subtoken_count' not in last_token:
                    last_token['subtoken_count'] = 1
                last_token['subtoken_count'] += 1
                
                # 누적 평균
                count = last_token['subtoken_count']
                last_token['embedding'] = (
                    last_token['embedding'] * (count - 1) + result.embedding
                ) / count
            else:
                aggregated.append({
                'position': result.position,
                'token': tkn,
                'embedding': result.embedding.copy(),  # 복사본 생성
                'subtoken_count': 1
            })
        
        for token_info in aggregated:
            token_info.pop('subtoken_count', None)
        
        return (aggregated)