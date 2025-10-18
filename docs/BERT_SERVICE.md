## BertService 사용 가이드 (DistilBERT 기반 임베딩 서비스)

본 문서는 `core/services/DBert/BertService.py`에서 제공하는 DistilBERT 기반 임베딩 추출 및 학습 파이프라인에 대해 설명합니다. 토큰/단어/문장 단위 임베딩을 생성하고, `DocumentService`의 단어 트라이(WordTrie)에 임베딩을 업데이트하는 과정을 다룹니다.

---

### 개요
- **역할**: 텍스트 입력에 대해 DistilBERT(`distilbert-base-uncased`)로부터 임베딩을 추출하고, 토큰 서브워드를 단어 단위로 결합한 임베딩을 제공. 또한 코퍼스(문장 리스트)에 대해 반복적으로 임베딩을 계산하여 단어 트라이에 누적 업데이트.
- **특징**:
  - Hugging Face Transformers의 `DistilBertTokenizer`, `DistilBertModel` 사용
  - 최대 시퀀스 길이 512, 자동 소문자화(uncased)
  - `[CLS]` 임베딩을 문장 임베딩으로 사용
  - 서브워드 토큰(`##`)을 평균으로 결합하여 단어 임베딩 구성
  - 특수 토큰(`[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]`)은 WordTrie 업데이트에서 제외

---

### 의존성 및 환경
- Python 패키지: `torch`, `transformers`, `numpy`
- 모델/토크나이저: `distilbert-base-uncased`
- 캐시 디렉터리: `model_path` 인자(기본 `./DistillBERT/model`)가 `transformers`의 `cache_dir`로 사용됨
- 첫 실행 시 모델/토크나이저가 다운로드되어 `cache_dir`에 캐시됨

GPU 사용을 원할 경우 모델/텐서를 명시적으로 디바이스로 이동하는 보일러플레이트를 추가할 수 있습니다(기본 구현은 CPU에서 동작). 예시:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# encoded = tokenizer(...);
# outputs = model(**{k: v.to(device) for k, v in encoded.items()})
```

---

### 초기화
```python
from core.services.DBert.BertService import BertService
from core.services.Document.DocumentService import DocumentService

docs = DocumentService(...)
service = BertService(docs=docs, model_path="./DistillBERT/model")
```

- **docs**: `DocumentService` 인스턴스. 내부에서 `docs._word_service.update_word_bert_embedding(word_content, embedding)`를 호출하므로 WordTrie 업데이트 API를 제공해야 합니다.
- **model_path**: 모델/토크나이저 캐시 디렉터리로 사용됩니다.

---

### 주요 메소드 요약

- `get_token_embeddings(text: str) -> List[Dict[str, Any]]`
  - 입력 문장을 토크나이즈 하여 각 토큰의 임베딩을 반환합니다.
  - 반환 항목: `position`, `token`, `embedding (np.ndarray)`, `embedding_shape`.

- `combine_subword_embeddings(token_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
  - `##` 접두 서브워드들을 평균하여 원 단어 단위 임베딩으로 결합합니다.
  - 특수 토큰(`[CLS]`, `[SEP]` 등)은 그대로 통과합니다.
  - 반환 항목: `token`, `embedding (np.ndarray)`, `embedding_shape`, `position`(재할당됨).

- `get_combined_embeddings(text: str) -> List[Dict[str, Any]]`
  - `get_token_embeddings` + `combine_subword_embeddings`를 연쇄 호출하여 단어 단위 임베딩을 바로 반환합니다.

- `get_sentence_embedding(text: str) -> np.ndarray`
  - `[CLS]` 토큰의 임베딩을 문장 임베딩으로 반환합니다. 없을 경우 첫 토큰 임베딩을 fallback으로 사용합니다.

- `get_word_embedding(word: str) -> np.ndarray`
  - 입력 단어에 대해 첫 번째 비-특수 토큰의 결합 임베딩을 반환합니다.
  - 해당 토큰이 없으면 `[CLS]` 또는 영벡터(크기 768)를 반환합니다.

- `update_at_wordtrie(token_result: list) -> None`
  - 특수 토큰을 제외한 각 토큰에 대해 WordTrie에 BERT 임베딩을 업데이트합니다.

- `train() -> None`
  - `docs.Sentence_list`의 문장을 순회하며 임베딩 계산 후 WordTrie를 반복 업데이트합니다.
  - 10문장마다 진행 상황을 로그로 출력합니다.

---

### 반환 형식 및 차원
- DistilBERT `last_hidden_state`의 히든 크기는 768입니다.
- 토큰/단어 임베딩: `np.ndarray` (형상 대개 `(768,)`)가 각 항목의 `embedding`에 저장됩니다.
- 문장 임베딩(`[CLS]`): `np.ndarray` (형상 `(768,)`).

---

### 사용 예시

토큰 단위 임베딩:
```python
text = "Cringe content goes viral."
token_results = service.get_token_embeddings(text)
for r in token_results:
    print(r["position"], r["token"], r["embedding"].shape)
```

서브워드 결합(단어 단위 임베딩):
```python
combined = service.get_combined_embeddings(text)
for r in combined:
    print(r["position"], r["token"], r["embedding"].shape)
```

문장 임베딩(`[CLS]`):
```python
sent_vec = service.get_sentence_embedding(text)
print(sent_vec.shape)  # (768,)
```

단일 단어 임베딩:
```python
word_vec = service.get_word_embedding("cringe")
print(word_vec.shape)  # (768,)
```

WordTrie 업데이트(내부 사용):
```python
combined = service.get_combined_embeddings(text)
service.update_at_wordtrie(combined)
```

코퍼스 학습 파이프라인:
```python
# docs.Sentence_list에 문장들이 채워져 있다고 가정
service.train()
```

---

### 설계 노트 및 주의사항
- **특수 토큰 필터링**: WordTrie 업데이트 시 `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]`는 제외됩니다.
- **서브워드 결합**: `##`로 시작하는 서브워드는 이전 토큰에 속하는 서브워드로 간주하여 평균 임베딩으로 결합됩니다.
- **시퀀스 길이**: 최대 512 토큰. 긴 문장은 자동으로 truncate됩니다.
- **정규화/후처리**: 기본 구현은 임베딩 정규화나 추가 투영을 수행하지 않습니다. 필요 시 다운스트림 단계에서 추가하세요.
- **성능**: 기본은 단건 추론 루프입니다. 코퍼스가 큰 경우 배치 처리, GPU 사용, 또는 캐싱 전략을 고려하세요.

---

### 알려진 이슈 및 권장 수정

`train()` 내부 구현에서 `combine_subword_embeddings`의 입력 타입이 맞지 않을 가능성이 있습니다. 현재 코드는 문장 임베딩(`np.ndarray`)을 결합 함수에 전달하고 있어 타입 불일치가 발생할 수 있습니다. 의도대로라면 문장 임베딩이 아니라 토큰/단어 리스트를 전달해야 합니다.

의도된 동작에 맞춘 예시 수정안:

```python
# 기존
embed = self.get_sentence_embedding(sentence)
token_results = self.combine_subword_embeddings(embed)
self.update_at_wordtrie(token_results)

# 권장
token_results = self.get_combined_embeddings(sentence)
self.update_at_wordtrie(token_results)
```

위와 같이 수정하면 각 문장의 단어 단위 임베딩을 WordTrie에 업데이트하게 되어, 서브워드 결합 로직과 타입이 일관됩니다.

---

### 통합 포인트(API 계약)
- `DocumentService`는 다음 속성과 메서드를 제공해야 합니다:
  - `Sentence_list: List[str]` — 학습에 사용할 문장들의 리스트
  - `_word_service.update_word_bert_embedding(word_content: str, embedding: np.ndarray)` — 단어별 임베딩 업데이트 API

---

### 트러블슈팅
- 모델/토크나이저 다운로드 실패: 네트워크/권한을 확인하고 `model_path`(cache_dir) 존재 여부를 확인하세요.
- 메모리 부족(OOM): 배치 크기를 줄이거나, 긴 입력을 슬라이딩 윈도우로 분할하세요.
- 속도 저하: GPU 사용, FP16 활용(가능 시), 문장 길이 제한, 배치 처리 도입을 고려하세요.

---

### 변경 이력
- v1.0: 초기 버전 문서화. DistilBERT 기반 임베딩 추출/학습 파이프라인 개요 및 사용법 정리.


