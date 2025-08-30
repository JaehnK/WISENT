FROM python:3.12-trixie

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# # spaCy 한국어 모델 다운로드 (필요하다면)
# RUN python -m spacy download ko_core_news_sm

# # NLTK 데이터 다운로드
# RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 소스 코드 복사
COPY core/ /app/core/

# 컨테이너 실행 시 실행할 명령어
CMD ["python", "main.py"]
