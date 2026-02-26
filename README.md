# 📄 JIHA PDF 요약 AI 웹사이트 (PDF Summarizer)

본 프로젝트는 사용자가 업로드한 PDF 파일을 AI가 읽고 핵심 내용을 3~5문장으로 요약해 주는 지능형 웹 애플리케이션입니다. 
**LangChain**과 **OpenAI LLM(gpt-4o-mini)**, 그리고 빠르고 간편한 UI 구성을 위해 **Streamlit**을 사용하여 구현되었습니다.

---

## 🛠️ 구현 기술 (Tech Stack)

* **프레임워크 (Frontend / UI)**
  * `Streamlit`: 파이썬 스크립트 하나로 반응형 웹사이트 UI를 빠르게 구축
* **AI & LLM (Backend)**
  * `LangChain`: 다양한 LLM 도구들과 문서를 엮어주는 핵심 프레임워크
  * `OpenAI API`: `gpt-4o-mini` 모델을 사용한 텍스트 퀄리티 높은 요약 생성
* **문서 처리 및 데이터 (Data Processing & DB)**
  * `PyPDF2`: PDF 파일 내 텍스트 추출 모듈
  * `OpenAIEmbeddings` (`text-embedding-3-small`): 텍스트를 AI가 이해할 수 있는 숫자 벡터로 변환 (임베딩)
  * `FAISS` (Facebook AI Similarity Search): 생성된 임베딩 벡터를 고속으로 검색할 수 있는 로컬 벡터 저장소(Vector DB)

---

## 📂 소스코드 구조 (Repository Structure)

```text
📦JIHA_PDFwebsite_DeploymentCheck
 ┣ 📜 5_3_PDF_요약_웹사이트_만들기.py   # 🌟 메인 애플리케이션 코드 (UI 기반 및 AI 핵심 로직 구현)
 ┣ 📜 requirements.txt                   # 클라우드 배포 및 로컬 구동을 위한 파이썬 패키지 의존성 명세 서
 ┗ 📜 README.md                          # 프로젝트 설명 문서 (현재 위치)
```

---

## 💻 핵심 기술 코드 예시 (Core Code Snippets)

이 애플리케이션은 **RAG (Retrieval-Augmented Generation, 검색 증강 생성)** 방식을 기반으로 동작하며, 크게 세 단계로 나뉩니다.

### 1. 전처리 과정: 텍스트 분할 및 벡터 DB(FAISS) 임베딩
PDF에서 추출한 긴 텍스트를 한 번에 LLM에 넣지 않고, 의미 단위(`chunk`)로 쪼갠 후 벡터 DB에 저장하여 검색 효율을 극대화합니다.

```python
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def process_text(text: str):
    # 1) 문서를 1000자 단위로 자르되, 중간에 내용이 끊기지 않도록 200자씩 겹치게 설정
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    # 2) 쪼개진 조각들을 OpenAI 임베딩 모델을 통해 벡터화 시키고 FAISS DB에 저장
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=API_KEY
    )
    db = FAISS.from_texts(chunks, embeddings)
    
    return db
```

### 2. 문서 검색: 사용자의 목적에 맞는 문서 유사도 추출
AI 모델에게 지시할 목적(프롬프트)에 맞춰, 방금 만든 벡터 DB 안에서 관련성(유사도)이 가장 높은 4개의 문서 조각만 재빨리 찾아옵니다.

```python
# 'query'는 시스템에서 프롬프트로 지정해 둔 요약 목적 문구입니다.
query = "업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요."

# FAISS DB에서 사용자의 목적(query)과 가장 유사도 높은 문서 k=4 개를 추출함
docs = db.similarity_search(query, k=4)
```

### 3. 응답 생성: LLM을 이용한 QA 체인
찾아온 핵심 조각들(`docs`)을 OpenAI 언어모델의 컨텍스트(Context)로 전달하여 요약문을 생성합니다. 비용(Token) 관리 기능이 함께 작동합니다.

```python
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback

# LLM 셋팅 (gpt-4o-mini 모델 적용, 답변 일관성을 위해 온도를 0.1로 낮춤)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=API_KEY,
    temperature=0.1
)

# QA 체인 로드 (검색된 모든 문서를 context 로 밀어넣는 "stuff" 방식)
chain = load_qa_chain(llm, chain_type="stuff")

# 토큰 비용 추적과 함께 요청 실행 (Streamlit UI에서 바로 확인 가능)
with get_openai_callback() as cost:
    response = chain.run(input_documents=docs, question=query)
    
st.write(response)    # Streamlit을 통해 요약 결과 출력
st.caption(str(cost)) # 사용된 API 비용 하단 노출
```

---

## 🚀 로컬 실행 방법 (How to run locally)

1. 리포지토리를 원격에서 클론합니다.
2. `.env` 파일을 프로젝트 루트 공간에 생성하고, 아래와 같이 OpenAI API 키를 입력합니다.
   ```text
   OPENAI_API_KEY="sk-당신의-오픈API-키"
   ```
3. 터미널을 열고 애플리케이션 구동에 필요한 패키지를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
4. Streamlit 서버 구동 명령어를 입력합니다.
   ```bash
   streamlit run 5_3_PDF_요약_웹사이트_만들기.py
   ```
5. 연결된 기본 브라우저에서 `http://localhost:8501` 창으로 접속되면 구동 성공!
