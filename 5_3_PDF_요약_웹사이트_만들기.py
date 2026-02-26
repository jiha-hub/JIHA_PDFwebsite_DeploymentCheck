import os
from dotenv import load_dotenv

from PyPDF2 import PdfReader
import streamlit as st

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


# =========================
# 0) 환경변수(.env) 로드
# =========================
# 같은 폴더에 .env 파일을 만들고 아래처럼 넣어두세요:
# OPENAI_API_KEY=sk-....
load_dotenv()

# Streamlit 배포 환경(st.secrets)과 로컬 환경(.env) 모두 지원
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    try:
        API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass


def process_text(text: str):
    """PDF 텍스트를 청크로 나누고 임베딩 후 FAISS 벡터DB 생성"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=API_KEY
    )
    db = FAISS.from_texts(chunks, embeddings)
    return db


def main():
    st.set_page_config(page_title="PDF 요약하기", page_icon="📄")
    st.title("📄 PDF 요약하기")
    st.divider()

    if not API_KEY or not API_KEY.startswith("sk-"):
        st.error("OPENAI_API_KEY를 찾을 수 없어요. .env 파일에 OPENAI_API_KEY=sk-... 를 설정해 주세요.")
        st.stop()

    pdf = st.file_uploader("PDF 파일을 업로드해주세요", type="pdf")

    if pdf is not None:
        # PDF 텍스트 추출
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        if not text.strip():
            st.warning("PDF에서 텍스트를 추출하지 못했어요. (스캔본/이미지 PDF일 수 있어요)")
            st.stop()

        # 벡터DB 생성
        with st.spinner("PDF를 읽고 임베딩 중..."):
            db = process_text(text)

        query = "업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요."

        # 유사 문서 검색
        docs = db.similarity_search(query, k=4)

        # LLM + QA 체인
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=API_KEY,
            temperature=0.1
        )
        chain = load_qa_chain(llm, chain_type="stuff")

        with st.spinner("요약 생성 중..."):
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                st.caption(str(cost))

        st.subheader("요약 결과")
        st.write(response)


if __name__ == "__main__":
    main()
