'''import os
import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.schema import Document

# Directory where ChromaDB will store vectors
PERSIST_DIR = "chroma_store"

# Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Create embeddings and store in ChromaDB
def process_pdf(pdf_path: str):
    text = extract_text_from_pdf(pdf_path)

    # Split text into chunks
    chunk_size = 500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    docs = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    return vectordb

# Ask question using Ollama + Chroma
def ask_question(question: str):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(question)

    # Combine retrieved text for LLM
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    llm = OllamaLLM(model="qwen:0.5b")  # No API key needed

    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"
    answer = llm.invoke(prompt)

    return answer'''

'''import os
from typing import Iterable, List
import pdfplumber

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_ROOT = os.path.join(BASE_DIR, "chroma_store")

# --- Models ---
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast, good quality
LLM_MODEL = "qwen:0.5b"                                      # Ollama local

# --- Prompt templates ---
QA_PROMPT = """
You are a helpful assistant.
Answer the question using ONLY the context. If the answer is not present, say: "Answer not found in the document."

Context:
{context}

Question:
{question}

Answer:
""".strip()

REASONING_PROMPT = """
You are a careful assistant. Use ONLY the context to reason step by step.
If the answer is not in the context, say: "Answer not found in the document."

Context:
{context}

Question:
{question}

Step-by-step reasoning and final answer:
""".strip()

TRANSLATE_PROMPT = """
You are a multilingual assistant.
Answer the question using ONLY the context, and respond in {language}.
If the answer is not in the context, say it is not found.

Context:
{context}

Question:
{question}

Answer in {language}:
""".strip()

SUMMARIZE_PROMPT = """
You are a summarizer. Summarize the context in 5 concise bullet points.

Context:
{context}

Summary:
""".strip()


def _emb():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def _splitter():
    # Bigger chunk size for fewer LLM calls; overlap keeps coherence
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def stream_pdf_text(pdf_path: str) -> Iterable[str]:
    """
    Stream text page-by-page to avoid loading entire PDFs (10k pages) into memory.
    Yields raw text for each page.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                yield text


def chunk_pages_to_docs(pages: Iterable[str]) -> Iterable[Document]:
    """
    Convert streamed pages into chunked Documents using the splitter.
    Processes in small buffers to control memory usage.
    """
    splitter = _splitter()
    buffer: List[str] = []
    buf_chars = 0
    MAX_BUF_CHARS = 60_000  # ~50 pages; tune for memory

    for page_text in pages:
        buffer.append(page_text)
        buf_chars += len(page_text)
        if buf_chars >= MAX_BUF_CHARS:
            text = "\n\n".join(buffer)
            for i, chunk in enumerate(splitter.split_text(text)):
                yield Document(page_content=chunk)
            buffer.clear()
            buf_chars = 0

    if buffer:
        text = "\n\n".join(buffer)
        for i, chunk in enumerate(splitter.split_text(text)):
            yield Document(page_content=chunk)


def ensure_collection(collection_name: str) -> Chroma:
    """
    Open or create a Chroma collection. No-op if exists.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=_emb(),
        persist_directory=CHROMA_ROOT,
    )


def index_pdf(pdf_path: str, collection_name: str, batch_size: int = 64) -> None:
    """
    Incrementally index a large PDF into Chroma (batched writes).
    Safe for 1–10,000 pages.
    """
    os.makedirs(CHROMA_ROOT, exist_ok=True)
    vectordb = ensure_collection(collection_name)

    batch: List[Document] = []
    for doc in chunk_pages_to_docs(stream_pdf_text(pdf_path)):
        batch.append(doc)
        if len(batch) >= batch_size:
            vectordb.add_documents(batch)
            batch.clear()
    if batch:
        vectordb.add_documents(batch)
    # Chroma persists automatically via persist_directory.


def _build_prompt(mode: str, question: str, context: str, language: str = "English") -> str:
    m = (mode or "qa").lower()
    if m == "reasoning":
        return REASONING_PROMPT.format(context=context, question=question)
    if m == "translate":
        return TRANSLATE_PROMPT.format(context=context, question=question, language=language)
    if m == "summary":
        return SUMMARIZE_PROMPT.format(context=context)
    return QA_PROMPT.format(context=context, question=question)


def ask_question(
    question: str,
    collection_name: str,
    mode: str = "qa",
    language: str = "English",
    k: int = 5,
) -> str:
    """
    RAG query: retrieve top-k chunks then query Ollama.
    """
    if not question or not question.strip():
        return "Please provide a question."

    vectordb = ensure_collection(collection_name)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "No relevant context found in this document."

    context = "\n\n".join(d.page_content for d in docs)
    prompt = _build_prompt(mode, question, context, language)

    llm = OllamaLLM(model=LLM_MODEL)
    result = llm.invoke(prompt)

    if isinstance(result, str):
        return result.strip()
    return getattr(result, "content", str(result)).strip()
'''

import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langdetect import detect
from deep_translator import GoogleTranslator

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_ROOT = os.path.join(BASE_DIR, "chroma_store")
os.makedirs(CHROMA_ROOT, exist_ok=True)

# ---------------------------
# Models
# ---------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:1.5b"  # Ollama local model

# ---------------------------
# PDF Text Extraction
# ---------------------------
def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()

# ---------------------------
# Build Chroma store
# ---------------------------
def process_pdf_to_chroma(pdf_path: str, collection_name: str) -> Chroma:
    raw_text = extract_text_from_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_ROOT
    )
    vector_db.persist()
    return vector_db

# ---------------------------
# Ask question (RAG)
# ---------------------------
def ask_question(vector_db: Chroma, question: str, k: int = 5) -> str:
    if not question.strip():
        return "Please provide a question."
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "No relevant context found in this document."
    context = "\n\n".join([d.page_content for d in docs])
    llm = OllamaLLM(model=LLM_MODEL)
    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context. If the answer is not present, say: "Answer not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""
    result = llm(prompt)
    return result.strip() if isinstance(result, str) else getattr(result, "content", str(result)).strip()

# ---------------------------
# Multilingual QA
# ---------------------------
def ask_question_multilingual(vector_db: Chroma, user_question: str, target_lang: str = "en") -> str:
    try:
        src_lang = detect(user_question)
    except:
        src_lang = "en"
    q_en = GoogleTranslator(source=src_lang, target="en").translate(user_question) if src_lang != "en" else user_question
    answer_en = ask_question(vector_db, q_en)
    return GoogleTranslator(source="en", target=target_lang).translate(answer_en) if target_lang != "en" else answer_en
