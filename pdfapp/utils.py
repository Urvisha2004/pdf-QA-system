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


'''ater delete...........
import os
import uuid
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_ollama import OllamaLLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_ROOT = os.path.join(BASE_DIR, "chroma_store")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(CHROMA_ROOT, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Models ----------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:1.5b"  # Ollama local model

# ---------------- PDF Extraction ----------------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    # Fallback OCR
    if not text.strip() and OCR_AVAILABLE:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path)
        for page_image in pages:
            text += pytesseract.image_to_string(page_image) + "\n"

    return text.strip()

# ---------------- Vector DB ----------------
def process_pdf_to_chroma(pdf_path: str, collection_name: str) -> Chroma:
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        raise ValueError("No text found in PDF")

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

# ---------------- Ask Question ----------------
def ask_question(vector_db: Chroma, question: str, k: int = 5) -> str:
    if not question.strip():
        return "Please provide a question."

    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "Answer not found in the document."

    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
Answer the question using ONLY the context below.
Context:
{context}

Question:
{question}

Answer:
"""

    # Try Ollama first
    try:
        llm = OllamaLLM(model=LLM_MODEL)
        result = llm(prompt)
        return result.strip() if isinstance(result, str) else getattr(result, "content", str(result)).strip()
    except Exception as e:
        print("Ollama error:", e)
        # Fallback to HuggingFace
        pipe = pipeline("text-generation", model="gpt2", max_new_tokens=200)
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        result = hf_llm(prompt)
        return result.strip() if isinstance(result, str) else str(result)

# ---------------- Multilingual QA ----------------
def ask_question_multilingual(vector_db: Chroma, user_question: str, target_lang: str = "en") -> str:
    try:
        src_lang = detect(user_question)
    except:
        src_lang = "en"

    # Translate question to English if needed
    q_en = GoogleTranslator(source=src_lang, target="en").translate(user_question) if src_lang != "en" else user_question

    # Get answer in English
    answer_en = ask_question(vector_db, q_en)

    # Translate answer back to target language
    if target_lang != "en":
        return GoogleTranslator(source="en", target=target_lang).translate(answer_en)
    return answer_en'''

import os
import uuid
import shutil
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
import logging
from .models import ChatHistory


try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_ROOT = os.path.join(BASE_DIR, "chroma_store")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(CHROMA_ROOT, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Models ----------------
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "qwen2.5:1.5b"  # Ollama local model

# Ollama LLM
ollama_llm = OllamaLLM(model=LLM_MODEL)

# Fallback local HF model
fallback_llm = pipeline("text-generation", model="TheBloke/guanaco-7B-HF", max_new_tokens=200)

# Cache
VECTOR_CACHE = {}
TRANSLATION_CACHE = {}

# ---------------- Helpers ----------------
def get_splitter(text_length):
    if text_length > 50000:
        return RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def translate_cached(text, source, target):
    key = f"{source}->{target}:{text}"
    if key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[key]
    translated = GoogleTranslator(source=source, target=target).translate(text)
    TRANSLATION_CACHE[key] = translated
    return translated

# ---------------- PDF Extraction ----------------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    if not text.strip() and OCR_AVAILABLE:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path)
        for page_image in pages:
            text += pytesseract.image_to_string(page_image) + "\n"

    return text.strip()

# ---------------- Process PDF ----------------
def process_pdf_to_chroma(pdf_path: str, collection_name: str) -> str:
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        raise ValueError("No text found in PDF")

    splitter = get_splitter(len(raw_text))
    chunks = splitter.split_text(raw_text)
    logging.info(f"PDF split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_ROOT
    )
    vector_db.persist()
    VECTOR_CACHE[collection_name] = vector_db
    return collection_name

def load_vector_db(collection_name: str) -> Chroma:
    if collection_name in VECTOR_CACHE:
        return VECTOR_CACHE[collection_name]
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME),
        persist_directory=CHROMA_ROOT
    )
    VECTOR_CACHE[collection_name] = vector_db
    return vector_db

# ---------------- Ask Question ----------------
def ask_question(vector_db: Chroma, question: str, k: int = 5) -> str:
    if not question.strip():
        return "Please provide a question."

    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    logging.info(f"Retrieved {len(docs)} docs for question: {question}")

    if not docs:
        return "❌ Not found in PDF"

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a PDF Question Answering Assistant.
ONLY use the context below to answer.
If the answer is not in the context, reply with: "❌ Not found in PDF".

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        result = ollama_llm(prompt)
        return result.strip() if isinstance(result, str) else getattr(result, "content", str(result)).strip()
    except Exception as e:
        logging.warning(f"Ollama error: {e}, using fallback LLM")
        return fallback_llm(prompt)[0]["generated_text"]

# ---------------- Multilingual QA ----------------
def ask_question_multilingual(vector_db: Chroma, user_question: str, target_lang: str = "en") -> str:
    try:
        src_lang = detect(user_question)
    except:
        src_lang = "en"

    q_en = translate_cached(user_question, src_lang, "en") if src_lang != "en" else user_question
    answer_en = ask_question(vector_db, q_en)

    if target_lang != "en":
        return translate_cached(answer_en, "en", target_lang)
    return answer_en

# ---------------- Delete Collection ----------------
def delete_collection(collection_name: str):
    path = os.path.join(CHROMA_ROOT, collection_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    if collection_name in VECTOR_CACHE:
        del VECTOR_CACHE[collection_name]

def cleanup_all_chroma():
    for collection_name in list(VECTOR_CACHE.keys()):
        delete_collection(collection_name)
    VECTOR_CACHE.clear()
    logging.info("All Chroma clients closed and cache cleared.")    

# ---------------- Format Answer ----------------
def format_answer(raw_answer: str) -> str:
    text = raw_answer.strip()
    formatted = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            if ":" in line:
                title, desc = line.split(":", 1)
                formatted.append(f"<b>{title.strip()}</b><br>{desc.strip()}")
            else:
                formatted.append(f"• {line}")
    return "<br><br>".join(formatted)
