'''from django.shortcuts import render
from .forms import PDFQuestionForm
from .models import PDFDocument, QuestionHistory
from .utils import process_pdf, ask_question

def index(request):
    answer = None
    error = None

    if request.method == 'POST':
        form = PDFQuestionForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_file = form.cleaned_data['pdf']
            question = form.cleaned_data['question']

            try:
                # Save PDF
                pdf_doc = PDFDocument.objects.create(file=pdf_file)

                # Process PDF and store in Chroma
                process_pdf(pdf_doc.file.path)

                # Ask question
                answer = ask_question(question)

                # Save question & answer history
                QuestionHistory.objects.create(
                    question=question,
                    answer=answer,
                    pdf=pdf_doc
                )

            except Exception as e:
                error = str(e)

    else:
        form = PDFQuestionForm()

    return render(request, 'index.html', {
        'form': form,
        'answer': answer,
        'error': error
    })'''
'''from django.shortcuts import render
from .forms import PDFQuestionForm
from .models import PDFDocument, QuestionHistory
from .utils import index_pdf, ask_question

def index(request):
    answer = None
    error = None

    if request.method == 'POST':
        form = PDFQuestionForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_file = form.cleaned_data['pdf']
            question = form.cleaned_data['question']
            mode = form.cleaned_data.get('mode', 'qa')
            language = form.cleaned_data.get('language', 'English') or 'English'

            try:
                # Save file to disk via model
                pdf_doc = PDFDocument.objects.create(file=pdf_file)
                pdf_path = pdf_doc.file.path

                # Per-PDF collection to avoid mixing
                collection_name = f"pdf_{pdf_doc.id}"

                # Streamed + batched indexing (handles up to 10k pages)
                index_pdf(pdf_path, collection_name=collection_name)

                # Ask against this PDF only
                answer = ask_question(
                    question=question,
                    collection_name=collection_name,
                    mode=mode,
                    language=language
                )

                # Save history
                QuestionHistory.objects.create(
                    pdf=pdf_doc,
                    question=question,
                    answer=answer,
                    mode=mode,
                    language=language
                )

            except Exception as e:
                error = str(e)
        else:
            error = "Please fix form errors."
    else:
        form = PDFQuestionForm()

    return render(request, 'index.html', {'form': form, 'answer': answer, 'error': error})
'''
'''=========================================================================my =========================
from django.shortcuts import render
from django.conf import settings
from .forms import PDFQuestionForm
from .models import PDFDocument, QuestionHistory
from .utils import process_pdf_to_chroma, ask_question_multilingual
import uuid


def index(request):
    answer = None
    error = None

    if request.method == 'POST':
        form = PDFQuestionForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_file = form.cleaned_data['pdf']
            question = form.cleaned_data['question']
            answer_lang = form.cleaned_data['answer_lang']

            try:
                # 1) Save uploaded PDF
                doc = PDFDocument.objects.create(file=pdf_file)
                pdf_path = doc.file.path

                # 2) Build a fresh Chroma collection for this doc
                collection_name = f"doc_{doc.id}_{uuid.uuid4().hex[:8]}"
                vector_db = process_pdf_to_chroma(pdf_path, collection_name)

                # 3) Ask and translate to chosen language
                answer = ask_question_multilingual(vector_db, question, target_lang=answer_lang)

                # 4) Save Q/A
                QuestionHistory.objects.create(document=doc, question=question, answer=answer)

            except Exception as e:
                error = f"Error while processing: {e}"
    else:
        form = PDFQuestionForm()

    return render(request, 'index.html', {"form": form, "answer": answer, "error": error})'''
'''from django.shortcuts import render, redirect
from django.conf import settings
from .forms import PDFUploadForm, QuestionForm
from .models import PDFDocument
from .utils import process_pdf_to_chroma, ask_question_multilingual
import os
import uuid

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings




def upload_pdf(request):
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_file = form.cleaned_data['pdf']
            doc = PDFDocument.objects.create(file=pdf_file)
            pdf_path = doc.file.path

            # Generate a hash for the PDF file to use as collection name
            pdf_hash = get_file_hash(pdf_path)
            collection_name = f"pdf_{pdf_hash}"
            persist_dir = os.path.join(settings.CHROMA_ROOT, collection_name)

            # Only process if the vector DB doesn't exist yet for this PDF
            if not os.path.exists(persist_dir):
                process_pdf_to_chroma(pdf_path, collection_name)

            # Save collection info in session for asking questions
            request.session['collection_name'] = collection_name
            request.session['document_id'] = doc.id

            return redirect('ask_question')
    else:
        form = PDFUploadForm()

    return render(request, 'upload.html', {"form": form})




def ask_question(request):
    answer = None
    error = None
    collection_name = request.session.get('collection_name')

    if not collection_name:
        error = "No PDF uploaded. Please upload a PDF first."
        return render(request, 'ask.html', {"error": error})

    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['question']
            lang = form.cleaned_data['answer_lang']

            try:
                vector_db = Chroma(
                    collection_name=collection_name,
                    persist_directory=os.path.join(settings.CHROMA_ROOT, collection_name),
                    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                )
                answer_raw = ask_question_multilingual(vector_db, question, target_lang=lang)

                # If the answer contains newlines, format it as bullet list for display
                lines = [line.strip() for line in answer_raw.split('\n') if line.strip()]
                if len(lines) > 1:
                    answer = "<ul>" + "".join(f"<li>{line}</li>" for line in lines) + "</ul>"
                else:
                    answer = answer_raw

            except Exception as e:
                error = f"Error: {e}"
    else:
        form = QuestionForm()

    return render(request, 'ask.html', {"form": form, "answer": answer, "error": error})



# Generate a hash of the PDF file to identify it uniquely
import hashlib

def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def reset_session(request):
    request.session.flush()
    return redirect('upload_pdf')'''
import os
import uuid
from django.shortcuts import render, redirect
from .forms import PDFUploadForm, QuestionForm
from .utils import process_pdf_to_chroma, ask_question, ask_question_multilingual

# Temporary in-memory storage for vector DBs
SESSION_VECTOR_STORES = {}

# ---------------------------
# Upload PDF & create vector DB
# ---------------------------
def upload_pdf(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("pdf_file")
        if not uploaded_file:
            return render(request, "uploadpdf.html", {"error": "Please select a PDF file!"})

        # Save PDF temporarily
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(BASE_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        pdf_filename = f"{uuid.uuid4()}.pdf"
        pdf_path = os.path.join(upload_dir, pdf_filename)

        with open(pdf_path, "wb+") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Process PDF into vector DB
        vector_db = process_pdf_to_chroma(pdf_path, collection_name="my_pdf_collection")

        # Store vector DB in session
        user_id = str(uuid.uuid4())
        request.session['user_id'] = user_id
        SESSION_VECTOR_STORES[user_id] = vector_db

        # Delete PDF to save space
        os.remove(pdf_path)

        return render(request, "ask.html", {"message": "PDF processed successfully!"})

    return render(request, "uploadpdf.html")


# ---------------------------
# Ask question using vector DB
# ---------------------------
def ask_question_view(request):
    answer = None
    error = None

    user_id = request.session.get('user_id')
    vector_db = SESSION_VECTOR_STORES.get(user_id)

    if not vector_db:
        error = "Your session expired or no PDF uploaded. Please re-upload the PDF."
        form = QuestionForm()
    else:
        if request.method == "POST":
            form = QuestionForm(request.POST)
            if form.is_valid():
                question = form.cleaned_data['question']
                # You can switch between ask_question or multilingual version
                answer = ask_question_multilingual(vector_db, question)
        else:
            form = QuestionForm()

    return render(request, "ask.html", {"form": form, "answer": answer, "error": error})


# ---------------------------
# Reset session (clear vector DB)
# ---------------------------
def reset_session(request):
    user_id = request.session.get('user_id')
    if user_id in SESSION_VECTOR_STORES:
        del SESSION_VECTOR_STORES[user_id]

    request.session.flush()
    return redirect('/')
