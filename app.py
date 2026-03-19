import os
from dotenv import load_dotenv
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

import PyPDF2

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY missing")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
VECTOR_INDEX = "vector_index"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

llm = ChatOpenAI(
    model="mistralai/devstral-2512:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.2,
)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def extract_text_from_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text.strip()


def build_vector_index(resume_text: str):
    docs = [Document(page_content=resume_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embedding)
    db.save_local(VECTOR_INDEX)


def load_retriever():
    if not os.path.exists(VECTOR_INDEX):
        raise RuntimeError("No resume indexed yet")

    db = FAISS.load_local(
        VECTOR_INDEX,
        embedding,
        allow_dangerous_deserialization=True,
    )
    return db.as_retriever(search_kwargs={"k": 4})


def build_rag_chain():
    retriever = load_retriever()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AI career assistant. "
            "Answer ONLY using the resume content below.\n\n{context}"
        ),
        ("human", "{input}")
    ])

    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return rag_chain


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    resume_text = extract_text_from_pdf(file_path)
    if not resume_text.strip():
        return "Failed to extract resume text"


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(resume_text)

    vectorstore = FAISS.from_texts(chunks, embedding)
    vectorstore.save_local("vector_index")


    resume_analysis_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert AI career coach. "
            "Analyze the resume and provide a clear, structured summary."
        ),
        ("human", "{resume}")
    ])

    resume_analysis_chain = resume_analysis_prompt | llm

    resume_analysis = resume_analysis_chain.invoke(
        {"resume": resume_text}
    ).content

    return render_template(
        "results.html",
        resume_analysis=resume_analysis
    )


@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "POST":
        question = request.form["query"]

        rag_chain = build_rag_chain()
        response = rag_chain.invoke(question)

        return render_template(
            "qa_results.html",
            query=question,
            result=response.content,
        )

    return render_template("ask.html")


if __name__ == "__main__":
    app.run(debug=True)
