import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

load_dotenv()
CHROMA_DIR = "chroma_db"

def load_pdfs(folder_path):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            doc = fitz.open(path)
            text = "\n".join([page.get_text() for page in doc])
            file_chunks = splitter.create_documents([text])
            for chunk in file_chunks:
                chunk.metadata["source"] = filename  # Required for citation
            chunks.extend(file_chunks)
    
    print(f"✅ Loaded {len(chunks)} chunks from {folder_path}")
    return chunks

def create_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        chunks = load_pdfs("documents/")
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # ✅ More precise retrieval

    llm = ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )

    # ✅ Improved prompt for natural rephrasing and source grounding
    custom_prompt = PromptTemplate.from_template(
        """You are a regulatory assistant specializing in FDA drug approval.

Use the following guidance document excerpts to answer the user's question as clearly and accurately as possible.

Do not copy text directly. Instead, rephrase and summarize the most relevant information. If you're unsure, say "I don't know." Always cite the document filename (e.g., GMP.pdf) in your answer.

### CONTEXT:
{summaries}

### QUESTION:
{question}

### ANSWER:"""
    )

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )

    return qa_chain
