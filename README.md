# FDA Regulatory RAG Chatbot

This is a retrieval-augmented chatbot designed to answer questions about FDA and ICH regulatory guidance using official documents. The system uses semantic search over guidance PDFs and responds with accurate, source-grounded answers.

The project is structured to grow into a broader regulatory assistant by incorporating more documents over time.

---

## Purpose

This chatbot is intended for regulatory consultants, pharmaceutical professionals, and researchers who need fast, context-specific answers from complex regulatory documents. It helps surface relevant sections of official guidance and summarizes them into concise responses.

---

## How It Works

- Guidance documents (PDF) are parsed and split into small, overlapping text chunks.
- Chunks are embedded using a sentence-transformer model and indexed using Chroma vector storage.
- When a user submits a question, the most relevant chunks are retrieved semantically.
- A language model generates a response using only the retrieved context, with citations to the original source documents.

---

## Example Use Cases

- Clarifying FDA requirements for GMP documentation
- Understanding stages of process validation
- Summarizing ICH perspectives on quality systems
- Extracting retention requirements for regulatory records

---

## Project Direction

This assistant is designed for expansion. New documents can be added to the `documents/` directory and will be automatically incorporated into the search index. The goal is to scale the assistant into a domain-specific tool for navigating pharmaceutical regulatory frameworks.

Future iterations may include:
- Deployment as a web application
- Support for document filtering by agency, year, or type
- Integration with live FDA/EMA databases or APIs
- Inference optimization with rerankers or hybrid search

---

## Technologies

- LangChain
- Chroma (vector store)
- HuggingFace sentence-transformers
- OpenRouter (OpenAI-compatible LLM API)
- Streamlit (for local interface)
- PyMuPDF (for PDF parsing)

---

## License

This project is for demonstration and educational purposes. All guidance documents are publicly available from FDA
