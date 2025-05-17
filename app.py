import streamlit as st
from rag_pipeline import create_qa_chain

st.set_page_config(page_title="FDA RAG Chatbot", layout="wide")
st.title("💊 FDA Regulatory Chatbot")
st.markdown("Ask questions about FDA drug approval using guidance documents.")

qa_chain = create_qa_chain()
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain(query)
        answer = result.get("answer", "No answer returned.")
        docs = result.get("source_documents", [])

        st.markdown("### ✅ Answer:")
        st.write(answer)

        if docs:
            st.markdown("---")
            st.markdown("### 📚 Sources used:")
            for i, doc in enumerate(docs):
                text = doc.page_content.strip().replace("\n", " ")
                preview = text[:400] + "..." if len(text) > 400 else text
                st.markdown(f"**Source {i + 1}:**")
                st.code(preview, language="markdown")
        else:
            st.markdown("⚠️ No documents were retrieved as sources.")
