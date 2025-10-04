"""
Streamlit UI for RAG PDF Query API.

This module provides a simple web interface for uploading PDFs and querying them
using the existing FastAPI endpoints.
"""

import streamlit as st
import requests
import tempfile
import os
from pathlib import Path

API_BASE = "http://localhost:8000"

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, f"✅ API is healthy - {data['active_sessions']} active sessions"
        else:
            return False, f"❌ API health check failed: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"❌ Cannot connect to API: {e}\nMake sure the API server is running on http://localhost:8000"

def upload_pdfs(uploaded_files):
    """Handle multiple PDF uploads."""
    try:
        files = []
        for uploaded_file in uploaded_files:
            files.append(("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")))
        
        response = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            return data["session_id"], f"✅ Upload successful!\nFiles: {len(uploaded_files)} | Chunks: {data['document_count']} | Time: {data['processing_time']:.2f}s"
        else:
            return None, f"❌ Upload failed: {response.status_code}\n{response.text}"
            
    except Exception as e:
        return None, f"❌ Upload error: {str(e)}"

def query_document(question, session_id):
    """Handle document querying."""
    try:
        payload = {"question": question, "session_id": session_id}
        response = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            return data["answer"]
        else:
            return f"❌ Query failed: {response.status_code}\n{response.text}"
            
    except Exception as e:
        return f"❌ Query error: {str(e)}"

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RAG PDF Query",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG PDF Query Interface")
    st.markdown("Upload one or more PDF documents and ask questions about their content using AI-powered retrieval-augmented generation.")
    
    # API Health Check
    st.sidebar.header("API Status")
    if st.sidebar.button("Check API Health"):
        is_healthy, status = check_api_health()
        if is_healthy:
            st.sidebar.success(status)
        else:
            st.sidebar.error(status)
    
    # Session Management
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    
    # PDF Upload Section
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Select PDF Files", 
        type=['pdf'], 
        accept_multiple_files=True,
        help="You can upload one or more PDF files to create a combined knowledge base"
    )
    
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} file(s): {', '.join([f.name for f in uploaded_files])}")
        
        if st.button("Upload PDFs"):
            with st.spinner("Uploading and processing PDFs..."):
                session_id, status = upload_pdfs(uploaded_files)
                if session_id:
                    st.session_state.session_id = session_id
                    st.success(status)
                else:
                    st.error(status)
    
    # Display current session
    if st.session_state.session_id:
        st.info(f"**Current Session ID:** `{st.session_state.session_id}`")
    
    # Query Section
    st.header("❓ Ask Questions")
    
    question = st.text_area(
        "Your Question",
        placeholder="What would you like to know about the document?",
        height=100
    )
    
    if st.button("Ask Question", disabled=not st.session_state.session_id):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not st.session_state.session_id:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Processing your question..."):
                answer = query_document(question, st.session_state.session_id)
                st.markdown("### Answer:")
                st.markdown(answer)
    
    # Example questions
    st.header("💡 Example Questions")
    examples = [
        "What is the main topic across all documents?",
        "Can you summarize the key points from all documents?",
        "What are the main conclusions?",
        "Compare the methodologies used in different documents",
        "What are the common themes across all documents?"
    ]
    
    # Example questions section
    st.markdown("Click any example below to use it:")
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"**{example}**", key=f"example_{i}", help="Click to use this question"):
                question = example

if __name__ == "__main__":
    main()
