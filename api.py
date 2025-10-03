import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from main import OptimizedRAGSystem

# Initialize FastAPI app
app = FastAPI(
    title="RAG PDF Query API",
    description="A simple API for uploading PDFs and querying them using RAG with MLX",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store active RAG systems
active_rag_systems: Dict[str, OptimizedRAGSystem] = {}

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    processing_time: float

class UploadResponse(BaseModel):
    message: str
    session_id: str
    document_count: int
    processing_time: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG PDF Query API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_sessions": len(active_rag_systems)}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and initialize a RAG system for it.
    
    Args:
        file: PDF file to upload
        
    Returns:
        UploadResponse with session_id and processing info
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported"
        )
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    try:
        # Create temporary directory for this session
        temp_dir = Path(tempfile.mkdtemp(prefix=f"rag_session_{session_id}_"))
        
        # Save uploaded file
        pdf_path = temp_dir / file.filename
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize RAG system
        rag_system = OptimizedRAGSystem(
            pdf_path=str(pdf_path),
            model_id="mlx-community/granite-4.0-h-tiny-6bit-MLX",
            cache_dir=str(temp_dir / "cache"),
            chunk_size=1500,
            chunk_overlap=100,
            max_tokens=500,
            temperature=0.1
        )
        
        # Initialize the system (this processes the PDF)
        rag_system.initialize()
        
        # Store the RAG system
        active_rag_systems[session_id] = rag_system
        
        # Get processing time from the system
        total_time = sum(rag_system.timing_stats.values())
        
        return UploadResponse(
            message="PDF uploaded and processed successfully",
            session_id=session_id,
            document_count=len(rag_system.documents) if rag_system.documents else 0,
            processing_time=total_time
        )
        
    except Exception as e:
        # Clean up on error
        if session_id in active_rag_systems:
            del active_rag_systems[session_id]
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query a document using the RAG system.
    
    Args:
        request: QueryRequest with question and optional session_id
        
    Returns:
        QueryResponse with answer and processing info
    """
    # Use provided session_id or get the first available one
    session_id = request.session_id
    if not session_id:
        if not active_rag_systems:
            raise HTTPException(
                status_code=400,
                detail="No active sessions. Please upload a PDF first."
            )
        session_id = list(active_rag_systems.keys())[0]
    
    # Check if session exists
    if session_id not in active_rag_systems:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found. Please upload a PDF first."
        )
    
    try:
        # Get the RAG system
        rag_system = active_rag_systems[session_id]
        
        # Process the query
        response = rag_system.query(request.question)
        
        # Get processing time
        query_time = rag_system.timing_stats.get("Query Processing", 0.0)
        
        return QueryResponse(
            answer=response['answer'],
            session_id=session_id,
            processing_time=query_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "active_sessions": list(active_rag_systems.keys()),
        "count": len(active_rag_systems)
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and clean up resources."""
    if session_id not in active_rag_systems:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    try:
        # Clean up the RAG system
        rag_system = active_rag_systems[session_id]
        
        # Remove from active systems
        del active_rag_systems[session_id]
        
        # Clean up temporary files if possible
        if hasattr(rag_system, 'cache_dir'):
            cache_path = Path(rag_system.cache_dir)
            if cache_path.exists() and cache_path.parent.name.startswith("rag_session_"):
                shutil.rmtree(cache_path.parent, ignore_errors=True)
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting session: {str(e)}"
        )

@app.delete("/sessions")
async def delete_all_sessions():
    """Delete all active sessions and clean up resources."""
    try:
        # Clean up all sessions
        for session_id in list(active_rag_systems.keys()):
            rag_system = active_rag_systems[session_id]
            
            # Clean up temporary files if possible
            if hasattr(rag_system, 'cache_dir'):
                cache_path = Path(rag_system.cache_dir)
                if cache_path.exists() and cache_path.parent.name.startswith("rag_session_"):
                    shutil.rmtree(cache_path.parent, ignore_errors=True)
        
        # Clear all sessions
        active_rag_systems.clear()
        
        return {"message": "All sessions deleted successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting sessions: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
