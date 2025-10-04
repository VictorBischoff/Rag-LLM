import asyncio
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from main import OptimizedRAGSystem

# Initialize FastAPI app
app = FastAPI(
    title="RAG PDF Query API",
    description="A simple API for uploading PDFs and querying them using RAG with MLX",
    version="1.0.0",
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
# Track session metadata and directories for cleanup
session_metadata: Dict[str, float] = {}
session_directories: Dict[str, Path] = {}

# Lock to guard session registry mutations
session_lock = asyncio.Lock()

# Session time-to-live in seconds (default: 1 hour)
SESSION_TTL_SECONDS = int(os.getenv("RAG_SESSION_TTL_SECONDS", "3600"))


def _cleanup_session_directory(directory: Optional[Path]) -> None:
    """Remove the temporary directory associated with a session."""
    if not directory:
        return
    try:
        if directory.exists() and directory.name.startswith("rag_session_"):
            shutil.rmtree(directory, ignore_errors=True)
    except Exception:
        # Intentionally swallow cleanup errors – avoid masking upstream exceptions
        pass


def _collect_session_cleanup(session_id: str) -> Optional[Tuple[str, OptimizedRAGSystem, Optional[Path]]]:
    """Remove session bookkeeping under lock and return info for deferred cleanup."""
    rag_system = active_rag_systems.pop(session_id, None)
    session_metadata.pop(session_id, None)
    session_dir = session_directories.pop(session_id, None)
    if rag_system:
        return session_id, rag_system, session_dir
    return None


async def _cleanup_expired_sessions() -> None:
    """Expire sessions that have outlived the TTL."""
    if SESSION_TTL_SECONDS <= 0:
        return

    now = time.time()
    expired: List[Tuple[str, OptimizedRAGSystem, Optional[Path]]] = []

    async with session_lock:
        for session_id, created in list(session_metadata.items()):
            if now - created > SESSION_TTL_SECONDS:
                info = _collect_session_cleanup(session_id)
                if info:
                    expired.append(info)

    for _, _, directory in expired:
        _cleanup_session_directory(directory)


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
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    await _cleanup_expired_sessions()
    async with session_lock:
        active_count = len(active_rag_systems)
    return {"status": "healthy", "active_sessions": active_count}


@app.get("/logs/{session_id}")
async def get_session_logs(session_id: str):
    """Get logs for a specific session."""
    await _cleanup_expired_sessions()

    async with session_lock:
        if session_id not in active_rag_systems:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        rag_system = active_rag_systems[session_id]

    # Get timing stats and basic info
    logs = {
        "session_id": session_id,
        "timing_stats": dict(rag_system.timing_stats),
        "model_id": rag_system.model_id,
        "document_count": len(rag_system.documents) if rag_system.documents else 0,
        "cache_dir": str(rag_system.cache_dir),
        "system_config": {
            "chunk_size": rag_system.chunk_size,
            "chunk_overlap": rag_system.chunk_overlap,
            "max_tokens": rag_system.max_tokens,
            "temperature": rag_system.temperature,
        },
    }

    return logs


@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files and initialize a RAG system for them.

    Args:
        files: List of PDF files to upload

    Returns:
        UploadResponse with session_id and processing info
    """
    if not files:
        raise HTTPException(
            status_code=400, detail="At least one file must be uploaded"
        )

    # Validate file types
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported. Found: {file.filename}",
            )

    await _cleanup_expired_sessions()

    session_id = str(uuid.uuid4())
    temp_dir: Optional[Path] = None

    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"rag_session_{session_id}_"))

        pdf_paths: List[str] = []
        for index, file in enumerate(files):
            if not file.filename:
                continue

            safe_name = Path(file.filename).name or f"document_{index}.pdf"
            stem = Path(safe_name).stem or f"document_{index}"
            suffix = Path(safe_name).suffix or ".pdf"
            target_path = temp_dir / f"{stem}{suffix}"
            counter = 1

            while target_path.exists():
                target_path = temp_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            pdf_paths.append(str(target_path))

        if not pdf_paths:
            raise HTTPException(
                status_code=400,
                detail="No valid PDF files were provided.",
            )

        # Initialize RAG system with multiple PDFs
        rag_system = OptimizedRAGSystem(
            pdf_paths=pdf_paths,
            model_id="mlx-community/granite-4.0-h-tiny-4bit",
            cache_dir=str(temp_dir / "cache"),
            chunk_size=1500,
            chunk_overlap=100,
            max_tokens=500,
            temperature=0.1,
        )

        await run_in_threadpool(rag_system.initialize)

        total_time = sum(rag_system.timing_stats.values())
        total_chunks = len(rag_system.documents) if rag_system.documents else 0

        async with session_lock:
            active_rag_systems[session_id] = rag_system
            session_metadata[session_id] = time.time()
            session_directories[session_id] = temp_dir

        return UploadResponse(
            message=f"Successfully uploaded and processed {len(pdf_paths)} PDF file(s)",
            session_id=session_id,
            document_count=total_chunks,
            processing_time=total_time,
        )

    except HTTPException as exc:
        cleanup_dirs = {temp_dir} if temp_dir else set()

        async with session_lock:
            info = _collect_session_cleanup(session_id) if session_id in active_rag_systems else None

        if info and info[2]:
            cleanup_dirs.add(info[2])

        for directory in cleanup_dirs:
            _cleanup_session_directory(directory)

        raise exc

    except Exception as e:
        cleanup_dirs = {temp_dir} if temp_dir else set()

        async with session_lock:
            info = _collect_session_cleanup(session_id) if session_id in active_rag_systems else None

        if info and info[2]:
            cleanup_dirs.add(info[2])

        for directory in cleanup_dirs:
            _cleanup_session_directory(directory)

        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")


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
    await _cleanup_expired_sessions()

    session_id = request.session_id

    async with session_lock:
        if not session_id:
            if not active_rag_systems:
                raise HTTPException(
                    status_code=400,
                    detail="No active sessions. Please upload a PDF first.",
                )
            session_id = next(iter(active_rag_systems))

        if session_id not in active_rag_systems:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found. Please upload a PDF first.",
            )

        rag_system = active_rag_systems[session_id]

    try:
        response = await run_in_threadpool(rag_system.query, request.question)
        query_time = rag_system.timing_stats.get("Query Processing", 0.0)

        async with session_lock:
            session_metadata[session_id] = time.time()

        return QueryResponse(
            answer=response["answer"], session_id=session_id, processing_time=query_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    await _cleanup_expired_sessions()

    async with session_lock:
        sessions = list(active_rag_systems.keys())

    return {"active_sessions": sessions, "count": len(sessions)}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and clean up resources."""
    await _cleanup_expired_sessions()

    async with session_lock:
        if session_id not in active_rag_systems:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        info = _collect_session_cleanup(session_id)

    if info:
        _, _, directory = info
        _cleanup_session_directory(directory)

    return {"message": f"Session {session_id} deleted successfully"}


@app.delete("/sessions")
async def delete_all_sessions():
    """Delete all active sessions and clean up resources."""
    await _cleanup_expired_sessions()

    async with session_lock:
        infos = [
            _collect_session_cleanup(session_id)
            for session_id in list(active_rag_systems.keys())
        ]

    for info in infos:
        if info:
            _cleanup_session_directory(info[2])

    return {"message": "All sessions deleted successfully"}


# Streamlit UI is run separately with: streamlit run ui.py

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",  # Changed to debug for more verbose logging
    )
