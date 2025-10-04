"""
Shared test fixtures and utilities for RAG-LLM project.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch
import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the modules we're testing
from main import OptimizedRAGSystem
import api


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp(prefix="rag_test_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Create a dummy PDF file for testing."""
    pdf_path = temp_dir / "test.pdf"
    # Create a minimal PDF file (just dummy bytes)
    pdf_path.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"
    )
    return str(pdf_path)


@pytest.fixture
def sample_documents():
    """Create sample Document objects for testing."""
    return [
        Document(
            page_content="This is a test document about machine learning and artificial intelligence.",
            metadata={"source_file": "test.pdf", "source_path": "/path/to/test.pdf"},
        ),
        Document(
            page_content="Another document discussing natural language processing and transformers.",
            metadata={"source_file": "test.pdf", "source_path": "/path/to/test.pdf"},
        ),
        Document(
            page_content="A third document about computer vision and deep learning applications.",
            metadata={"source_file": "test.pdf", "source_path": "/path/to/test.pdf"},
        ),
    ]


@pytest.fixture
def mock_pymupdf_loader():
    """Mock PyMuPDFLoader for testing."""
    with patch("main.PyMuPDFLoader") as mock_loader:
        mock_instance = Mock()
        mock_loader.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_huggingface_embeddings():
    """Mock HuggingFaceEmbeddings for testing."""
    with patch("main.HuggingFaceEmbeddings") as mock_embeddings:
        mock_instance = Mock()
        mock_embeddings.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_faiss():
    """Mock FAISS for testing."""
    with patch("main.FAISS") as mock_faiss:
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss.from_documents.return_value = mock_vectorstore
        yield mock_vectorstore, mock_retriever


@pytest.fixture
def mock_mlx_pipeline():
    """Mock MLXPipeline for testing."""
    with patch("main.MLXPipeline") as mock_pipeline:
        mock_instance = Mock()
        mock_pipeline.from_model_id.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_rag_chain():
    """Mock RAG chain for testing."""
    mock_chain = Mock()
    mock_chain.invoke.return_value = {
        "answer": "This is a mock answer to your question.",
        "context": "Mock context from documents",
    }
    return mock_chain


@pytest.fixture
def stubbed_rag_system(temp_dir, sample_documents):
    """Create a RAG system with all external dependencies mocked."""
    # Create a test PDF file
    pdf_path = temp_dir / "test.pdf"
    create_test_pdf(pdf_path, "Test content for stubbed RAG system.")

    # Create RAG system
    rag = OptimizedRAGSystem(
        pdf_paths=[str(pdf_path)],
        cache_dir=str(temp_dir / "cache"),
        model_id="test-model",
    )

    # Mock all the components without actually initializing
    rag.documents = sample_documents
    rag.vectorstore = Mock()
    rag.retriever = Mock()
    rag.llm = Mock()

    # Create a mock chain
    mock_chain = Mock()
    mock_chain.invoke.return_value = {
        "answer": "Mock answer",
        "context": "Mock context",
    }
    rag.chain = mock_chain

    # Add some timing stats
    rag.timing_stats = {
        "Document Processing": 1.0,
        "MLX Model Loading": 2.0,
        "Chain Setup": 0.5,
    }

    yield rag


@pytest.fixture
def fastapi_app():
    """Create a FastAPI app instance for testing."""
    return api.app


@pytest.fixture
def mock_requests():
    """Mock requests for HTTP client testing."""
    with (
        patch("requests.get") as mock_get,
        patch("requests.post") as mock_post,
        patch("requests.delete") as mock_delete,
    ):
        yield {"get": mock_get, "post": mock_post, "delete": mock_delete}


@pytest.fixture
def sample_session_data():
    """Sample session data for API testing."""
    return {
        "session_id": "test-session-123",
        "timing_stats": {
            "Document Processing": 1.5,
            "Embeddings & Vector Store Setup": 2.3,
            "MLX Model Loading": 3.1,
            "Chain Setup": 0.8,
        },
        "model_id": "test-model",
        "document_count": 3,
        "cache_dir": "/tmp/test_cache",
        "system_config": {
            "chunk_size": 1500,
            "chunk_overlap": 100,
            "max_tokens": 500,
            "temperature": 0.1,
        },
    }


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for UI testing."""
    with (
        patch("streamlit.set_page_config") as mock_config,
        patch("streamlit.title") as mock_title,
        patch("streamlit.markdown") as mock_markdown,
        patch("streamlit.sidebar") as mock_sidebar,
        patch("streamlit.header") as mock_header,
        patch("streamlit.file_uploader") as mock_uploader,
        patch("streamlit.button") as mock_button,
        patch("streamlit.text_area") as mock_text_area,
        patch("streamlit.spinner") as mock_spinner,
        patch("streamlit.info") as mock_info,
        patch("streamlit.success") as mock_success,
        patch("streamlit.error") as mock_error,
        patch("streamlit.warning") as mock_warning,
        patch("streamlit.columns") as mock_columns,
        patch("streamlit.session_state", {}) as mock_session_state,
    ):
        # Setup common mock behaviors
        mock_sidebar.header.return_value = mock_sidebar
        mock_sidebar.button.return_value = False
        mock_sidebar.success.return_value = None
        mock_sidebar.error.return_value = None

        mock_uploader.return_value = []
        mock_button.return_value = False
        mock_text_area.return_value = ""
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()

        mock_columns.return_value = [Mock(), Mock()]

        yield {
            "config": mock_config,
            "title": mock_title,
            "markdown": mock_markdown,
            "sidebar": mock_sidebar,
            "header": mock_header,
            "uploader": mock_uploader,
            "button": mock_button,
            "text_area": mock_text_area,
            "spinner": mock_spinner,
            "info": mock_info,
            "success": mock_success,
            "error": mock_error,
            "warning": mock_warning,
            "columns": mock_columns,
            "session_state": mock_session_state,
        }


@pytest.fixture(autouse=True)
def cleanup_active_sessions():
    """Clean up active RAG systems after each test."""
    yield
    api.active_rag_systems.clear()


@pytest.fixture
def mock_time():
    """Mock time for deterministic timing tests."""
    with patch("time.perf_counter") as mock_counter:
        # Use a generator that provides increasing times for each call
        # This is more robust than a fixed list
        call_count = 0

        def time_generator():
            nonlocal call_count
            call_count += 1
            return call_count * 1.5  # Each call returns 1.5s more than the previous

        mock_counter.side_effect = time_generator
        yield mock_counter


@pytest.fixture
def mock_uuid():
    """Mock UUID generation for consistent session IDs."""
    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value.hex = "test-session-123"
        yield mock_uuid


# Utility functions for tests
def create_test_pdf(path: Path, content: str = "Test PDF content") -> None:
    """Create a minimal test PDF file."""
    # This is a very basic PDF structure - in real tests you might want to use a proper PDF library
    pdf_content = f"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length {len(content)}
>>
stream
{content}
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000200 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
{250 + len(content)}
%%EOF"""
    path.write_text(pdf_content)


def assert_timing_stats_present(
    rag_system: OptimizedRAGSystem, expected_operations: List[str]
):
    """Assert that timing stats contain expected operations."""
    for operation in expected_operations:
        assert operation in rag_system.timing_stats, f"Missing timing stat: {operation}"
        assert isinstance(rag_system.timing_stats[operation], (int, float)), (
            f"Timing stat {operation} should be numeric"
        )


def create_mock_upload_file(filename: str, content: bytes = b"test content") -> Mock:
    """Create a mock UploadFile for testing."""
    mock_file = Mock()
    mock_file.filename = filename
    mock_file.file = Mock()
    mock_file.file.read.return_value = content
    return mock_file
