"""
Unit tests for API endpoints in api.py.

These tests isolate the upload_pdfs, query_document, and session management
functionality with mocked RAG systems and HTTP responses.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

import api


@pytest.mark.unit
class TestAPIRoot:
    """Test cases for root endpoint."""

    def test_root_endpoint(self, fastapi_app):
        """Test the root endpoint returns API information."""
        client = TestClient(fastapi_app)

        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "RAG PDF Query API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert "upload" in data["endpoints"]
        assert "query" in data["endpoints"]
        assert "health" in data["endpoints"]


@pytest.mark.unit
class TestHealthCheck:
    """Test cases for health check endpoint."""

    def test_health_check_success(self, fastapi_app):
        """Test successful health check."""
        client = TestClient(fastapi_app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "active_sessions" in data
        assert isinstance(data["active_sessions"], int)


@pytest.mark.unit
class TestSessionLogs:
    """Test cases for session logs endpoint."""

    def test_get_session_logs_success(self, fastapi_app, sample_session_data):
        """Test successful retrieval of session logs."""
        # Add a session to active_rag_systems
        session_id = sample_session_data["session_id"]
        mock_rag = Mock()
        mock_rag.timing_stats = sample_session_data["timing_stats"]
        mock_rag.model_id = sample_session_data["model_id"]
        mock_rag.documents = [Mock()] * sample_session_data["document_count"]
        mock_rag.cache_dir = sample_session_data["cache_dir"]
        mock_rag.chunk_size = sample_session_data["system_config"]["chunk_size"]
        mock_rag.chunk_overlap = sample_session_data["system_config"]["chunk_overlap"]
        mock_rag.max_tokens = sample_session_data["system_config"]["max_tokens"]
        mock_rag.temperature = sample_session_data["system_config"]["temperature"]

        api.active_rag_systems[session_id] = mock_rag

        client = TestClient(fastapi_app)
        response = client.get(f"/logs/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["timing_stats"] == sample_session_data["timing_stats"]
        assert data["model_id"] == sample_session_data["model_id"]
        assert data["document_count"] == sample_session_data["document_count"]

    def test_get_session_logs_not_found(self, fastapi_app):
        """Test getting logs for non-existent session."""
        client = TestClient(fastapi_app)

        response = client.get("/logs/non-existent-session")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


@pytest.mark.unit
class TestUploadPDFs:
    """Test cases for PDF upload endpoint."""

    def test_upload_pdfs_success_single_file(self, fastapi_app, temp_dir):
        """Test successful upload of single PDF file."""
        # Create a test PDF file
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        client = TestClient(fastapi_app)

        with open(pdf_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}

            with patch("api.OptimizedRAGSystem") as mock_rag_class:
                mock_rag = Mock()
                mock_rag.documents = [Mock(), Mock(), Mock()]  # 3 documents
                mock_rag.timing_stats = {
                    "Document Processing": 1.5,
                    "MLX Model Loading": 2.0,
                }
                mock_rag_class.return_value = mock_rag

                response = client.post("/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["message"].startswith("Successfully uploaded")
        assert data["document_count"] == 3
        assert data["processing_time"] == 3.5  # Sum of timing stats

        # Verify RAG system was created and initialized
        mock_rag_class.assert_called_once()
        mock_rag.initialize.assert_called_once()

    def test_upload_pdfs_success_multiple_files(self, fastapi_app, temp_dir):
        """Test successful upload of multiple PDF files."""
        # Create test PDF files
        pdf1 = temp_dir / "test1.pdf"
        pdf2 = temp_dir / "test2.pdf"
        pdf1.write_bytes(b"dummy pdf content 1")
        pdf2.write_bytes(b"dummy pdf content 2")

        client = TestClient(fastapi_app)

        with open(pdf1, "rb") as f1, open(pdf2, "rb") as f2:
            files = [
                ("files", ("test1.pdf", f1, "application/pdf")),
                ("files", ("test2.pdf", f2, "application/pdf")),
            ]

            with patch("api.OptimizedRAGSystem") as mock_rag_class:
                mock_rag = Mock()
                mock_rag.documents = [Mock()] * 5  # 5 documents total
                mock_rag.timing_stats = {"Document Processing": 2.0}
                mock_rag_class.return_value = mock_rag

                response = client.post("/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "2 PDF file(s)" in data["message"]
        assert data["document_count"] == 5

    def test_upload_pdfs_no_files(self, fastapi_app):
        """Test upload with no files provided."""
        client = TestClient(fastapi_app)

        response = client.post("/upload", files=[])

        assert response.status_code == 422  # FastAPI returns 422 for validation errors
        data = response.json()
        assert "detail" in data

    def test_upload_pdfs_invalid_file_type(self, fastapi_app, temp_dir):
        """Test upload with invalid file type."""
        # Create a non-PDF file
        txt_path = temp_dir / "test.txt"
        txt_path.write_text("This is not a PDF")

        client = TestClient(fastapi_app)

        with open(txt_path, "rb") as f:
            files = {"files": ("test.txt", f, "text/plain")}
            response = client.post("/upload", files=files)

        assert response.status_code == 400
        data = response.json()
        assert "only pdf files" in data["detail"].lower()

    def test_upload_pdfs_missing_filename(self, fastapi_app):
        """Test upload with file missing filename."""
        client = TestClient(fastapi_app)

        files = {"files": (None, b"dummy content", "application/pdf")}
        response = client.post("/upload", files=files)

        assert response.status_code == 422  # FastAPI returns 422 for validation errors
        data = response.json()
        assert "detail" in data

    @patch("api.OptimizedRAGSystem")
    def test_upload_pdfs_rag_initialization_error(
        self, mock_rag_class, fastapi_app, temp_dir
    ):
        """Test upload when RAG system initialization fails."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        # Make RAG initialization raise an exception
        mock_rag = Mock()
        mock_rag.initialize.side_effect = Exception("Initialization failed")
        mock_rag_class.return_value = mock_rag

        client = TestClient(fastapi_app)

        with open(pdf_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}
            response = client.post("/upload", files=files)

        assert response.status_code == 500
        data = response.json()
        assert "error processing pdfs" in data["detail"].lower()
        assert "initialization failed" in data["detail"].lower()

    @patch("api.OptimizedRAGSystem")
    def test_upload_pdfs_cleanup_on_error(self, mock_rag_class, fastapi_app, temp_dir):
        """Test that sessions are cleaned up on upload error."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        # Make RAG initialization raise an exception
        mock_rag = Mock()
        mock_rag.initialize.side_effect = Exception("Initialization failed")
        mock_rag_class.return_value = mock_rag

        client = TestClient(fastapi_app)

        with open(pdf_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}
            response = client.post("/upload", files=files)

        assert response.status_code == 500
        # Verify no sessions were left in active_rag_systems
        assert len(api.active_rag_systems) == 0


@pytest.mark.unit
class TestQueryDocument:
    """Test cases for document query endpoint."""

    def test_query_document_success_with_session_id(self, fastapi_app):
        """Test successful query with specific session ID."""
        session_id = "test-session-123"

        # Setup mock RAG system
        mock_rag = Mock()
        mock_rag.query.return_value = {
            "answer": "This is a test answer",
            "context": "Test context",
        }
        mock_rag.timing_stats = {"Query Processing": 1.5}
        api.active_rag_systems[session_id] = mock_rag

        client = TestClient(fastapi_app)

        payload = {"question": "What is this document about?", "session_id": session_id}
        response = client.post("/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a test answer"
        assert data["session_id"] == session_id
        assert data["processing_time"] == 1.5

        # Verify RAG system was called correctly
        mock_rag.query.assert_called_once_with("What is this document about?")

    def test_query_document_success_without_session_id(self, fastapi_app):
        """Test successful query using first available session."""
        session_id = "test-session-123"

        # Setup mock RAG system
        mock_rag = Mock()
        mock_rag.query.return_value = {
            "answer": "This is a test answer",
            "context": "Test context",
        }
        mock_rag.timing_stats = {"Query Processing": 2.0}
        api.active_rag_systems[session_id] = mock_rag

        client = TestClient(fastapi_app)

        payload = {"question": "What is this document about?"}
        response = client.post("/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a test answer"
        assert data["session_id"] == session_id
        assert data["processing_time"] == 2.0

    def test_query_document_no_active_sessions(self, fastapi_app):
        """Test query when no active sessions exist."""
        client = TestClient(fastapi_app)

        payload = {"question": "What is this document about?"}
        response = client.post("/query", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "no active sessions" in data["detail"].lower()

    def test_query_document_session_not_found(self, fastapi_app):
        """Test query with non-existent session ID."""
        client = TestClient(fastapi_app)

        payload = {
            "question": "What is this document about?",
            "session_id": "non-existent-session",
        }
        response = client.post("/query", json=payload)

        assert response.status_code == 404
        data = response.json()
        assert "session" in data["detail"].lower()
        assert "not found" in data["detail"].lower()

    def test_query_document_rag_query_error(self, fastapi_app):
        """Test query when RAG system query fails."""
        session_id = "test-session-123"

        # Setup mock RAG system that raises an exception
        mock_rag = Mock()
        mock_rag.query.side_effect = Exception("Query processing failed")
        api.active_rag_systems[session_id] = mock_rag

        client = TestClient(fastapi_app)

        payload = {"question": "What is this document about?", "session_id": session_id}
        response = client.post("/query", json=payload)

        assert response.status_code == 500
        data = response.json()
        assert "error processing query" in data["detail"].lower()
        assert "query processing failed" in data["detail"].lower()


@pytest.mark.unit
class TestListSessions:
    """Test cases for list sessions endpoint."""

    def test_list_sessions_empty(self, fastapi_app):
        """Test listing sessions when none exist."""
        client = TestClient(fastapi_app)

        response = client.get("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert data["active_sessions"] == []
        assert data["count"] == 0

    def test_list_sessions_with_active_sessions(self, fastapi_app):
        """Test listing sessions with active sessions."""
        # Add some sessions
        session1 = "session-1"
        session2 = "session-2"
        api.active_rag_systems[session1] = Mock()
        api.active_rag_systems[session2] = Mock()

        client = TestClient(fastapi_app)

        response = client.get("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert set(data["active_sessions"]) == {session1, session2}
        assert data["count"] == 2


@pytest.mark.unit
class TestDeleteSession:
    """Test cases for delete session endpoint."""

    def test_delete_session_success(self, fastapi_app, temp_dir):
        """Test successful session deletion."""
        session_id = "test-session-123"

        # Setup mock RAG system with cache directory
        mock_rag = Mock()
        mock_rag.cache_dir = str(temp_dir / "rag_session_test_cache")
        api.active_rag_systems[session_id] = mock_rag

        client = TestClient(fastapi_app)

        response = client.delete(f"/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"].lower()

        # Verify session was removed
        assert session_id not in api.active_rag_systems

    def test_delete_session_not_found(self, fastapi_app):
        """Test deletion of non-existent session."""
        client = TestClient(fastapi_app)

        response = client.delete("/sessions/non-existent-session")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_delete_session_cleanup_error(self, fastapi_app):
        """Test session deletion when cleanup fails."""
        session_id = "test-session-123"

        # Setup mock RAG system
        mock_rag = Mock()
        mock_rag.cache_dir = "/invalid/path"
        api.active_rag_systems[session_id] = mock_rag

        client = TestClient(fastapi_app)

        response = client.delete(f"/sessions/{session_id}")

        # Should still succeed even if cleanup fails
        assert response.status_code == 200
        assert session_id not in api.active_rag_systems

    def test_delete_all_sessions_success(self, fastapi_app, temp_dir):
        """Test successful deletion of all sessions."""
        # Add multiple sessions
        session1 = "session-1"
        session2 = "session-2"

        mock_rag1 = Mock()
        mock_rag1.cache_dir = str(temp_dir / "rag_session_1_cache")
        mock_rag2 = Mock()
        mock_rag2.cache_dir = str(temp_dir / "rag_session_2_cache")

        api.active_rag_systems[session1] = mock_rag1
        api.active_rag_systems[session2] = mock_rag2

        client = TestClient(fastapi_app)

        response = client.delete("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "all sessions deleted" in data["message"].lower()

        # Verify all sessions were removed
        assert len(api.active_rag_systems) == 0

    def test_delete_all_sessions_empty(self, fastapi_app):
        """Test deletion of all sessions when none exist."""
        client = TestClient(fastapi_app)

        response = client.delete("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "all sessions deleted" in data["message"].lower()


@pytest.mark.unit
class TestActiveRAGSystemsLifecycle:
    """Test cases for active_rag_systems lifecycle management."""

    def test_active_rag_systems_initialization(self):
        """Test that active_rag_systems is properly initialized."""
        assert hasattr(api, "active_rag_systems")
        assert isinstance(api.active_rag_systems, dict)

    def test_session_lifecycle_complete(self, fastapi_app, temp_dir):
        """Test complete session lifecycle: create, use, delete."""
        client = TestClient(fastapi_app)

        # 1. Upload PDF to create session
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with open(pdf_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}

            with patch("api.OptimizedRAGSystem") as mock_rag_class:
                mock_rag = Mock()
                mock_rag.documents = [Mock()]
                mock_rag.timing_stats = {"Document Processing": 1.0}
                mock_rag.query.return_value = {"answer": "Test answer"}
                mock_rag_class.return_value = mock_rag

                upload_response = client.post("/upload", files=files)

        assert upload_response.status_code == 200
        session_id = upload_response.json()["session_id"]
        assert session_id in api.active_rag_systems

        # 2. Query the session
        query_payload = {"question": "What is this about?", "session_id": session_id}
        query_response = client.post("/query", json=query_payload)

        assert query_response.status_code == 200
        assert query_response.json()["answer"] == "Test answer"

        # 3. Delete the session
        delete_response = client.delete(f"/sessions/{session_id}")

        assert delete_response.status_code == 200
        assert session_id not in api.active_rag_systems

    def test_multiple_sessions_isolation(self, fastapi_app, temp_dir):
        """Test that multiple sessions are properly isolated."""
        client = TestClient(fastapi_app)

        # Create two sessions
        pdf1 = temp_dir / "test1.pdf"
        pdf2 = temp_dir / "test2.pdf"
        pdf1.write_bytes(b"content 1")
        pdf2.write_bytes(b"content 2")

        session_ids = []

        for pdf_path in [pdf1, pdf2]:
            with open(pdf_path, "rb") as f:
                files = {"files": (pdf_path.name, f, "application/pdf")}

                with patch("api.OptimizedRAGSystem") as mock_rag_class:
                    mock_rag = Mock()
                    mock_rag.documents = [Mock()]
                    mock_rag.timing_stats = {"Document Processing": 1.0}
                    mock_rag.query.return_value = {
                        "answer": f"Answer for {pdf_path.name}"
                    }
                    mock_rag_class.return_value = mock_rag

                    response = client.post("/upload", files=files)
                    session_ids.append(response.json()["session_id"])

        # Verify both sessions exist
        assert len(api.active_rag_systems) == 2
        assert all(sid in api.active_rag_systems for sid in session_ids)

        # Query each session independently
        for i, session_id in enumerate(session_ids):
            query_payload = {
                "question": "What is this about?",
                "session_id": session_id,
            }
            response = client.post("/query", json=query_payload)
            assert response.status_code == 200
            assert f"Answer for test{i + 1}.pdf" in response.json()["answer"]

    def test_session_cleanup_on_upload_error(self, fastapi_app, temp_dir):
        """Test that sessions are cleaned up when upload fails."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        client = TestClient(fastapi_app)

        with open(pdf_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}

            with patch("api.OptimizedRAGSystem") as mock_rag_class:
                mock_rag = Mock()
                mock_rag.initialize.side_effect = Exception("Upload failed")
                mock_rag_class.return_value = mock_rag

                response = client.post("/upload", files=files)

        assert response.status_code == 500
        # Verify no sessions were left behind
        assert len(api.active_rag_systems) == 0

    def test_timing_aggregation(self, fastapi_app, temp_dir):
        """Test that timing stats are properly aggregated."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        client = TestClient(fastapi_app)

        with open(pdf_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}

            with patch("api.OptimizedRAGSystem") as mock_rag_class:
                mock_rag = Mock()
                mock_rag.documents = [Mock()]
                # Multiple timing stats
                mock_rag.timing_stats = {
                    "Document Processing": 1.5,
                    "MLX Model Loading": 2.0,
                    "Chain Setup": 0.5,
                }
                mock_rag_class.return_value = mock_rag

                response = client.post("/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        # Should sum all timing stats: 1.5 + 2.0 + 0.5 = 4.0
        assert data["processing_time"] == 4.0
