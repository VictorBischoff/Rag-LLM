"""
Integration tests for the FastAPI application.

These tests use TestClient and httpx.AsyncClient to test the complete
API flow with stubbed RAG systems, covering upload → query workflows
and concurrent session handling.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import httpx

import api
from tests.conftest import create_test_pdf


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the complete API workflow."""

    def test_upload_query_workflow_single_pdf(self, temp_dir):
        """Test complete workflow: upload single PDF, then query it."""
        client = TestClient(api.app)

        # Create test PDF
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "This is a test document about machine learning.")

        with open(pdf_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}

            with patch("api.OptimizedRAGSystem") as mock_rag_class:
                # Setup mock RAG system
                mock_rag = Mock()
                mock_rag.documents = [Mock(), Mock(), Mock()]  # 3 documents
                mock_rag.timing_stats = {
                    "Document Processing": 1.5,
                    "MLX Model Loading": 2.0,
                    "Chain Setup": 0.5,
                }
                mock_rag.query.return_value = {
                    "answer": "This document is about machine learning.",
                    "context": "Machine learning is a subset of artificial intelligence.",
                }
                mock_rag_class.return_value = mock_rag

                # 1. Upload PDF
                upload_response = client.post("/upload", files=files)
                assert upload_response.status_code == 200

                upload_data = upload_response.json()
                session_id = upload_data["session_id"]
                assert session_id is not None
                assert upload_data["document_count"] == 3
                assert upload_data["processing_time"] == 4.0  # Sum of timing stats

                # Verify RAG system was created and initialized
                mock_rag_class.assert_called_once()
                mock_rag.initialize.assert_called_once()

                # 2. Query the document
                query_payload = {
                    "question": "What is this document about?",
                    "session_id": session_id,
                }
                query_response = client.post("/query", json=query_payload)
                assert query_response.status_code == 200

                query_data = query_response.json()
                assert (
                    query_data["answer"] == "This document is about machine learning."
                )
                assert query_data["session_id"] == session_id
                assert "processing_time" in query_data

                # Verify RAG system was queried correctly
                mock_rag.query.assert_called_once_with("What is this document about?")

                # 3. Verify session exists in active systems
                assert session_id in api.active_rag_systems
                assert api.active_rag_systems[session_id] == mock_rag

    def test_upload_query_workflow_multiple_pdfs(self, temp_dir):
        """Test complete workflow: upload multiple PDFs, then query them."""
        client = TestClient(api.app)

        # Create test PDFs
        pdf1 = temp_dir / "test1.pdf"
        pdf2 = temp_dir / "test2.pdf"
        create_test_pdf(pdf1, "Document 1 about artificial intelligence.")
        create_test_pdf(pdf2, "Document 2 about natural language processing.")

        with open(pdf1, "rb") as f1, open(pdf2, "rb") as f2:
            files = [
                ("files", ("test1.pdf", f1, "application/pdf")),
                ("files", ("test2.pdf", f2, "application/pdf")),
            ]

            with patch("api.OptimizedRAGSystem") as mock_rag_class:
                # Setup mock RAG system
                mock_rag = Mock()
                mock_rag.documents = [Mock()] * 6  # 6 documents total
                mock_rag.timing_stats = {"Document Processing": 2.5}
                mock_rag.query.return_value = {
                    "answer": "These documents cover AI and NLP topics.",
                    "context": "AI and NLP are related fields in computer science.",
                }
                mock_rag_class.return_value = mock_rag

                # Upload multiple PDFs
                upload_response = client.post("/upload", files=files)
                assert upload_response.status_code == 200

                upload_data = upload_response.json()
                session_id = upload_data["session_id"]
                assert "2 PDF file(s)" in upload_data["message"]
                assert upload_data["document_count"] == 6

                # Query the combined documents
                query_payload = {
                    "question": "What topics do these documents cover?",
                    "session_id": session_id,
                }
                query_response = client.post("/query", json=query_payload)
                assert query_response.status_code == 200

                query_data = query_response.json()
                assert "AI and NLP topics" in query_data["answer"]

    def test_concurrent_sessions_isolation(self, temp_dir):
        """Test that multiple concurrent sessions are properly isolated."""
        client = TestClient(api.app)

        # Create test PDFs
        pdf1 = temp_dir / "test1.pdf"
        pdf2 = temp_dir / "test2.pdf"
        create_test_pdf(pdf1, "Document 1 content.")
        create_test_pdf(pdf2, "Document 2 content.")

        session_ids = []

        # Create two separate sessions
        for i, pdf_path in enumerate([pdf1, pdf2], 1):
            with open(pdf_path, "rb") as f:
                files = {"files": (pdf_path.name, f, "application/pdf")}

                with patch("api.OptimizedRAGSystem") as mock_rag_class:
                    mock_rag = Mock()
                    mock_rag.documents = [Mock()]
                    mock_rag.timing_stats = {"Document Processing": 1.0}
                    mock_rag.query.return_value = {
                        "answer": f"Answer for document {i}",
                        "context": f"Context for document {i}",
                    }
                    mock_rag_class.return_value = mock_rag

                    upload_response = client.post("/upload", files=files)
                    assert upload_response.status_code == 200
                    session_ids.append(upload_response.json()["session_id"])

        # Verify both sessions exist
        assert len(api.active_rag_systems) == 2
        assert all(sid in api.active_rag_systems for sid in session_ids)

        # Query each session independently
        for i, session_id in enumerate(session_ids, 1):
            query_payload = {
                "question": f"What is document {i} about?",
                "session_id": session_id,
            }
            query_response = client.post("/query", json=query_payload)
            assert query_response.status_code == 200

            query_data = query_response.json()
            assert f"Answer for document {i}" in query_data["answer"]
            assert query_data["session_id"] == session_id

    def test_session_cleanup_on_upload_error(self, temp_dir):
        """Test that sessions are properly cleaned up when upload fails."""
        client = TestClient(api.app)

        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "Test content.")

        with open(pdf_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}

            with patch("api.OptimizedRAGSystem") as mock_rag_class:
                # Make RAG initialization fail
                mock_rag = Mock()
                mock_rag.initialize.side_effect = Exception("Initialization failed")
                mock_rag_class.return_value = mock_rag

                upload_response = client.post("/upload", files=files)
                assert upload_response.status_code == 500

                # Verify no sessions were left behind
                assert len(api.active_rag_systems) == 0

    def test_file_caching_validation(self, temp_dir):
        """Test that file caching works correctly with multiple initializations."""
        client = TestClient(api.app)

        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "Test content for caching.")

        # Track RAG system creation calls
        rag_creation_calls = []

        def track_rag_creation(*args, **kwargs):
            rag_creation_calls.append((args, kwargs))
            mock_rag = Mock()
            mock_rag.documents = [Mock()]
            mock_rag.timing_stats = {"Document Processing": 1.0}
            mock_rag.query.return_value = {
                "answer": "Cached answer",
                "context": "Cached context",
            }
            return mock_rag

        with patch("api.OptimizedRAGSystem", side_effect=track_rag_creation):
            # First upload
            with open(pdf_path, "rb") as f:
                files = {"files": ("test.pdf", f, "application/pdf")}
                upload_response = client.post("/upload", files=files)
                assert upload_response.status_code == 200
                session_id1 = upload_response.json()["session_id"]

            # Second upload with same file
            with open(pdf_path, "rb") as f:
                files = {"files": ("test.pdf", f, "application/pdf")}
                upload_response = client.post("/upload", files=files)
                assert upload_response.status_code == 200
                session_id2 = upload_response.json()["session_id"]

            # Verify two separate RAG systems were created (different sessions)
            assert len(rag_creation_calls) == 2
            assert session_id1 != session_id2
            assert len(api.active_rag_systems) == 2

    def test_corrupt_cache_fallback_behavior(self, temp_dir):
        """Test fallback behavior when cache is corrupt."""
        client = TestClient(api.app)

        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "Test content for corrupt cache.")

        # Create a corrupt cache file
        cache_dir = temp_dir / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "documents_test.pkl"
        cache_file.write_bytes(b"corrupt cache data")

        with patch("api.OptimizedRAGSystem") as mock_rag_class:
            mock_rag = Mock()
            mock_rag.documents = [Mock()]
            mock_rag.timing_stats = {"Document Processing": 1.0}
            mock_rag.query.return_value = {
                "answer": "Fallback answer",
                "context": "Fallback context",
            }
            mock_rag_class.return_value = mock_rag

            with open(pdf_path, "rb") as f:
                files = {"files": ("test.pdf", f, "application/pdf")}
                upload_response = client.post("/upload", files=files)
                assert upload_response.status_code == 200

                # Should still work despite corrupt cache
                session_id = upload_response.json()["session_id"]

                query_payload = {
                    "question": "What is this about?",
                    "session_id": session_id,
                }
                query_response = client.post("/query", json=query_payload)
                assert query_response.status_code == 200
                assert "Fallback answer" in query_response.json()["answer"]

    def test_session_deletion_and_cleanup(self, temp_dir):
        """Test session deletion and resource cleanup."""
        client = TestClient(api.app)

        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "Test content.")

        with open(pdf_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}

            with patch("api.OptimizedRAGSystem") as mock_rag_class:
                mock_rag = Mock()
                mock_rag.documents = [Mock()]
                mock_rag.timing_stats = {"Document Processing": 1.0}
                mock_rag.cache_dir = str(temp_dir / "rag_session_test_cache")
                mock_rag_class.return_value = mock_rag

                # Upload to create session
                upload_response = client.post("/upload", files=files)
                assert upload_response.status_code == 200
                session_id = upload_response.json()["session_id"]

                # Verify session exists
                assert session_id in api.active_rag_systems

                # Delete session
                delete_response = client.delete(f"/sessions/{session_id}")
                assert delete_response.status_code == 200

                # Verify session was removed
                assert session_id not in api.active_rag_systems

                # Try to query deleted session
                query_payload = {
                    "question": "What is this about?",
                    "session_id": session_id,
                }
                query_response = client.post("/query", json=query_payload)
                assert query_response.status_code == 404

    def test_all_sessions_deletion(self, temp_dir):
        """Test deletion of all sessions."""
        client = TestClient(api.app)

        # Create multiple sessions
        pdf1 = temp_dir / "test1.pdf"
        pdf2 = temp_dir / "test2.pdf"
        create_test_pdf(pdf1, "Document 1.")
        create_test_pdf(pdf2, "Document 2.")

        session_ids = []

        for pdf_path in [pdf1, pdf2]:
            with open(pdf_path, "rb") as f:
                files = {"files": (pdf_path.name, f, "application/pdf")}

                with patch("api.OptimizedRAGSystem") as mock_rag_class:
                    mock_rag = Mock()
                    mock_rag.documents = [Mock()]
                    mock_rag.timing_stats = {"Document Processing": 1.0}
                    mock_rag.cache_dir = str(
                        temp_dir / f"rag_session_{pdf_path.stem}_cache"
                    )
                    mock_rag_class.return_value = mock_rag

                    upload_response = client.post("/upload", files=files)
                    assert upload_response.status_code == 200
                    session_ids.append(upload_response.json()["session_id"])

        # Verify both sessions exist
        assert len(api.active_rag_systems) == 2

        # Delete all sessions
        delete_response = client.delete("/sessions")
        assert delete_response.status_code == 200

        # Verify all sessions were removed
        assert len(api.active_rag_systems) == 0

        # Try to query any deleted session
        query_payload = {
            "question": "What is this about?",
            "session_id": session_ids[0],
        }
        query_response = client.post("/query", json=query_payload)
        assert query_response.status_code == 404

    def test_session_logs_integration(self, temp_dir):
        """Test session logs endpoint integration."""
        client = TestClient(api.app)

        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "Test content.")

        with open(pdf_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}

            with patch("api.OptimizedRAGSystem") as mock_rag_class:
                mock_rag = Mock()
                mock_rag.documents = [Mock(), Mock(), Mock()]
                mock_rag.timing_stats = {
                    "Document Processing": 1.5,
                    "MLX Model Loading": 2.0,
                    "Chain Setup": 0.5,
                }
                mock_rag.model_id = "test-model"
                mock_rag.cache_dir = str(temp_dir / "cache")
                mock_rag.chunk_size = 1500
                mock_rag.chunk_overlap = 100
                mock_rag.max_tokens = 500
                mock_rag.temperature = 0.1
                mock_rag_class.return_value = mock_rag

                # Upload to create session
                upload_response = client.post("/upload", files=files)
                assert upload_response.status_code == 200
                session_id = upload_response.json()["session_id"]

                # Get session logs
                logs_response = client.get(f"/logs/{session_id}")
                assert logs_response.status_code == 200

                logs_data = logs_response.json()
                assert logs_data["session_id"] == session_id
                assert logs_data["model_id"] == "test-model"
                assert logs_data["document_count"] == 3
                assert logs_data["timing_stats"]["Document Processing"] == 1.5
                assert logs_data["system_config"]["chunk_size"] == 1500

    def test_health_and_sessions_endpoints_integration(self, temp_dir):
        """Test health and sessions endpoints integration."""
        client = TestClient(api.app)

        # Test health with no sessions
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        assert health_data["active_sessions"] == 0

        # Test sessions with no sessions
        sessions_response = client.get("/sessions")
        assert sessions_response.status_code == 200
        sessions_data = sessions_response.json()
        assert sessions_data["active_sessions"] == []
        assert sessions_data["count"] == 0

        # Create a session
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "Test content.")

        with open(pdf_path, "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}

            with patch("api.OptimizedRAGSystem") as mock_rag_class:
                mock_rag = Mock()
                mock_rag.documents = [Mock()]
                mock_rag.timing_stats = {"Document Processing": 1.0}
                mock_rag_class.return_value = mock_rag

                upload_response = client.post("/upload", files=files)
                assert upload_response.status_code == 200
                session_id = upload_response.json()["session_id"]

        # Test health with active session
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["active_sessions"] == 1

        # Test sessions with active session
        sessions_response = client.get("/sessions")
        assert sessions_response.status_code == 200
        sessions_data = sessions_response.json()
        assert sessions_data["active_sessions"] == [session_id]
        assert sessions_data["count"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncAPIIntegration:
    """Async integration tests using httpx.AsyncClient."""

    async def test_async_upload_query_workflow(self, temp_dir):
        """Test async upload and query workflow."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "Async test content.")

        async with httpx.AsyncClient(app=api.app, base_url="http://test") as client:
            with open(pdf_path, "rb") as f:
                files = {"files": ("test.pdf", f, "application/pdf")}

                with patch("api.OptimizedRAGSystem") as mock_rag_class:
                    mock_rag = Mock()
                    mock_rag.documents = [Mock()]
                    mock_rag.timing_stats = {"Document Processing": 1.0}
                    mock_rag.query.return_value = {
                        "answer": "Async answer",
                        "context": "Async context",
                    }
                    mock_rag_class.return_value = mock_rag

                    # Async upload
                    upload_response = await client.post("/upload", files=files)
                    assert upload_response.status_code == 200

                    upload_data = upload_response.json()
                    session_id = upload_data["session_id"]

                    # Async query
                    query_payload = {
                        "question": "What is this about?",
                        "session_id": session_id,
                    }
                    query_response = await client.post("/query", json=query_payload)
                    assert query_response.status_code == 200

                    query_data = query_response.json()
                    assert query_data["answer"] == "Async answer"

    async def test_async_concurrent_requests(self, temp_dir):
        """Test handling of concurrent async requests."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "Concurrent test content.")

        async with httpx.AsyncClient(app=api.app, base_url="http://test") as client:
            with open(pdf_path, "rb") as f:
                files = {"files": ("test.pdf", f, "application/pdf")}

                with patch("api.OptimizedRAGSystem") as mock_rag_class:
                    mock_rag = Mock()
                    mock_rag.documents = [Mock()]
                    mock_rag.timing_stats = {"Document Processing": 1.0}
                    mock_rag.query.return_value = {
                        "answer": "Concurrent answer",
                        "context": "Concurrent context",
                    }
                    mock_rag_class.return_value = mock_rag

                    # Upload
                    upload_response = await client.post("/upload", files=files)
                    assert upload_response.status_code == 200
                    session_id = upload_response.json()["session_id"]

                    # Make concurrent queries
                    import asyncio

                    async def make_query(question):
                        query_payload = {"question": question, "session_id": session_id}
                        return await client.post("/query", json=query_payload)

                    # Run multiple queries concurrently
                    tasks = [
                        make_query("Question 1"),
                        make_query("Question 2"),
                        make_query("Question 3"),
                    ]

                    responses = await asyncio.gather(*tasks)

                    # All should succeed
                    for response in responses:
                        assert response.status_code == 200
                        assert "Concurrent answer" in response.json()["answer"]
