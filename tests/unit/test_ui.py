"""
Unit tests for Streamlit utilities in ui.py.

These tests mock Streamlit components and requests to test the utility functions
without requiring a running Streamlit app or API server.
"""

import pytest
import requests
from unittest.mock import Mock
import responses

# Import the functions we're testing
from ui import check_api_health, upload_pdfs, query_document


@pytest.mark.unit
class TestCheckAPIHealth:
    """Test cases for check_api_health function."""

    def test_check_api_health_success(self):
        """Test successful API health check."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"status": "healthy", "active_sessions": 2},
                status=200,
            )

            is_healthy, status = check_api_health()

            assert is_healthy is True
            assert "✅ API is healthy" in status
            assert "2 active sessions" in status

    def test_check_api_health_api_error(self):
        """Test API health check when API returns error."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"error": "Service unavailable"},
                status=503,
            )

            is_healthy, status = check_api_health()

            assert is_healthy is False
            assert "❌ API health check failed: 503" in status

    def test_check_api_health_network_error(self):
        """Test API health check when network request fails."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            is_healthy, status = check_api_health()

            assert is_healthy is False
            assert "❌ Cannot connect to API" in status
            assert "Connection failed" in status
            assert "Make sure the API server is running" in status

    def test_check_api_health_timeout(self):
        """Test API health check with timeout."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                body=requests.exceptions.Timeout("Request timeout"),
            )

            is_healthy, status = check_api_health()

            assert is_healthy is False
            assert "❌ Cannot connect to API" in status
            assert "Request timeout" in status


@pytest.mark.unit
class TestUploadPDFs:
    """Test cases for upload_pdfs function."""

    def test_upload_pdfs_success_single_file(self):
        """Test successful upload of single PDF file."""
        # Create mock uploaded file
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={
                    "session_id": "test-session-123",
                    "document_count": 5,
                    "processing_time": 2.5,
                },
                status=200,
            )

            session_id, status = upload_pdfs([mock_file])

            assert session_id == "test-session-123"
            assert "✅ Upload successful!" in status
            assert "Files: 1" in status
            assert "Chunks: 5" in status
            assert "Time: 2.50s" in status

    def test_upload_pdfs_success_multiple_files(self):
        """Test successful upload of multiple PDF files."""
        # Create mock uploaded files
        mock_file1 = Mock()
        mock_file1.name = "test1.pdf"
        mock_file1.getvalue.return_value = b"dummy pdf content 1"

        mock_file2 = Mock()
        mock_file2.name = "test2.pdf"
        mock_file2.getvalue.return_value = b"dummy pdf content 2"

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={
                    "session_id": "test-session-456",
                    "document_count": 8,
                    "processing_time": 3.2,
                },
                status=200,
            )

            session_id, status = upload_pdfs([mock_file1, mock_file2])

            assert session_id == "test-session-456"
            assert "✅ Upload successful!" in status
            assert "Files: 2" in status
            assert "Chunks: 8" in status
            assert "Time: 3.20s" in status

    def test_upload_pdfs_api_error(self):
        """Test upload when API returns error."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={"error": "Invalid file type"},
                status=400,
            )

            session_id, status = upload_pdfs([mock_file])

            assert session_id is None
            assert "❌ Upload failed: 400" in status
            assert "Invalid file type" in status

    def test_upload_pdfs_network_error(self):
        """Test upload when network request fails."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            session_id, status = upload_pdfs([mock_file])

            assert session_id is None
            assert "❌ Upload error" in status
            assert "Connection failed" in status

    def test_upload_pdfs_timeout(self):
        """Test upload with timeout."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                body=requests.exceptions.Timeout("Request timeout"),
            )

            session_id, status = upload_pdfs([mock_file])

            assert session_id is None
            assert "❌ Upload error" in status
            assert "Request timeout" in status

    def test_upload_pdfs_empty_files_list(self):
        """Test upload with empty files list."""
        session_id, status = upload_pdfs([])

        assert session_id is None
        assert "❌ Upload error" in status

    def test_upload_pdfs_file_without_name(self):
        """Test upload with file missing name."""
        mock_file = Mock()
        mock_file.name = None
        mock_file.getvalue.return_value = b"dummy pdf content"

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={"error": "Missing filename"},
                status=400,
            )

            session_id, status = upload_pdfs([mock_file])

            assert session_id is None
            assert "❌ Upload failed: 400" in status


@pytest.mark.unit
class TestQueryDocument:
    """Test cases for query_document function."""

    def test_query_document_success(self):
        """Test successful document query."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                json={
                    "answer": "This is a test answer to your question.",
                    "session_id": "test-session-123",
                    "processing_time": 1.5,
                },
                status=200,
            )

            answer = query_document("What is this document about?", "test-session-123")

            assert answer == "This is a test answer to your question."

    def test_query_document_api_error(self):
        """Test query when API returns error."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                json={"error": "Session not found"},
                status=404,
            )

            answer = query_document("What is this about?", "invalid-session")

            assert "❌ Query failed: 404" in answer
            assert "Session not found" in answer

    def test_query_document_network_error(self):
        """Test query when network request fails."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            answer = query_document("What is this about?", "test-session")

            assert "❌ Query error" in answer
            assert "Connection failed" in answer

    def test_query_document_timeout(self):
        """Test query with timeout."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                body=requests.exceptions.Timeout("Request timeout"),
            )

            answer = query_document("What is this about?", "test-session")

            assert "❌ Query error" in answer
            assert "Request timeout" in answer

    def test_query_document_empty_response(self):
        """Test query with empty response."""
        with responses.RequestsMock() as rsps:
            rsps.add(responses.POST, "http://localhost:8000/query", json={}, status=200)

            answer = query_document("What is this about?", "test-session")

            # Should handle missing answer gracefully
            assert answer is not None

    def test_query_document_malformed_response(self):
        """Test query with malformed response."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                body="Invalid JSON response",
                status=200,
            )

            answer = query_document("What is this about?", "test-session")

            assert "❌ Query error" in answer


@pytest.mark.unit
class TestUIUtilitiesIntegration:
    """Integration tests for UI utility functions."""

    def test_complete_workflow_success(self):
        """Test complete workflow: health check, upload, query."""
        # Mock uploaded file
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with responses.RequestsMock() as rsps:
            # Mock health check
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"status": "healthy", "active_sessions": 0},
                status=200,
            )

            # Mock upload
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={
                    "session_id": "test-session-123",
                    "document_count": 3,
                    "processing_time": 2.0,
                },
                status=200,
            )

            # Mock query
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                json={
                    "answer": "This document is about testing.",
                    "session_id": "test-session-123",
                    "processing_time": 1.2,
                },
                status=200,
            )

            # Test health check
            is_healthy, health_status = check_api_health()
            assert is_healthy is True

            # Test upload
            session_id, upload_status = upload_pdfs([mock_file])
            assert session_id == "test-session-123"
            assert "✅ Upload successful!" in upload_status

            # Test query
            answer = query_document("What is this document about?", session_id)
            assert answer == "This document is about testing."

    def test_error_handling_workflow(self):
        """Test error handling in complete workflow."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with responses.RequestsMock() as rsps:
            # Mock health check failure
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            # Mock upload failure
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={"error": "Invalid file format"},
                status=400,
            )

            # Test health check fails
            is_healthy, health_status = check_api_health()
            assert is_healthy is False
            assert "❌ Cannot connect to API" in health_status

            # Test upload fails
            session_id, upload_status = upload_pdfs([mock_file])
            assert session_id is None
            assert "❌ Upload failed: 400" in upload_status

            # Test query fails without valid session
            with responses.RequestsMock() as query_rsps:
                query_rsps.add(
                    responses.POST,
                    "http://localhost:8000/query",
                    json={"error": "No active sessions"},
                    status=400,
                )

                answer = query_document("What is this about?", "invalid-session")
                assert "❌ Query failed: 400" in answer

    def test_multiple_file_upload_workflow(self):
        """Test workflow with multiple file upload."""
        # Create multiple mock files
        mock_files = []
        for i in range(3):
            mock_file = Mock()
            mock_file.name = f"test{i + 1}.pdf"
            mock_file.getvalue.return_value = f"dummy pdf content {i + 1}".encode()
            mock_files.append(mock_file)

        with responses.RequestsMock() as rsps:
            # Mock upload
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={
                    "session_id": "multi-session-123",
                    "document_count": 12,
                    "processing_time": 4.5,
                },
                status=200,
            )

            # Mock query
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                json={
                    "answer": "These documents are about testing multiple files.",
                    "session_id": "multi-session-123",
                    "processing_time": 1.8,
                },
                status=200,
            )

            # Test upload
            session_id, upload_status = upload_pdfs(mock_files)
            assert session_id == "multi-session-123"
            assert "Files: 3" in upload_status
            assert "Chunks: 12" in upload_status

            # Test query
            answer = query_document("What are these documents about?", session_id)
            assert answer == "These documents are about testing multiple files."

    def test_timeout_handling_workflow(self):
        """Test timeout handling in workflow."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with responses.RequestsMock() as rsps:
            # Mock health check timeout
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                body=requests.exceptions.Timeout("Health check timeout"),
            )

            # Mock upload timeout
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                body=requests.exceptions.Timeout("Upload timeout"),
            )

            # Mock query timeout
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                body=requests.exceptions.Timeout("Query timeout"),
            )

            # Test all operations handle timeouts gracefully
            is_healthy, health_status = check_api_health()
            assert is_healthy is False
            assert "Request timeout" in health_status

            session_id, upload_status = upload_pdfs([mock_file])
            assert session_id is None
            assert "Request timeout" in upload_status

            answer = query_document("What is this about?", "test-session")
            assert "Request timeout" in answer

    def test_response_formatting_consistency(self):
        """Test that response formatting is consistent."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with responses.RequestsMock() as rsps:
            # Mock successful responses
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"status": "healthy", "active_sessions": 1},
                status=200,
            )

            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={
                    "session_id": "test-session-123",
                    "document_count": 5,
                    "processing_time": 2.5,
                },
                status=200,
            )

            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                json={
                    "answer": "Test answer",
                    "session_id": "test-session-123",
                    "processing_time": 1.5,
                },
                status=200,
            )

            # Test response formats
            is_healthy, health_status = check_api_health()
            assert health_status.startswith("✅")
            assert "API is healthy" in health_status
            assert "1 active sessions" in health_status

            session_id, upload_status = upload_pdfs([mock_file])
            assert upload_status.startswith("✅")
            assert "Upload successful!" in upload_status
            assert "Files: 1" in upload_status
            assert "Chunks: 5" in upload_status
            assert "Time: 2.50s" in upload_status

            answer = query_document("Test question", session_id)
            assert answer == "Test answer"
