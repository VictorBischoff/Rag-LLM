"""
Unit tests for client_example.py helper functions.

These tests use responses/requests-mock to test HTTP client functionality
without requiring a running API server.
"""

import pytest
import requests
from unittest.mock import patch
import responses

# Import the functions we're testing
from client_example import (
    upload_pdf,
    query_document,
    list_sessions,
    delete_session,
    check_api_health,
    interactive_mode,
)


@pytest.mark.unit
class TestUploadPDF:
    """Test cases for upload_pdf function."""

    def test_upload_pdf_success(self, temp_dir):
        """Test successful PDF upload."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={
                    "message": "Successfully uploaded and processed 1 PDF file(s)",
                    "session_id": "test-session-123",
                    "document_count": 5,
                    "processing_time": 2.5,
                },
                status=200,
            )

            session_id = upload_pdf(str(pdf_path))

            assert session_id == "test-session-123"

    def test_upload_pdf_file_not_found(self, temp_dir):
        """Test upload with non-existent file."""
        missing_pdf = temp_dir / "missing.pdf"

        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            upload_pdf(str(missing_pdf))

    def test_upload_pdf_api_error(self, temp_dir):
        """Test upload when API returns error."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={"error": "Invalid file type"},
                status=400,
            )

            session_id = upload_pdf(str(pdf_path))

            assert session_id is None

    def test_upload_pdf_network_error(self, temp_dir):
        """Test upload when network request fails."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            session_id = upload_pdf(str(pdf_path))

            assert session_id is None


@pytest.mark.unit
class TestQueryDocument:
    """Test cases for query_document function."""

    def test_query_document_success_with_session_id(self):
        """Test successful query with specific session ID."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                json={
                    "answer": "This is a test answer",
                    "session_id": "test-session-123",
                    "processing_time": 1.5,
                },
                status=200,
            )

            answer = query_document("What is this about?", "test-session-123")

            assert answer == "This is a test answer"

    def test_query_document_success_without_session_id(self):
        """Test successful query without session ID (uses first available)."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                json={
                    "answer": "This is a test answer",
                    "session_id": "auto-session-456",
                    "processing_time": 2.0,
                },
                status=200,
            )

            answer = query_document("What is this about?")

            assert answer == "This is a test answer"

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

            assert answer is None

    def test_query_document_network_error(self):
        """Test query when network request fails."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                body=requests.exceptions.Timeout("Request timeout"),
            )

            answer = query_document("What is this about?")

            assert answer is None


@pytest.mark.unit
class TestListSessions:
    """Test cases for list_sessions function."""

    def test_list_sessions_success(self):
        """Test successful session listing."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={
                    "active_sessions": ["session-1", "session-2", "session-3"],
                    "count": 3,
                },
                status=200,
            )

            sessions = list_sessions()

            assert sessions == ["session-1", "session-2", "session-3"]

    def test_list_sessions_empty(self):
        """Test session listing when no sessions exist."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": [], "count": 0},
                status=200,
            )

            sessions = list_sessions()

            assert sessions == []

    def test_list_sessions_api_error(self):
        """Test session listing when API returns error."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"error": "Internal server error"},
                status=500,
            )

            sessions = list_sessions()

            assert sessions == []

    def test_list_sessions_network_error(self):
        """Test session listing when network request fails."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            sessions = list_sessions()

            assert sessions == []


@pytest.mark.unit
class TestDeleteSession:
    """Test cases for delete_session function."""

    def test_delete_session_success(self):
        """Test successful session deletion."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.DELETE,
                "http://localhost:8000/sessions/test-session-123",
                json={"message": "Session deleted successfully"},
                status=200,
            )

            # Should not raise an exception
            delete_session("test-session-123")

    def test_delete_session_not_found(self):
        """Test deletion of non-existent session."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.DELETE,
                "http://localhost:8000/sessions/invalid-session",
                json={"error": "Session not found"},
                status=404,
            )

            # Should not raise an exception, just print error
            delete_session("invalid-session")

    def test_delete_session_network_error(self):
        """Test session deletion when network request fails."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.DELETE,
                "http://localhost:8000/sessions/test-session-123",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            # Should not raise an exception, just print error
            delete_session("test-session-123")


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

            is_healthy = check_api_health()

            assert is_healthy is True

    def test_check_api_health_api_error(self):
        """Test API health check when API returns error."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"error": "Service unavailable"},
                status=503,
            )

            is_healthy = check_api_health()

            assert is_healthy is False

    def test_check_api_health_network_error(self):
        """Test API health check when network request fails."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            is_healthy = check_api_health()

            assert is_healthy is False

    def test_check_api_health_timeout(self):
        """Test API health check with timeout."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                body=requests.exceptions.Timeout("Request timeout"),
            )

            is_healthy = check_api_health()

            assert is_healthy is False


@pytest.mark.unit
class TestInteractiveMode:
    """Test cases for interactive_mode function."""

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_mode_quit_command(self, mock_print, mock_input):
        """Test interactive mode with quit command."""
        mock_input.side_effect = ["quit"]

        interactive_mode()

        mock_input.assert_called_once()
        mock_print.assert_any_call("👋 Goodbye!")

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_mode_sessions_command(self, mock_print, mock_input):
        """Test interactive mode with sessions command."""
        mock_input.side_effect = ["sessions", "quit"]

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": ["session-1", "session-2"], "count": 2},
                status=200,
            )

            interactive_mode()

        # Should have called list_sessions
        mock_print.assert_any_call("📋 Listing active sessions...")
        mock_print.assert_any_call("✅ Active sessions: 2")

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_mode_help_command(self, mock_print, mock_input):
        """Test interactive mode with help command."""
        mock_input.side_effect = ["help", "quit"]

        interactive_mode()

        mock_print.assert_any_call("Available commands:")
        mock_print.assert_any_call("  - Ask any question about the uploaded document")
        mock_print.assert_any_call("  - 'sessions' - List active sessions")
        mock_print.assert_any_call("  - 'quit' - Exit the program")

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_mode_query_processing(self, mock_print, mock_input):
        """Test interactive mode processes queries correctly."""
        mock_input.side_effect = ["What is this about?", "quit"]

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                json={
                    "answer": "This is a test answer",
                    "session_id": "test-session",
                    "processing_time": 1.5,
                },
                status=200,
            )

            interactive_mode()

        mock_print.assert_any_call("❓ Querying: What is this about?")
        mock_print.assert_any_call("✅ Query successful!")
        mock_print.assert_any_call("   Answer: This is a test answer")

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_mode_empty_input(self, mock_print, mock_input):
        """Test interactive mode with empty input."""
        mock_input.side_effect = ["", "quit"]

        interactive_mode()

        # Should not process empty input
        mock_print.assert_any_call("👋 Goodbye!")

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_mode_keyboard_interrupt(self, mock_print, mock_input):
        """Test interactive mode with keyboard interrupt."""
        mock_input.side_effect = KeyboardInterrupt()

        interactive_mode()

        mock_print.assert_any_call("👋 Goodbye!")

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_mode_query_error(self, mock_print, mock_input):
        """Test interactive mode when query fails."""
        mock_input.side_effect = ["What is this about?", "quit"]

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                json={"error": "No active sessions"},
                status=400,
            )

            interactive_mode()

        mock_print.assert_any_call("❌ Query failed: 400")
        mock_print.assert_any_call('   Error: {"error": "No active sessions"}')

    @patch("builtins.input")
    @patch("builtins.print")
    def test_interactive_mode_exception_handling(self, mock_print, mock_input):
        """Test interactive mode exception handling."""
        mock_input.side_effect = ["What is this about?", "quit"]

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                body=Exception("Unexpected error"),
            )

            interactive_mode()

        mock_print.assert_any_call("❌ Error: Unexpected error")


@pytest.mark.unit
class TestClientExampleIntegration:
    """Integration tests for client_example functions."""

    def test_upload_and_query_workflow(self, temp_dir):
        """Test complete workflow: upload PDF, then query it."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with responses.RequestsMock() as rsps:
            # Mock upload response
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={
                    "message": "Successfully uploaded and processed 1 PDF file(s)",
                    "session_id": "test-session-123",
                    "document_count": 5,
                    "processing_time": 2.5,
                },
                status=200,
            )

            # Mock query response
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                json={
                    "answer": "This document is about testing.",
                    "session_id": "test-session-123",
                    "processing_time": 1.5,
                },
                status=200,
            )

            # Upload PDF
            session_id = upload_pdf(str(pdf_path))
            assert session_id == "test-session-123"

            # Query document
            answer = query_document("What is this document about?", session_id)
            assert answer == "This document is about testing."

    def test_error_handling_workflow(self, temp_dir):
        """Test error handling in complete workflow."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with responses.RequestsMock() as rsps:
            # Mock upload failure
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={"error": "Invalid file format"},
                status=400,
            )

            # Upload should fail
            session_id = upload_pdf(str(pdf_path))
            assert session_id is None

            # Query should also fail without valid session
            with responses.RequestsMock() as query_rsps:
                query_rsps.add(
                    responses.POST,
                    "http://localhost:8000/query",
                    json={"error": "No active sessions"},
                    status=400,
                )

                answer = query_document("What is this about?")
                assert answer is None

    def test_session_management_workflow(self):
        """Test session management workflow: list, delete."""
        with responses.RequestsMock() as rsps:
            # Mock list sessions
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": ["session-1", "session-2"], "count": 2},
                status=200,
            )

            # Mock delete session
            rsps.add(
                responses.DELETE,
                "http://localhost:8000/sessions/session-1",
                json={"message": "Session deleted successfully"},
                status=200,
            )

            # List sessions
            sessions = list_sessions()
            assert sessions == ["session-1", "session-2"]

            # Delete session
            delete_session("session-1")
            # Should not raise exception
