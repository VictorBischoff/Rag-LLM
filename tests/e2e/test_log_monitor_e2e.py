"""
End-to-end tests for log monitor functionality.

These tests use ThreadPoolExecutor with a mocked HTTP server to test
log monitoring behavior when sessions appear/disappear.
"""

import pytest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from pytest_httpserver import HTTPServer

from log_monitor import (
    monitor_logs,
    show_current_logs,
)


@pytest.mark.e2e
class TestLogMonitorE2E:
    """End-to-end tests for log monitor functionality."""

    @pytest.fixture
    def mock_http_server(self):
        """Create a mock HTTP server for testing."""
        server = HTTPServer(host="127.0.0.1", port=8002)
        server.start()
        yield server
        server.stop()

    def test_log_monitor_session_detection(self, mock_http_server, sample_session_data):
        """Test that log monitor detects new and removed sessions."""
        # Setup mock responses
        mock_http_server.expect_request("/health").respond_with_json(
            {"status": "healthy", "active_sessions": 1}
        )

        # Start with no sessions
        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": [], "count": 0}
        )

        # Then add a session
        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["session-123"], "count": 1}
        )

        # Session logs
        mock_http_server.expect_request("/logs/session-123").respond_with_json(
            sample_session_data
        )

        # Monitor for a short time
        with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
            with patch("log_monitor.time.sleep") as mock_sleep:
                mock_sleep.side_effect = [None, None, KeyboardInterrupt()]

                with patch("builtins.print") as mock_print:
                    monitor_logs(interval=1)

        # Verify monitoring behavior
        print_calls = [str(call) for call in mock_print.call_args_list]
        printed_text = " ".join(print_calls)

        # Should detect new session
        assert any("New session(s)" in call for call in print_calls)

    def test_log_monitor_api_unhealthy_handling(self, mock_http_server):
        """Test log monitor behavior when API is unhealthy."""
        # Mock API as unhealthy
        mock_http_server.expect_request("/health").respond_with_data(
            "Service Unavailable", status=503
        )

        with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
            with patch("log_monitor.time.sleep") as mock_sleep:
                mock_sleep.side_effect = [None, KeyboardInterrupt()]

                with patch("builtins.print") as mock_print:
                    monitor_logs(interval=1)

        # Should print API not responding message
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("API not responding" in call for call in print_calls)

    def test_log_monitor_session_removal_detection(
        self, mock_http_server, sample_session_data
    ):
        """Test that log monitor detects when sessions are removed."""
        # Setup mock responses
        mock_http_server.expect_request("/health").respond_with_json(
            {"status": "healthy", "active_sessions": 0}
        )

        # Start with a session
        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["session-123"], "count": 1}
        )

        # Then remove the session
        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": [], "count": 0}
        )

        with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
            with patch("log_monitor.time.sleep") as mock_sleep:
                mock_sleep.side_effect = [None, None, KeyboardInterrupt()]

                with patch("builtins.print") as mock_print:
                    monitor_logs(interval=1)

        # Should detect removed session
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Removed session(s)" in call for call in print_calls)

    def test_show_current_logs_single_session(
        self, mock_http_server, sample_session_data
    ):
        """Test showing current logs for a single session."""
        # Setup mock responses
        mock_http_server.expect_request("/health").respond_with_json(
            {"status": "healthy", "active_sessions": 1}
        )

        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["session-123"], "count": 1}
        )

        mock_http_server.expect_request("/logs/session-123").respond_with_json(
            sample_session_data
        )

        with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
            with patch("builtins.print") as mock_print:
                show_current_logs()

        # Verify logs were displayed
        print_calls = [str(call) for call in mock_print.call_args_list]
        printed_text = " ".join(print_calls)

        assert "Current RAG LLM Logs" in printed_text
        assert "session-123" in printed_text
        assert sample_session_data["model_id"] in printed_text

    def test_show_current_logs_multiple_sessions(
        self, mock_http_server, sample_session_data
    ):
        """Test showing current logs for multiple sessions."""
        # Setup mock responses
        mock_http_server.expect_request("/health").respond_with_json(
            {"status": "healthy", "active_sessions": 2}
        )

        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["session-1", "session-2"], "count": 2}
        )

        # Mock logs for both sessions
        session1_data = {**sample_session_data, "session_id": "session-1"}
        session2_data = {**sample_session_data, "session_id": "session-2"}

        mock_http_server.expect_request("/logs/session-1").respond_with_json(
            session1_data
        )
        mock_http_server.expect_request("/logs/session-2").respond_with_json(
            session2_data
        )

        with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
            with patch("builtins.print") as mock_print:
                show_current_logs()

        # Verify logs for both sessions were displayed
        print_calls = [str(call) for call in mock_print.call_args_list]
        printed_text = " ".join(print_calls)

        assert "Current RAG LLM Logs" in printed_text
        assert "session-1" in printed_text
        assert "session-2" in printed_text

    def test_show_current_logs_no_sessions(self, mock_http_server):
        """Test showing logs when no sessions exist."""
        # Setup mock responses
        mock_http_server.expect_request("/health").respond_with_json(
            {"status": "healthy", "active_sessions": 0}
        )

        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": [], "count": 0}
        )

        with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
            with patch("builtins.print") as mock_print:
                show_current_logs()

        # Should show no sessions message
        print_calls = [str(call) for call in mock_print.call_args_list]
        printed_text = " ".join(print_calls)

        assert "No active sessions found" in printed_text
        assert "Upload a PDF first" in printed_text

    def test_show_current_logs_api_unhealthy(self, mock_http_server):
        """Test showing logs when API is unhealthy."""
        # Mock API as unhealthy
        mock_http_server.expect_request("/health").respond_with_data(
            "Service Unavailable", status=503
        )

        with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
            with patch("builtins.print") as mock_print:
                show_current_logs()

        # Should show API not responding message
        print_calls = [str(call) for call in mock_print.call_args_list]
        printed_text = " ".join(print_calls)

        assert "API not responding" in printed_text
        assert "Make sure the API server is running" in printed_text

    def test_log_monitor_concurrent_sessions(
        self, mock_http_server, sample_session_data
    ):
        """Test log monitor with concurrent session changes."""
        # Setup mock responses for concurrent changes
        mock_http_server.expect_request("/health").respond_with_json(
            {"status": "healthy", "active_sessions": 2}
        )

        # Multiple session changes
        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["session-1"], "count": 1}
        )

        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["session-1", "session-2"], "count": 2}
        )

        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["session-2"], "count": 1}
        )

        # Mock logs for sessions
        session1_data = {**sample_session_data, "session_id": "session-1"}
        session2_data = {**sample_session_data, "session_id": "session-2"}

        mock_http_server.expect_request("/logs/session-1").respond_with_json(
            session1_data
        )
        mock_http_server.expect_request("/logs/session-2").respond_with_json(
            session2_data
        )

        with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
            with patch("log_monitor.time.sleep") as mock_sleep:
                mock_sleep.side_effect = [None, None, None, KeyboardInterrupt()]

                with patch("builtins.print") as mock_print:
                    monitor_logs(interval=1)

        # Should detect both new and removed sessions
        print_calls = [str(call) for call in mock_print.call_args_list]
        printed_text = " ".join(print_calls)

        assert "New session(s)" in printed_text
        assert "Removed session(s)" in printed_text

    def test_log_monitor_error_recovery(self, mock_http_server):
        """Test log monitor error recovery and resilience."""
        # Setup mixed responses - some successful, some errors
        mock_http_server.expect_request("/health").respond_with_json(
            {"status": "healthy", "active_sessions": 1}
        )

        # First request succeeds
        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["session-123"], "count": 1}
        )

        # Second request fails
        mock_http_server.expect_request("/sessions").respond_with_data(
            "Internal Server Error", status=500
        )

        # Third request succeeds again
        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["session-123"], "count": 1}
        )

        with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
            with patch("log_monitor.time.sleep") as mock_sleep:
                mock_sleep.side_effect = [None, None, None, KeyboardInterrupt()]

                with patch("builtins.print") as mock_print:
                    monitor_logs(interval=1)

        # Should continue monitoring despite errors
        print_calls = [str(call) for call in mock_print.call_args_list]
        printed_text = " ".join(print_calls)

        # Should show monitoring started
        assert "RAG LLM Log Monitor" in printed_text
        assert "Monitoring API at" in printed_text

    def test_log_monitor_timing_accuracy(self, mock_http_server, sample_session_data):
        """Test that log monitor shows accurate timing information."""
        # Setup mock responses
        mock_http_server.expect_request("/health").respond_with_json(
            {"status": "healthy", "active_sessions": 1}
        )

        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["session-123"], "count": 1}
        )

        mock_http_server.expect_request("/logs/session-123").respond_with_json(
            sample_session_data
        )

        with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
            with patch("log_monitor.time.sleep") as mock_sleep:
                mock_sleep.side_effect = [None, KeyboardInterrupt()]

                with patch("builtins.print") as mock_print:
                    monitor_logs(interval=1)

        # Verify timing information is displayed
        print_calls = [str(call) for call in mock_print.call_args_list]
        printed_text = " ".join(print_calls)

        # Should show timing stats from sample data
        for operation, time_taken in sample_session_data["timing_stats"].items():
            assert operation in printed_text
            assert f"{time_taken:.2f}s" in printed_text

    def test_log_monitor_thread_safety(self, mock_http_server, sample_session_data):
        """Test log monitor thread safety with concurrent operations."""
        # Setup mock responses
        mock_http_server.expect_request("/health").respond_with_json(
            {"status": "healthy", "active_sessions": 1}
        )

        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["session-123"], "count": 1}
        )

        mock_http_server.expect_request("/logs/session-123").respond_with_json(
            sample_session_data
        )

        def run_monitor():
            with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
                with patch("log_monitor.time.sleep") as mock_sleep:
                    mock_sleep.side_effect = [None, KeyboardInterrupt()]
                    with patch("builtins.print"):
                        monitor_logs(interval=1)

        def run_show_logs():
            with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
                with patch("builtins.print"):
                    show_current_logs()

        # Run multiple operations concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_monitor),
                executor.submit(run_show_logs),
                executor.submit(run_show_logs),
            ]

            # Wait for all to complete
            for future in futures:
                future.result(timeout=10)

        # All operations should complete without errors
        assert all(future.done() for future in futures)

    def test_log_monitor_dynamic_session_data(self, mock_http_server):
        """Test log monitor with dynamically changing session data."""
        # Create session data that changes over time
        base_session_data = {
            "session_id": "dynamic-session",
            "model_id": "test-model",
            "document_count": 3,
            "cache_dir": "/tmp/test",
            "system_config": {
                "chunk_size": 1500,
                "chunk_overlap": 100,
                "max_tokens": 500,
                "temperature": 0.1,
            },
            "timing_stats": {},
        }

        # Setup mock responses
        mock_http_server.expect_request("/health").respond_with_json(
            {"status": "healthy", "active_sessions": 1}
        )

        mock_http_server.expect_request("/sessions").respond_with_json(
            {"active_sessions": ["dynamic-session"], "count": 1}
        )

        # First call - no timing stats
        mock_http_server.expect_request("/logs/dynamic-session").respond_with_json(
            base_session_data
        )

        # Second call - with timing stats
        updated_session_data = {
            **base_session_data,
            "timing_stats": {"Document Processing": 1.5, "MLX Model Loading": 2.0},
        }
        mock_http_server.expect_request("/logs/dynamic-session").respond_with_json(
            updated_session_data
        )

        with patch("log_monitor.API_BASE", "http://127.0.0.1:8002"):
            with patch("log_monitor.time.sleep") as mock_sleep:
                mock_sleep.side_effect = [None, None, KeyboardInterrupt()]

                with patch("builtins.print") as mock_print:
                    monitor_logs(interval=1)

        # Should handle changing session data
        print_calls = [str(call) for call in mock_print.call_args_list]
        printed_text = " ".join(print_calls)

        assert "dynamic-session" in printed_text
        assert "Document Processing: 1.50s" in printed_text
        assert "MLX Model Loading: 2.00s" in printed_text
