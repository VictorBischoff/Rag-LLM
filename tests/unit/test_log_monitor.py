"""
Unit tests for log_monitor.py helper functions.

These tests use responses/requests-mock to test HTTP client functionality
and verify log formatting without requiring a running API server.
"""

import pytest
import requests
from unittest.mock import patch
import responses

# Import the functions we're testing
from log_monitor import (
    check_api_health,
    get_active_sessions,
    get_session_logs,
    format_logs,
    monitor_logs,
    show_current_logs,
)


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
class TestGetActiveSessions:
    """Test cases for get_active_sessions function."""

    def test_get_active_sessions_success(self):
        """Test successful retrieval of active sessions."""
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

            sessions = get_active_sessions()

            assert sessions == ["session-1", "session-2", "session-3"]

    def test_get_active_sessions_empty(self):
        """Test retrieval when no sessions exist."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": [], "count": 0},
                status=200,
            )

            sessions = get_active_sessions()

            assert sessions == []

    def test_get_active_sessions_api_error(self):
        """Test retrieval when API returns error."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"error": "Internal server error"},
                status=500,
            )

            sessions = get_active_sessions()

            assert sessions == []

    def test_get_active_sessions_network_error(self):
        """Test retrieval when network request fails."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            sessions = get_active_sessions()

            assert sessions == []


@pytest.mark.unit
class TestGetSessionLogs:
    """Test cases for get_session_logs function."""

    def test_get_session_logs_success(self, sample_session_data):
        """Test successful retrieval of session logs."""
        session_id = sample_session_data["session_id"]

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                f"http://localhost:8000/logs/{session_id}",
                json=sample_session_data,
                status=200,
            )

            logs = get_session_logs(session_id)

            assert logs == sample_session_data

    def test_get_session_logs_not_found(self):
        """Test retrieval of logs for non-existent session."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/logs/invalid-session",
                json={"error": "Session not found"},
                status=404,
            )

            logs = get_session_logs("invalid-session")

            assert logs is None

    def test_get_session_logs_network_error(self):
        """Test retrieval when network request fails."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/logs/test-session",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            logs = get_session_logs("test-session")

            assert logs is None


@pytest.mark.unit
class TestFormatLogs:
    """Test cases for format_logs function."""

    def test_format_logs_success(self, sample_session_data):
        """Test successful log formatting."""
        formatted = format_logs(sample_session_data)

        assert isinstance(formatted, str)
        assert "SESSION LOGS" in formatted
        assert sample_session_data["session_id"] in formatted
        assert sample_session_data["model_id"] in formatted
        assert str(sample_session_data["document_count"]) in formatted
        assert sample_session_data["cache_dir"] in formatted

        # Check system config
        config = sample_session_data["system_config"]
        assert str(config["chunk_size"]) in formatted
        assert str(config["chunk_overlap"]) in formatted
        assert str(config["max_tokens"]) in formatted
        assert str(config["temperature"]) in formatted

        # Check timing stats
        timing_stats = sample_session_data["timing_stats"]
        for operation, time_taken in timing_stats.items():
            assert operation in formatted
            assert f"{time_taken:.2f}s" in formatted

    def test_format_logs_none_input(self):
        """Test log formatting with None input."""
        formatted = format_logs(None)

        assert formatted == "No logs available"

    def test_format_logs_empty_timing_stats(self):
        """Test log formatting with empty timing stats."""
        logs_data = {
            "session_id": "test-session",
            "model_id": "test-model",
            "document_count": 0,
            "cache_dir": "/tmp/test",
            "system_config": {
                "chunk_size": 1000,
                "chunk_overlap": 50,
                "max_tokens": 300,
                "temperature": 0.2,
            },
            "timing_stats": {},
        }

        formatted = format_logs(logs_data)

        assert "No timing data available yet" in formatted

    def test_format_logs_missing_fields(self):
        """Test log formatting with missing optional fields."""
        logs_data = {
            "session_id": "test-session",
            "model_id": "test-model",
            "document_count": 1,
            "cache_dir": "/tmp/test",
            "system_config": {
                "chunk_size": 1000,
                "chunk_overlap": 50,
                "max_tokens": 300,
                "temperature": 0.2,
            },
            "timing_stats": {"Test Operation": 1.5},
        }

        formatted = format_logs(logs_data)

        assert "SESSION LOGS" in formatted
        assert "test-session" in formatted
        assert "Test Operation: 1.50s" in formatted

    def test_format_logs_snapshot_output(self, sample_session_data):
        """Test that format_logs produces consistent output (snapshot test)."""
        formatted = format_logs(sample_session_data)

        # Verify the structure and key elements
        lines = formatted.split("\n")

        # Check header
        assert "=" * 60 in lines[0]
        assert "SESSION LOGS" in lines[1]
        assert sample_session_data["session_id"] in lines[1]
        assert "=" * 60 in lines[2]

        # Check model and document info
        assert "🤖 Model:" in formatted
        assert "📄 Documents:" in formatted
        assert "📁 Cache Dir:" in formatted

        # Check system config section
        assert "⚙️  SYSTEM CONFIG:" in formatted
        assert "Chunk Size:" in formatted
        assert "Chunk Overlap:" in formatted
        assert "Max Tokens:" in formatted
        assert "Temperature:" in formatted

        # Check timing section
        assert "⏱️  PERFORMANCE TIMING:" in formatted

        # Check footer
        assert "=" * 60 in lines[-1]


@pytest.mark.unit
class TestMonitorLogs:
    """Test cases for monitor_logs function."""

    @patch("log_monitor.time.sleep")
    @patch("builtins.print")
    def test_monitor_logs_basic_flow(self, mock_print, mock_sleep, sample_session_data):
        """Test basic monitoring flow."""
        # Mock sleep to avoid actual waiting
        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        with responses.RequestsMock() as rsps:
            # Mock health check
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"status": "healthy", "active_sessions": 1},
                status=200,
            )

            # Mock sessions
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": ["test-session-123"], "count": 1},
                status=200,
            )

            # Mock session logs
            rsps.add(
                responses.GET,
                "http://localhost:8000/logs/test-session-123",
                json=sample_session_data,
                status=200,
            )

            monitor_logs(interval=1)

        # Verify monitoring started
        mock_print.assert_any_call("🔍 RAG LLM Log Monitor")
        mock_print.assert_any_call("Monitoring API at: http://localhost:8000")
        mock_print.assert_any_call("Press Ctrl+C to stop")

    @patch("log_monitor.time.sleep")
    @patch("builtins.print")
    def test_monitor_logs_api_unhealthy(self, mock_print, mock_sleep):
        """Test monitoring when API is unhealthy."""
        mock_sleep.side_effect = [None, KeyboardInterrupt()]

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            monitor_logs(interval=1)

        # Should print API not responding message
        mock_print.assert_any_call("❌ [", "API not responding")

    @patch("log_monitor.time.sleep")
    @patch("builtins.print")
    def test_monitor_logs_new_sessions(self, mock_print, mock_sleep):
        """Test monitoring detects new sessions."""
        mock_sleep.side_effect = [None, None, KeyboardInterrupt()]

        with responses.RequestsMock() as rsps:
            # Mock health checks
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"status": "healthy", "active_sessions": 1},
                status=200,
            )

            # First call: no sessions
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": [], "count": 0},
                status=200,
            )

            # Second call: new session appears
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": ["new-session"], "count": 1},
                status=200,
            )

            monitor_logs(interval=1)

        # Should detect new session
        mock_print.assert_any_call("✅ [", "New session(s): new-session")

    @patch("log_monitor.time.sleep")
    @patch("builtins.print")
    def test_monitor_logs_removed_sessions(self, mock_print, mock_sleep):
        """Test monitoring detects removed sessions."""
        mock_sleep.side_effect = [None, None, KeyboardInterrupt()]

        with responses.RequestsMock() as rsps:
            # Mock health checks
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"status": "healthy", "active_sessions": 0},
                status=200,
            )

            # First call: session exists
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": ["old-session"], "count": 1},
                status=200,
            )

            # Second call: session removed
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": [], "count": 0},
                status=200,
            )

            monitor_logs(interval=1)

        # Should detect removed session
        mock_print.assert_any_call("🗑️  [", "Removed session(s): old-session")


@pytest.mark.unit
class TestShowCurrentLogs:
    """Test cases for show_current_logs function."""

    @patch("builtins.print")
    def test_show_current_logs_success(self, mock_print, sample_session_data):
        """Test successful display of current logs."""
        with responses.RequestsMock() as rsps:
            # Mock health check
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"status": "healthy", "active_sessions": 1},
                status=200,
            )

            # Mock sessions
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": ["test-session-123"], "count": 1},
                status=200,
            )

            # Mock session logs
            rsps.add(
                responses.GET,
                "http://localhost:8000/logs/test-session-123",
                json=sample_session_data,
                status=200,
            )

            show_current_logs()

        # Verify logs were displayed
        mock_print.assert_any_call("📊 Current RAG LLM Logs")
        mock_print.assert_any_call("=" * 50)

    @patch("builtins.print")
    def test_show_current_logs_api_unhealthy(self, mock_print):
        """Test display when API is unhealthy."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            show_current_logs()

        mock_print.assert_any_call(
            "❌ API not responding. Make sure the API server is running."
        )

    @patch("builtins.print")
    def test_show_current_logs_no_sessions(self, mock_print):
        """Test display when no sessions exist."""
        with responses.RequestsMock() as rsps:
            # Mock health check
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"status": "healthy", "active_sessions": 0},
                status=200,
            )

            # Mock empty sessions
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": [], "count": 0},
                status=200,
            )

            show_current_logs()

        mock_print.assert_any_call("📭 No active sessions found.")
        mock_print.assert_any_call("Upload a PDF first to see logs.")

    @patch("builtins.print")
    def test_show_current_logs_multiple_sessions(self, mock_print, sample_session_data):
        """Test display with multiple sessions."""
        with responses.RequestsMock() as rsps:
            # Mock health check
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"status": "healthy", "active_sessions": 2},
                status=200,
            )

            # Mock sessions
            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": ["session-1", "session-2"], "count": 2},
                status=200,
            )

            # Mock session logs for both sessions
            rsps.add(
                responses.GET,
                "http://localhost:8000/logs/session-1",
                json={**sample_session_data, "session_id": "session-1"},
                status=200,
            )
            rsps.add(
                responses.GET,
                "http://localhost:8000/logs/session-2",
                json={**sample_session_data, "session_id": "session-2"},
                status=200,
            )

            show_current_logs()

        # Should display logs for both sessions
        mock_print.assert_any_call("📊 Current RAG LLM Logs")
        # Verify both session IDs appear in the output
        printed_text = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
        assert "session-1" in printed_text
        assert "session-2" in printed_text


@pytest.mark.unit
class TestLogMonitorIntegration:
    """Integration tests for log_monitor functions."""

    def test_complete_monitoring_workflow(self, sample_session_data):
        """Test complete monitoring workflow."""
        with responses.RequestsMock() as rsps:
            # Mock all API endpoints
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"status": "healthy", "active_sessions": 1},
                status=200,
            )

            rsps.add(
                responses.GET,
                "http://localhost:8000/sessions",
                json={"active_sessions": ["test-session-123"], "count": 1},
                status=200,
            )

            rsps.add(
                responses.GET,
                "http://localhost:8000/logs/test-session-123",
                json=sample_session_data,
                status=200,
            )

            # Test health check
            assert check_api_health() is True

            # Test get sessions
            sessions = get_active_sessions()
            assert sessions == ["test-session-123"]

            # Test get logs
            logs = get_session_logs("test-session-123")
            assert logs == sample_session_data

            # Test format logs
            formatted = format_logs(logs)
            assert "SESSION LOGS" in formatted
            assert "test-session-123" in formatted

    def test_error_handling_workflow(self):
        """Test error handling in complete workflow."""
        with responses.RequestsMock() as rsps:
            # Mock API as unhealthy
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                body=requests.exceptions.ConnectionError("Connection failed"),
            )

            # Test health check fails
            assert check_api_health() is False

            # Test get sessions returns empty list
            sessions = get_active_sessions()
            assert sessions == []

            # Test get logs returns None
            logs = get_session_logs("invalid-session")
            assert logs is None

            # Test format logs with None
            formatted = format_logs(None)
            assert formatted == "No logs available"

    def test_log_formatting_consistency(self, sample_session_data):
        """Test that log formatting is consistent and complete."""
        formatted = format_logs(sample_session_data)

        # Verify all required sections are present
        required_sections = [
            "SESSION LOGS",
            "🤖 Model:",
            "📄 Documents:",
            "📁 Cache Dir:",
            "⚙️  SYSTEM CONFIG:",
            "⏱️  PERFORMANCE TIMING:",
        ]

        for section in required_sections:
            assert section in formatted, f"Missing section: {section}"

        # Verify all timing stats are included
        for operation, time_taken in sample_session_data["timing_stats"].items():
            assert operation in formatted
            assert f"{time_taken:.2f}s" in formatted

        # Verify all system config values are included
        config = sample_session_data["system_config"]
        for key, value in config.items():
            assert str(value) in formatted

        # Verify session ID is included
        assert sample_session_data["session_id"] in formatted
