"""
End-to-end smoke tests for Streamlit UI.

These tests use streamlit.testing.script_runner to execute ui.main with
monkeypatched st.file_uploader and verify UI behavior with fake uploads.
"""

import pytest
from unittest.mock import Mock, patch
import responses

# Try to import streamlit testing, skip if not available
try:
    import streamlit.testing.script_runner as script_runner

    STREAMLIT_TESTING_AVAILABLE = True
except ImportError:
    STREAMLIT_TESTING_AVAILABLE = False

from ui import main as ui_main, check_api_health, upload_pdfs, query_document


@pytest.mark.e2e
@pytest.mark.skipif(
    not STREAMLIT_TESTING_AVAILABLE, reason="Streamlit testing not available"
)
class TestStreamlitSmoke:
    """Smoke tests for Streamlit UI functionality."""

    def test_streamlit_app_initialization(self):
        """Test that Streamlit app initializes without errors."""
        # Mock streamlit components
        with (
            patch("streamlit.set_page_config") as mock_config,
            patch("streamlit.title") as mock_title,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.sidebar") as mock_sidebar,
            patch("streamlit.header") as mock_header,
            patch("streamlit.file_uploader") as mock_uploader,
            patch("streamlit.button") as mock_button,
            patch("streamlit.text_area") as mock_text_area,
            patch("streamlit.info") as mock_info,
            patch("streamlit.success") as mock_success,
            patch("streamlit.error") as mock_error,
            patch("streamlit.warning") as mock_warning,
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.session_state", {}) as mock_session_state,
        ):
            # Setup mock behaviors
            mock_sidebar.header.return_value = mock_sidebar
            mock_sidebar.button.return_value = False
            mock_uploader.return_value = []
            mock_button.return_value = False
            mock_text_area.return_value = ""
            mock_columns.return_value = [Mock(), Mock()]

            # Run the main function
            ui_main()

            # Verify UI components were called
            mock_config.assert_called_once()
            mock_title.assert_called_once()
            mock_markdown.assert_called_once()
            mock_sidebar.header.assert_called_once()
            mock_header.assert_called()

    def test_streamlit_health_check_integration(self):
        """Test Streamlit health check button functionality."""
        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.title"),
            patch("streamlit.markdown"),
            patch("streamlit.sidebar") as mock_sidebar,
            patch("streamlit.header"),
            patch("streamlit.file_uploader") as mock_uploader,
            patch("streamlit.button") as mock_button,
            patch("streamlit.text_area") as mock_text_area,
            patch("streamlit.info"),
            patch("streamlit.success") as mock_success,
            patch("streamlit.error") as mock_error,
            patch("streamlit.warning"),
            patch("streamlit.columns"),
            patch("streamlit.session_state", {}) as mock_session_state,
        ):
            # Setup mock behaviors
            mock_sidebar.header.return_value = mock_sidebar
            mock_sidebar.button.return_value = True  # Health check button clicked
            mock_uploader.return_value = []
            mock_button.return_value = False
            mock_text_area.return_value = ""

            with responses.RequestsMock() as rsps:
                rsps.add(
                    responses.GET,
                    "http://localhost:8000/health",
                    json={"status": "healthy", "active_sessions": 2},
                    status=200,
                )

                # Run the main function
                ui_main()

                # Verify success message was shown
                mock_success.assert_called()
                success_calls = [str(call) for call in mock_success.call_args_list]
                assert any("API is healthy" in call for call in success_calls)

    def test_streamlit_upload_functionality(self):
        """Test Streamlit PDF upload functionality."""
        # Create mock uploaded files
        mock_file1 = Mock()
        mock_file1.name = "test1.pdf"
        mock_file1.getvalue.return_value = b"dummy pdf content 1"

        mock_file2 = Mock()
        mock_file2.name = "test2.pdf"
        mock_file2.getvalue.return_value = b"dummy pdf content 2"

        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.title"),
            patch("streamlit.markdown"),
            patch("streamlit.sidebar") as mock_sidebar,
            patch("streamlit.header"),
            patch("streamlit.file_uploader") as mock_uploader,
            patch("streamlit.button") as mock_button,
            patch("streamlit.text_area") as mock_text_area,
            patch("streamlit.info") as mock_info,
            patch("streamlit.success") as mock_success,
            patch("streamlit.error") as mock_error,
            patch("streamlit.warning"),
            patch("streamlit.columns"),
            patch("streamlit.spinner") as mock_spinner,
            patch("streamlit.session_state", {}) as mock_session_state,
        ):
            # Setup mock behaviors
            mock_sidebar.header.return_value = mock_sidebar
            mock_sidebar.button.return_value = False
            mock_uploader.return_value = [mock_file1, mock_file2]  # Files uploaded
            mock_button.return_value = True  # Upload button clicked
            mock_text_area.return_value = ""
            mock_spinner.return_value.__enter__ = Mock()
            mock_spinner.return_value.__exit__ = Mock()

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

                # Run the main function
                ui_main()

                # Verify upload success message
                mock_success.assert_called()
                success_calls = [str(call) for call in mock_success.call_args_list]
                assert any("Upload successful" in call for call in success_calls)

                # Verify session state was updated
                assert mock_session_state.get("session_id") == "test-session-123"

    def test_streamlit_query_functionality(self):
        """Test Streamlit query functionality."""
        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.title"),
            patch("streamlit.markdown"),
            patch("streamlit.sidebar") as mock_sidebar,
            patch("streamlit.header"),
            patch("streamlit.file_uploader") as mock_uploader,
            patch("streamlit.button") as mock_button,
            patch("streamlit.text_area") as mock_text_area,
            patch("streamlit.info") as mock_info,
            patch("streamlit.success"),
            patch("streamlit.error") as mock_error,
            patch("streamlit.warning"),
            patch("streamlit.columns"),
            patch("streamlit.spinner") as mock_spinner,
            patch(
                "streamlit.session_state", {"session_id": "test-session-123"}
            ) as mock_session_state,
        ):
            # Setup mock behaviors
            mock_sidebar.header.return_value = mock_sidebar
            mock_sidebar.button.return_value = False
            mock_uploader.return_value = []
            mock_button.return_value = True  # Query button clicked
            mock_text_area.return_value = "What is this document about?"
            mock_spinner.return_value.__enter__ = Mock()
            mock_spinner.return_value.__exit__ = Mock()

            with responses.RequestsMock() as rsps:
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

                # Run the main function
                ui_main()

                # Verify query was processed (no error calls)
                error_calls = [str(call) for call in mock_error.call_args_list]
                assert not any("Query failed" in call for call in error_calls)

    def test_streamlit_error_handling(self):
        """Test Streamlit error handling for API failures."""
        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.title"),
            patch("streamlit.markdown"),
            patch("streamlit.sidebar") as mock_sidebar,
            patch("streamlit.header"),
            patch("streamlit.file_uploader") as mock_uploader,
            patch("streamlit.button") as mock_button,
            patch("streamlit.text_area") as mock_text_area,
            patch("streamlit.info"),
            patch("streamlit.success"),
            patch("streamlit.error") as mock_error,
            patch("streamlit.warning"),
            patch("streamlit.columns"),
            patch("streamlit.spinner") as mock_spinner,
            patch("streamlit.session_state", {}) as mock_session_state,
        ):
            # Setup mock behaviors
            mock_sidebar.header.return_value = mock_sidebar
            mock_sidebar.button.return_value = True  # Health check button clicked
            mock_uploader.return_value = []
            mock_button.return_value = False
            mock_text_area.return_value = ""
            mock_spinner.return_value.__enter__ = Mock()
            mock_spinner.return_value.__exit__ = Mock()

            with responses.RequestsMock() as rsps:
                rsps.add(
                    responses.GET,
                    "http://localhost:8000/health",
                    body=Exception("Connection failed"),
                )

                # Run the main function
                ui_main()

                # Verify error message was shown
                mock_error.assert_called()
                error_calls = [str(call) for call in mock_error.call_args_list]
                assert any("Cannot connect to API" in call for call in error_calls)

    def test_streamlit_example_questions(self):
        """Test Streamlit example questions functionality."""
        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.title"),
            patch("streamlit.markdown"),
            patch("streamlit.sidebar") as mock_sidebar,
            patch("streamlit.header"),
            patch("streamlit.file_uploader") as mock_uploader,
            patch("streamlit.button") as mock_button,
            patch("streamlit.text_area") as mock_text_area,
            patch("streamlit.info"),
            patch("streamlit.success"),
            patch("streamlit.error"),
            patch("streamlit.warning"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.session_state", {}) as mock_session_state,
        ):
            # Setup mock behaviors
            mock_sidebar.header.return_value = mock_sidebar
            mock_sidebar.button.return_value = False
            mock_uploader.return_value = []
            mock_button.return_value = True  # Example question button clicked
            mock_text_area.return_value = ""

            # Mock columns
            mock_col1 = Mock()
            mock_col2 = Mock()
            mock_columns.return_value = [mock_col1, mock_col2]

            # Run the main function
            ui_main()

            # Verify example questions were set up
            mock_columns.assert_called()
            # Should have called button for each example question
            assert mock_button.call_count >= 5  # 5 example questions

    def test_streamlit_session_state_management(self):
        """Test Streamlit session state management."""
        initial_session_state = {}

        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.title"),
            patch("streamlit.markdown"),
            patch("streamlit.sidebar") as mock_sidebar,
            patch("streamlit.header"),
            patch("streamlit.file_uploader") as mock_uploader,
            patch("streamlit.button") as mock_button,
            patch("streamlit.text_area") as mock_text_area,
            patch("streamlit.info") as mock_info,
            patch("streamlit.success"),
            patch("streamlit.error"),
            patch("streamlit.warning"),
            patch("streamlit.columns"),
            patch(
                "streamlit.session_state", initial_session_state
            ) as mock_session_state,
        ):
            # Setup mock behaviors
            mock_sidebar.header.return_value = mock_sidebar
            mock_sidebar.button.return_value = False
            mock_uploader.return_value = []
            mock_button.return_value = False
            mock_text_area.return_value = ""

            # Run the main function
            ui_main()

            # Verify session state was initialized
            assert "session_id" in mock_session_state
            assert mock_session_state["session_id"] is None

    def test_streamlit_ui_components_rendering(self):
        """Test that all Streamlit UI components are rendered correctly."""
        with (
            patch("streamlit.set_page_config") as mock_config,
            patch("streamlit.title") as mock_title,
            patch("streamlit.markdown") as mock_markdown,
            patch("streamlit.sidebar") as mock_sidebar,
            patch("streamlit.header") as mock_header,
            patch("streamlit.file_uploader") as mock_uploader,
            patch("streamlit.button") as mock_button,
            patch("streamlit.text_area") as mock_text_area,
            patch("streamlit.info") as mock_info,
            patch("streamlit.success"),
            patch("streamlit.error"),
            patch("streamlit.warning"),
            patch("streamlit.columns") as mock_columns,
            patch("streamlit.session_state", {}) as mock_session_state,
        ):
            # Setup mock behaviors
            mock_sidebar.header.return_value = mock_sidebar
            mock_sidebar.button.return_value = False
            mock_uploader.return_value = []
            mock_button.return_value = False
            mock_text_area.return_value = ""
            mock_columns.return_value = [Mock(), Mock()]

            # Run the main function
            ui_main()

            # Verify all major UI components were called
            mock_config.assert_called_once()
            mock_title.assert_called_once()
            mock_markdown.assert_called()
            mock_sidebar.header.assert_called_once()
            mock_header.assert_called()  # Should be called multiple times
            mock_uploader.assert_called_once()
            mock_button.assert_called()  # Multiple buttons
            mock_text_area.assert_called_once()
            mock_info.assert_called()  # Session info
            mock_columns.assert_called_once()  # For example questions

    def test_streamlit_api_integration_workflow(self):
        """Test complete Streamlit API integration workflow."""
        # Create mock uploaded file
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.title"),
            patch("streamlit.markdown"),
            patch("streamlit.sidebar") as mock_sidebar,
            patch("streamlit.header"),
            patch("streamlit.file_uploader") as mock_uploader,
            patch("streamlit.button") as mock_button,
            patch("streamlit.text_area") as mock_text_area,
            patch("streamlit.info") as mock_info,
            patch("streamlit.success") as mock_success,
            patch("streamlit.error") as mock_error,
            patch("streamlit.warning"),
            patch("streamlit.columns"),
            patch("streamlit.spinner") as mock_spinner,
            patch("streamlit.session_state", {}) as mock_session_state,
        ):
            # Setup mock behaviors for complete workflow
            mock_sidebar.header.return_value = mock_sidebar
            mock_sidebar.button.return_value = False
            mock_uploader.return_value = [mock_file]  # File uploaded
            mock_button.side_effect = [True, False, True]  # Upload, then query
            mock_text_area.return_value = "What is this document about?"
            mock_spinner.return_value.__enter__ = Mock()
            mock_spinner.return_value.__exit__ = Mock()

            with responses.RequestsMock() as rsps:
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
                        "answer": "This document is about testing Streamlit integration.",
                        "session_id": "test-session-123",
                        "processing_time": 1.2,
                    },
                    status=200,
                )

                # Run the main function
                ui_main()

                # Verify complete workflow
                mock_success.assert_called()  # Upload success
                # Should not have query errors
                error_calls = [str(call) for call in mock_error.call_args_list]
                assert not any("Query failed" in call for call in error_calls)

                # Verify session state was updated
                assert mock_session_state.get("session_id") == "test-session-123"


@pytest.mark.e2e
class TestStreamlitUtilities:
    """Test Streamlit utility functions independently."""

    def test_check_api_health_utility(self):
        """Test check_api_health utility function."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.GET,
                "http://localhost:8000/health",
                json={"status": "healthy", "active_sessions": 1},
                status=200,
            )

            is_healthy, status = check_api_health()

            assert is_healthy is True
            assert "✅ API is healthy" in status
            assert "1 active sessions" in status

    def test_upload_pdfs_utility(self):
        """Test upload_pdfs utility function."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/upload",
                json={
                    "session_id": "test-session-123",
                    "document_count": 3,
                    "processing_time": 1.5,
                },
                status=200,
            )

            session_id, status = upload_pdfs([mock_file])

            assert session_id == "test-session-123"
            assert "✅ Upload successful!" in status
            assert "Files: 1" in status
            assert "Chunks: 3" in status

    def test_query_document_utility(self):
        """Test query_document utility function."""
        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://localhost:8000/query",
                json={
                    "answer": "This is a test answer.",
                    "session_id": "test-session-123",
                    "processing_time": 1.0,
                },
                status=200,
            )

            answer = query_document("What is this about?", "test-session-123")

            assert answer == "This is a test answer."
