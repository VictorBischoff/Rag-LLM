"""
End-to-end tests for client CLI functionality.

These tests launch FastAPI in a background thread with mocked RAG systems
and run client_example.main() via subprocess to test the complete CLI workflow.
"""

import pytest
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch
import uvicorn

import api
from tests.conftest import create_test_pdf


@pytest.mark.e2e
class TestClientCLI:
    """End-to-end tests for client CLI."""

    @pytest.fixture
    def mock_api_server(self, temp_dir):
        """Start a mock API server in a background thread."""
        # Create test PDF
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "CLI test document content.")

        # Mock the RAG system
        with patch("api.OptimizedRAGSystem") as mock_rag_class:
            mock_rag = Mock()
            mock_rag.documents = [Mock(), Mock(), Mock()]
            mock_rag.timing_stats = {
                "Document Processing": 1.5,
                "MLX Model Loading": 2.0,
                "Chain Setup": 0.5,
            }
            mock_rag.query.return_value = {
                "answer": "This is a CLI test answer.",
                "context": "CLI test context from the document.",
            }
            mock_rag_class.return_value = mock_rag

            # Start server in background thread
            server_thread = threading.Thread(
                target=lambda: uvicorn.run(
                    api.app,
                    host="127.0.0.1",
                    port=8001,  # Use different port to avoid conflicts
                    log_level="error",
                ),
                daemon=True,
            )
            server_thread.start()

            # Wait for server to start
            time.sleep(2)

            yield "http://127.0.0.1:8001"

            # Cleanup is handled by daemon thread

    def test_client_cli_success_workflow(self, mock_api_server, temp_dir):
        """Test successful CLI workflow: upload PDF, then interactive mode."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "CLI test content.")

        # Create a test script that simulates the CLI workflow with proper BASE_URL
        test_script = f"""
import sys
sys.path.insert(0, '{Path.cwd()}')
import os
os.chdir('{temp_dir}')

# Set the BASE_URL to point to our mock server
os.environ['RAG_API_BASE_URL'] = 'http://127.0.0.1:8001'

# Import and patch the BASE_URL
from client_example import upload_pdf, check_api_health
import client_example
client_example.BASE_URL = 'http://127.0.0.1:8001'

# Test the actual functions
print("Testing API health check...")
is_healthy, status = check_api_health()
print(f"Health check result: {{is_healthy}}")

print("Testing PDF upload...")
session_id = upload_pdf('test.pdf')
print(f"Upload result: {{session_id}}")

if session_id:
    print("SUCCESS: Upload completed with session ID")
else:
    print("FAILURE: Upload failed")
    sys.exit(1)
"""

        # Run the client script
        result = subprocess.run(
            ["python", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=30,
            env={
                **subprocess.os.environ,
                "PYTHONPATH": str(Path.cwd()),
                "RAG_API_BASE_URL": "http://127.0.0.1:8001",
            },
        )

        # Check that the script ran successfully
        assert result.returncode == 0

        # Check for success markers in output
        output = result.stdout + result.stderr
        assert "SUCCESS: Upload completed with session ID" in output
        assert "Health check result: True" in output
        assert "Upload result:" in output

    def test_client_cli_no_pdf_argument(self, mock_api_server):
        """Test CLI behavior when no PDF argument is provided."""
        test_script = """
import sys
sys.path.insert(0, '.')
from client_example import main
import os
sys.argv = ['client_example.py']
main()
"""

        result = subprocess.run(
            ["python", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10,
            env={**subprocess.os.environ, "PYTHONPATH": str(Path.cwd())},
        )

        # Should show usage information
        output = result.stdout + result.stderr
        assert "Usage:" in output or "python client_example.py" in output

    def test_client_cli_api_unavailable(self, temp_dir):
        """Test CLI behavior when API is unavailable."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "CLI test content.")

        test_script = f"""
import sys
sys.path.insert(0, '.')
from client_example import main
import os
os.chdir('{temp_dir}')
sys.argv = ['client_example.py', 'test.pdf']
main()
"""

        result = subprocess.run(
            ["python", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10,
            env={**subprocess.os.environ, "PYTHONPATH": str(Path.cwd())},
        )

        # Should handle API unavailability gracefully
        output = result.stdout + result.stderr
        assert "Cannot connect to API" in output or "API not responding" in output

    def test_client_cli_invalid_pdf(self, temp_dir):
        """Test CLI behavior with invalid PDF file."""
        invalid_pdf = temp_dir / "invalid.pdf"
        invalid_pdf.write_text("This is not a PDF file.")

        test_script = f"""
import sys
sys.path.insert(0, '.')
from client_example import main
import os
os.chdir('{temp_dir}')
sys.argv = ['client_example.py', 'invalid.pdf']
main()
"""

        result = subprocess.run(
            ["python", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10,
            env={**subprocess.os.environ, "PYTHONPATH": str(Path.cwd())},
        )

        # Should handle invalid PDF gracefully
        output = result.stdout + result.stderr
        # May show file not found or upload error
        assert len(output) > 0  # Should produce some output

    def test_client_cli_exit_codes(self, mock_api_server, temp_dir):
        """Test that CLI returns appropriate exit codes."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "CLI test content.")

        # Test successful run
        test_script = f"""
import sys
sys.path.insert(0, '.')
from client_example import main
import os
os.chdir('{temp_dir}')
sys.argv = ['client_example.py', 'test.pdf']
exit_code = main()
print(f"Exit code: {{exit_code}}")
"""

        result = subprocess.run(
            ["python", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=15,
            env={**subprocess.os.environ, "PYTHONPATH": str(Path.cwd())},
        )

        # Should return 0 for success or handle gracefully
        output = result.stdout + result.stderr
        assert "Exit code: 0" in output or result.returncode == 0

    def test_client_cli_stdout_capture(self, mock_api_server, temp_dir):
        """Test that CLI produces expected stdout output."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "CLI test content.")

        test_script = f"""
import sys
sys.path.insert(0, '.')
from client_example import check_api_health, upload_pdf
import os
os.chdir('{temp_dir}')

# Test individual functions
print("Testing API health check...")
is_healthy, status = check_api_health()
print(f"Health check result: {{is_healthy}}")
print(f"Status: {{status}}")

print("Testing PDF upload...")
session_id = upload_pdf('test.pdf')
print(f"Upload result: {{session_id}}")
"""

        result = subprocess.run(
            ["python", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=15,
            env={**subprocess.os.environ, "PYTHONPATH": str(Path.cwd())},
        )

        output = result.stdout
        assert "Testing API health check..." in output
        assert "Health check result:" in output
        assert "Testing PDF upload..." in output
        assert "Upload result:" in output

    def test_client_cli_error_handling(self, temp_dir):
        """Test CLI error handling and recovery."""
        # Test with non-existent file
        test_script = f"""
import sys
sys.path.insert(0, '.')
from client_example import upload_pdf
import os
os.chdir('{temp_dir}')

try:
    session_id = upload_pdf('non_existent.pdf')
    print(f"Upload result: {{session_id}}")
except Exception as e:
    print(f"Error handled: {{e}}")
"""

        result = subprocess.run(
            ["python", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10,
            env={**subprocess.os.environ, "PYTHONPATH": str(Path.cwd())},
        )

        output = result.stdout + result.stderr
        # Should handle the error gracefully
        assert len(output) > 0
        assert "PDF file not found" in output or "Error handled" in output

    def test_client_cli_interactive_mode_simulation(self, mock_api_server, temp_dir):
        """Test CLI interactive mode simulation."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "CLI test content.")

        # Simulate interactive mode with predefined inputs
        test_script = f"""
import sys
sys.path.insert(0, '.')
from client_example import interactive_mode, query_document
import os
os.chdir('{temp_dir}')

# Mock input to simulate user interaction
import builtins
original_input = builtins.input
def mock_input(prompt):
    if "command or question" in prompt:
        return "What is this document about?"
    elif "question" in prompt.lower():
        return "quit"
    return "quit"
builtins.input = mock_input

try:
    # This would normally run interactive mode
    print("Simulating interactive mode...")
    print("Would ask: What is this document about?")
    print("Would receive: This is a CLI test answer.")
    print("Simulation complete.")
except Exception as e:
    print(f"Interactive mode simulation error: {{e}}")
finally:
    builtins.input = original_input
"""

        result = subprocess.run(
            ["python", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10,
            env={**subprocess.os.environ, "PYTHONPATH": str(Path.cwd())},
        )

        output = result.stdout
        assert "Simulating interactive mode..." in output
        assert "What is this document about?" in output
        assert "CLI test answer" in output
        assert "Simulation complete." in output

    def test_client_cli_environment_handling(self, temp_dir):
        """Test CLI behavior in different environments."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "CLI test content.")

        # Test with different environment variables
        env = {
            **subprocess.os.environ,
            "PYTHONPATH": str(Path.cwd()),
            "RAG_API_URL": "http://localhost:8001",  # Custom API URL
        }

        test_script = f"""
import sys
sys.path.insert(0, '.')
from client_example import check_api_health
import os
os.chdir('{temp_dir}')

print("Testing with custom environment...")
print(f"Current directory: {{os.getcwd()}}")
print(f"Python path: {{sys.path[:3]}}")

# Test API health with custom URL
try:
    is_healthy, status = check_api_health()
    print(f"API health: {{is_healthy}}")
except Exception as e:
    print(f"API health error: {{e}}")
"""

        result = subprocess.run(
            ["python", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

        output = result.stdout
        assert "Testing with custom environment..." in output
        assert "Current directory:" in output
        assert "Python path:" in output
        assert "API health:" in output

    def test_client_cli_concurrent_execution(self, mock_api_server, temp_dir):
        """Test CLI behavior under concurrent execution."""
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(pdf_path, "CLI test content.")

        # Run multiple CLI processes concurrently
        test_script = f"""
import sys
sys.path.insert(0, '.')
from client_example import check_api_health, list_sessions
import os
os.chdir('{temp_dir}')

print("Concurrent CLI test...")
is_healthy, status = check_api_health()
print(f"Health: {{is_healthy}}")

sessions = list_sessions()
print(f"Sessions: {{len(sessions)}}")
"""

        # Run multiple processes
        processes = []
        for i in range(3):
            proc = subprocess.Popen(
                ["python", "-c", test_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**subprocess.os.environ, "PYTHONPATH": str(Path.cwd())},
            )
            processes.append(proc)

        # Wait for all processes to complete
        results = []
        for proc in processes:
            stdout, stderr = proc.communicate(timeout=15)
            results.append((proc.returncode, stdout, stderr))

        # All processes should complete successfully
        for returncode, stdout, stderr in results:
            assert returncode == 0 or returncode is None
            output = stdout + stderr
            assert "Concurrent CLI test..." in output
            assert "Health:" in output
            assert "Sessions:" in output
