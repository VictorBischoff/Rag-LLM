#!/usr/bin/env python3
"""
Example client script for the RAG PDF Query API.

This script demonstrates how to:
1. Upload a PDF file
2. Query the document
3. Handle responses and errors
"""

import os
import requests
import sys
from pathlib import Path

# API base URL - can be overridden by environment variable
BASE_URL = os.environ.get("RAG_API_BASE_URL", "http://localhost:8000")


def upload_pdf(pdf_path: str) -> str | None:
    """
    Upload a PDF file to the API.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Session ID for the uploaded document
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"📤 Uploading PDF: {pdf_path}")

    try:
        with open(pdf_path, "rb") as f:
            files = [
                ("files", (Path(pdf_path).name, f, "application/pdf")),
            ]
            response = requests.post(f"{BASE_URL}/upload", files=files)
    except requests.exceptions.RequestException as exc:
        print(f"❌ Upload error: {exc}")
        return None

    if response.status_code == 200:
        data = response.json()
        print("✅ Upload successful!")
        print(f"   Session ID: {data['session_id']}")
        print(f"   Document chunks: {data['document_count']}")
        print(f"   Processing time: {data['processing_time']:.2f}s")
        return data["session_id"]
    else:
        print(f"❌ Upload failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None


def query_document(question: str, session_id: str | None = None) -> str | None:
    """
    Query the document using the API.

    Args:
        question: Question to ask about the document
        session_id: Optional session ID (uses first available if not provided)

    Returns:
        Answer from the RAG system
    """
    print(f"❓ Querying: {question}")

    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id

    try:
        response = requests.post(f"{BASE_URL}/query", json=payload)
    except requests.exceptions.RequestException as exc:
        print(f"❌ Query error: {exc}")
        return None

    if response.status_code == 200:
        data = response.json()
        print("✅ Query successful!")
        print(f"   Answer: {data['answer']}")
        print(f"   Processing time: {data['processing_time']:.2f}s")
        return data["answer"]
    else:
        print(f"❌ Query failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None


def list_sessions():
    """List all active sessions."""
    print("📋 Listing active sessions...")

    try:
        response = requests.get(f"{BASE_URL}/sessions")
    except requests.exceptions.RequestException as exc:
        print(f"❌ Failed to list sessions: {exc}")
        return []

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Active sessions: {data['count']}")
        for session_id in data["active_sessions"]:
            print(f"   - {session_id}")
        return data["active_sessions"]
    else:
        print(f"❌ Failed to list sessions: {response.status_code}")
        return []


def delete_session(session_id: str):
    """Delete a specific session."""
    print(f"🗑️  Deleting session: {session_id}")

    try:
        response = requests.delete(f"{BASE_URL}/sessions/{session_id}")
    except requests.exceptions.RequestException as exc:
        print(f"❌ Failed to delete session: {exc}")
        return

    if response.status_code == 200:
        print("✅ Session deleted successfully")
    else:
        print(f"❌ Failed to delete session: {response.status_code}")


def check_api_health():
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy - {data['active_sessions']} active sessions")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure the API server is running on http://localhost:8000")
        return False


def interactive_mode():
    """Run interactive mode for querying documents."""
    print("🎯 Interactive Query Mode")
    print("Type 'quit' to exit, 'sessions' to list sessions, 'help' for commands")
    print("-" * 50)

    while True:
        try:
            user_input = input("💬 Enter command or question: ").strip()

            if user_input.lower() == "quit":
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "sessions":
                list_sessions()
            elif user_input.lower() == "help":
                print("Available commands:")
                print("  - Ask any question about the uploaded document")
                print("  - 'sessions' - List active sessions")
                print("  - 'quit' - Exit the program")
            elif not user_input:
                continue
            else:
                # Treat as a question
                query_document(user_input)

        except KeyboardInterrupt:
            print("👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function to demonstrate API usage."""
    print("🚀 RAG PDF Query API Client")
    print("=" * 40)

    # Check API health
    if not check_api_health():
        return 1

    # Check if PDF file is provided as argument
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        session_id = upload_pdf(pdf_path)

        if session_id:
            # Run interactive mode
            interactive_mode()
    else:
        print("Usage: python client_example.py <pdf_file>")
        print("Example: python client_example.py test.pdf")
        print("\nOr run without arguments to use existing sessions:")

        # List existing sessions
        sessions = list_sessions()
        if sessions:
            interactive_mode()
        else:
            print("No active sessions found. Please upload a PDF first.")

    return 0


if __name__ == "__main__":
    exit(main())
