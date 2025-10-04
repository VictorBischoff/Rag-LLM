#!/usr/bin/env python3
"""
Log Monitor for RAG LLM System

This script helps you monitor logs from your RAG system in real-time.
Run this alongside your API server to see detailed LLM logs.
"""

import argparse
import logging
import os
import time
from datetime import datetime
from json import JSONDecodeError

import requests

API_BASE = os.environ.get("RAG_API_BASE_URL", "http://localhost:8000")
logger = logging.getLogger("rag_log_monitor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.warning("API health check failed: %s", exc)
        return False


def get_active_sessions():
    """Get list of active sessions."""
    try:
        response = requests.get(f"{API_BASE}/sessions", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("active_sessions", [])
    except (requests.RequestException, JSONDecodeError) as exc:
        logger.warning("Failed to retrieve active sessions: %s", exc)
        return []


def get_session_logs(session_id):
    """Get logs for a specific session."""
    try:
        response = requests.get(f"{API_BASE}/logs/{session_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, JSONDecodeError) as exc:
        logger.warning("Failed to retrieve logs for %s: %s", session_id, exc)
        return None


def format_logs(logs_data):
    """Format logs data for display."""
    if not logs_data:
        return "No logs available"

    output = []
    output.append("=" * 60)
    output.append(f"📊 SESSION LOGS - {logs_data['session_id']}")
    output.append("=" * 60)
    output.append(f"🤖 Model: {logs_data['model_id']}")
    output.append(f"📄 Documents: {logs_data['document_count']}")
    output.append(f"📁 Cache Dir: {logs_data['cache_dir']}")
    output.append("")

    # System configuration
    config = logs_data["system_config"]
    output.append("⚙️  SYSTEM CONFIG:")
    output.append(f"   Chunk Size: {config['chunk_size']}")
    output.append(f"   Chunk Overlap: {config['chunk_overlap']}")
    output.append(f"   Max Tokens: {config['max_tokens']}")
    output.append(f"   Temperature: {config['temperature']}")
    output.append("")

    # Timing statistics
    timing_stats = logs_data["timing_stats"]
    if timing_stats:
        output.append("⏱️  PERFORMANCE TIMING:")
        for operation, time_taken in timing_stats.items():
            output.append(f"   {operation}: {time_taken:.2f}s")
    else:
        output.append("⏱️  PERFORMANCE TIMING: No timing data available yet")

    output.append("=" * 60)
    return "\n".join(output)


def monitor_logs(interval=5):
    """Monitor logs in real-time."""
    print("🔍 RAG LLM Log Monitor")
    print("=" * 50)
    print(f"Monitoring API at: {API_BASE}")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    last_sessions = set()

    try:
        while True:
            # Check API health
            if not check_api_health():
                print("❌ [", "API not responding")
                time.sleep(interval)
                continue

            # Get current sessions
            current_sessions = set(get_active_sessions())

            # Check for new sessions
            new_sessions = current_sessions - last_sessions
            if new_sessions:
                print("✅ [", f"New session(s): {', '.join(new_sessions)}")

            # Check for removed sessions
            removed_sessions = last_sessions - current_sessions
            if removed_sessions:
                print("🗑️  [", f"Removed session(s): {', '.join(removed_sessions)}")

            # Display logs for all active sessions
            if current_sessions:
                for session_id in current_sessions:
                    logs = get_session_logs(session_id)
                    if logs:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]")
                        print(format_logs(logs))
            else:
                print(f"📭 [{datetime.now().strftime('%H:%M:%S')}] No active sessions")

            last_sessions = current_sessions
            time.sleep(interval)

    except KeyboardInterrupt:
        print("👋 Log monitoring stopped")


def show_current_logs():
    """Show current logs for all active sessions."""
    print("📊 Current RAG LLM Logs")
    print("=" * 50)

    if not check_api_health():
        print("❌ API not responding. Make sure the API server is running.")
        return

    sessions = get_active_sessions()
    if not sessions:
        print("📭 No active sessions found.")
        print("Upload a PDF first to see logs.")
        return

    for session_id in sessions:
        logs = get_session_logs(session_id)
        if logs:
            print(format_logs(logs))
            print("")


def main():
    parser = argparse.ArgumentParser(description="Monitor RAG LLM logs")
    parser.add_argument(
        "--monitor", "-m", action="store_true", help="Monitor logs in real-time"
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=5,
        help="Monitoring interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--once", "-o", action="store_true", help="Show current logs once and exit"
    )

    args = parser.parse_args()

    if args.once:
        show_current_logs()
    elif args.monitor:
        monitor_logs(args.interval)
    else:
        # Default: show current logs
        show_current_logs()


if __name__ == "__main__":
    main()
