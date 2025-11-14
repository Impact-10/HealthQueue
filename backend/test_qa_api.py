"""Test script for the QA API endpoint.

Run:
    python -u test_qa_api.py

Starts API server in background and sends test queries.
"""
from __future__ import annotations

import time
import requests
import sys
from pathlib import Path

# Test queries
TEST_QUERIES = [
    "What causes diabetes?",
    "How to treat hypertension?",
    "What are symptoms of pneumonia?",
    "What is the treatment for asthma?",
    "What causes chest pain?",
]

API_BASE = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    resp = requests.get(f"{API_BASE}/health", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    print(f"✓ Health: {data.get('status')}")
    print(f"  BERT QA loaded: {data.get('models', {}).get('bert_qa', {}).get('loaded')}")
    print()


def test_qa(question: str, k: int = 5, confidence_threshold: float = 0.1):
    """Test QA endpoint."""
    print(f"Question: {question}")
    payload = {
        "question": question,
        "k": k,
        "confidence_threshold": confidence_threshold,
    }
    print("  Sending request (may take 1-2 min on first call to load models)...")
    resp = requests.post(f"{API_BASE}/api/qa", json=payload, timeout=180)  # 3 min timeout for model loading
    resp.raise_for_status()
    result = resp.json()
    
    mode = result.get("mode")
    answer = result.get("answer")
    confidence = result.get("confidence", 0.0)
    
    print(f"  Mode: {mode}")
    print(f"  Confidence: {confidence:.3f}")
    
    if answer:
        print(f"  Answer: {answer}")
        source = result.get("source", {})
        if source:
            print(f"  Source label: {source.get('label')}")
            print(f"  Source Q: {source.get('question', '')[:100]}...")
    else:
        print(f"  Message: {result.get('message')}")
    
    print()
    return result


def main():
    print("=" * 60)
    print("QA API Test Suite")
    print("=" * 60)
    print()
    
    # Wait for server to be ready
    print("Waiting for API server...")
    for attempt in range(10):
        try:
            requests.get(f"{API_BASE}/", timeout=2)
            print("✓ Server is ready\n")
            break
        except requests.exceptions.RequestException:
            time.sleep(2)
    else:
        print("✗ Server not responding. Start it with:")
        print("  uvicorn api.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Health check
    try:
        test_health()
    except Exception as e:
        print(f"✗ Health check failed: {e}\n")
    
    # Test queries
    print("=" * 60)
    print("Running Test Queries")
    print("=" * 60)
    print()
    
    results = []
    for query in TEST_QUERIES:
        try:
            result = test_qa(query, k=5, confidence_threshold=0.05)
            results.append((query, result))
        except Exception as e:
            print(f"✗ Error: {e}\n")
            results.append((query, None))
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    successful = sum(1 for _, r in results if r and r.get("mode") == "success")
    low_conf = sum(1 for _, r in results if r and r.get("mode") == "low_confidence")
    failed = sum(1 for _, r in results if r is None or r.get("mode") not in ["success", "low_confidence"])
    
    print(f"Total queries: {len(TEST_QUERIES)}")
    print(f"  Success: {successful}")
    print(f"  Low confidence: {low_conf}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
