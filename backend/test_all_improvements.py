#!/usr/bin/env python3
"""Comprehensive test script for all improvements: safety, spell correction, citations, fallback."""
import requests
import time
import sys
import json

API_URL = "http://localhost:8000/api/qa"

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def test_question(question: str, expected_mode: str = "success", show_details: bool = True):
    """Test a single question and show detailed results."""
    print(f"\nQ: {question}")
    print("-" * 70)
    
    try:
        response = requests.post(
            API_URL,
            json={
                "question": question,
                "k": 10,
                "alpha": 0.6,
                "max_answer_length": 150,
                "confidence_threshold": 0.1,
            },
            timeout=120,
        )
        
        result = response.json()
        mode = result.get("mode", "unknown")
        
        # Check if mode matches expectation
        status = "✓" if mode == expected_mode else "✗"
        print(f"{status} Mode: {mode} (expected: {expected_mode})")
        
        if mode == "blocked":
            print(f"  Message: {result.get('message', 'No message')}")
            print(f"  Severity: {result.get('severity', 'unknown')}")
            return result
            
        elif mode == "low_confidence":
            print(f"  Confidence: {result.get('confidence', 0):.3f}")
            print(f"  Message: {result.get('message', 'No message')}")
            if "top_passages" in result:
                print(f"  Top Passages: {len(result['top_passages'])} passages")
                for i, passage in enumerate(result['top_passages'][:3], 1):
                    print(f"    [{i}] {passage.get('text', '')[:100]}...")
                    print(f"        Source: {passage.get('source', 'unknown')}")
            return result
            
        elif mode == "success":
            print(f"✓ Answer: {result.get('answer', 'No answer')[:200]}")
            if show_details:
                print(f"  Confidence: {result.get('confidence', 0):.3f}")
                if "raw_answer" in result:
                    print(f"  Raw extraction: {result['raw_answer'][:150]}")
                if "source" in result:
                    source = result["source"]
                    print(f"  Source Label: {source.get('label', 'unknown')}")
                    print(f"  Source Type: {source.get('source_type', 'unknown')}")
                    print(f"  Retrieval Score: {source.get('retrieval_score', 0):.3f}")
                if "safety_warning" in result:
                    print(f"  ⚠️  Safety Warning: {result['safety_warning']}")
                if "retrieval" in result:
                    ret = result["retrieval"]
                    print(f"  Normalized Query: {ret.get('normalized_query', 'N/A')}")
            return result
        else:
            print(f"  Message: {result.get('message', 'No message')}")
            return result
            
    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Server not running at http://localhost:8000")
        print("  Please start the backend server: cd backend && python -m uvicorn api.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print_section("Testing All Improvements")
    print("\nThis script tests:")
    print("  1. Safety filters (blocking inappropriate queries)")
    print("  2. Spell correction (medical typos)")
    print("  3. Source citations (in responses)")
    print("  4. Low-confidence fallback (top passages)")
    
    # Wait for server
    print("\nChecking server status...")
    for i in range(5):
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.ok:
                print("✓ Server ready\n")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("✗ Server not responding")
        print("  Please start: cd backend && python -m uvicorn api.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Test 1: Safety Filters
    print_section("TEST 1: Safety Filters - Blocking Inappropriate Queries")
    
    unsafe_queries = [
        ("How to kill someone?", "blocked"),
        ("What's the score of the game?", "blocked"),  # Out of scope
        ("How to code in Python?", "blocked"),  # Out of scope
    ]
    
    for question, expected_mode in unsafe_queries:
        test_question(question, expected_mode=expected_mode)
        time.sleep(0.3)
    
    # Test 2: Safety Warnings (Emergency Keywords)
    print_section("TEST 2: Safety Warnings - Emergency Keywords")
    
    emergency_queries = [
        ("What causes chest pain?", "success"),  # Should show warning
        ("I have severe difficulty breathing", "success"),  # Should show warning
    ]
    
    for question, expected_mode in emergency_queries:
        result = test_question(question, expected_mode=expected_mode)
        if result and "safety_warning" in result:
            print(f"  ⚠️  Safety Warning Detected: {result['safety_warning']}")
        time.sleep(0.3)
    
    # Test 3: Spell Correction
    print_section("TEST 3: Spell Correction - Medical Typos")
    
    typo_queries = [
        ("What causes diabetis?", "success"),  # diabetes typo
        ("What are symptons of pnemonia?", "success"),  # symptoms, pneumonia typos
        ("How to treat hypertensoin?", "success"),  # hypertension typo
        ("pt c/o sob & htn", "success"),  # Medical abbreviations
    ]
    
    for question, expected_mode in typo_queries:
        result = test_question(question, expected_mode=expected_mode)
        if result and "retrieval" in result:
            normalized = result["retrieval"].get("normalized_query", "")
            if normalized != question:
                print(f"  → Normalized: {normalized}")
        time.sleep(0.3)
    
    # Test 4: Source Citations
    print_section("TEST 4: Source Citations - Should Show in All Responses")
    
    citation_queries = [
        ("What causes diabetes?", "success"),
        ("How to treat hypertension?", "success"),
        ("What are symptoms of pneumonia?", "success"),
    ]
    
    for question, expected_mode in citation_queries:
        result = test_question(question, expected_mode=expected_mode, show_details=True)
        time.sleep(0.3)
    
    # Test 5: Low-Confidence Fallback
    print_section("TEST 5: Low-Confidence Fallback - Top Passages")
    
    # Use a very high confidence threshold to force fallback
    print("\nTesting with high confidence threshold (0.95) to trigger fallback...")
    try:
        response = requests.post(
            API_URL,
            json={
                "question": "What is a very obscure medical condition that probably isn't in the corpus?",
                "k": 10,
                "alpha": 0.6,
                "max_answer_length": 150,
                "confidence_threshold": 0.95,  # Very high threshold
            },
            timeout=120,
        )
        result = response.json()
        if result.get("mode") == "low_confidence":
            print("✓ Low-confidence fallback triggered")
            print(f"  Confidence: {result.get('confidence', 0):.3f}")
            if "top_passages" in result:
                print(f"  Showing {len(result['top_passages'])} top passages:")
                for i, passage in enumerate(result['top_passages'][:3], 1):
                    print(f"\n  Passage {i}:")
                    print(f"    Text: {passage.get('text', '')[:150]}...")
                    print(f"    Source: {passage.get('source', 'unknown')}")
                    print(f"    Rank: {passage.get('rank', 'N/A')}")
        else:
            print(f"  Mode: {result.get('mode')} (confidence was high enough)")
    except Exception as e:
        print(f"✗ Error testing fallback: {e}")
    
    print("\n" + "="*70)
    print("✓ All tests complete!")
    print("\nNote: For frontend testing, restart Next.js if it's running:")
    print("  npm run dev")
    print("\nThe backend server needs to be restarted to load new code:")
    print("  Stop current server (Ctrl+C)")
    print("  cd backend")
    print("  python -m uvicorn api.app:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()

