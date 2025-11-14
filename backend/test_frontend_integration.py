#!/usr/bin/env python3
"""Test script to verify frontend integration - citations, safety warnings, low-confidence fallback."""
import requests
import json
import sys

API_URL = "http://localhost:8000/api/qa"

def test_citation_response():
    """Test that responses include proper source citations for frontend display."""
    print("=" * 70)
    print("Testing Frontend Integration - Citations & Metadata")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "Standard QA with citation",
            "question": "What causes diabetes?",
            "expected_fields": ["answer", "source", "confidence"],
        },
        {
            "name": "Query with safety warning",
            "question": "What causes chest pain?",
            "expected_fields": ["answer", "source", "safety_warning"],
        },
        {
            "name": "Low confidence fallback",
            "question": "What is a very obscure medical condition?",
            "expected_fields": ["top_passages", "mode"],
            "confidence_threshold": 0.95,
        },
    ]
    
    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"Test: {test['name']}")
        print(f"Question: {test['question']}")
        print("-" * 70)
        
        try:
            payload = {
                "question": test["question"],
                "k": 10,
                "alpha": 0.6,
                "max_answer_length": 150,
                "confidence_threshold": test.get("confidence_threshold", 0.1),
            }
            
            response = requests.post(API_URL, json=payload, timeout=30)
            
            if response.status_code != 200:
                print(f"✗ ERROR: Status {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                continue
            
            result = response.json()
            
            # Check expected fields
            missing_fields = []
            for field in test["expected_fields"]:
                if field not in result:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"✗ Missing fields: {missing_fields}")
            else:
                print("✓ All expected fields present")
            
            # Display key information
            print(f"\nResponse Structure:")
            print(f"  Mode: {result.get('mode', 'N/A')}")
            
            if "source" in result:
                source = result["source"]
                print(f"  Source Label: {source.get('label', 'N/A')}")
                print(f"  Source Type: {source.get('source_type', 'N/A')}")
                print(f"  Retrieval Score: {source.get('retrieval_score', 0):.3f}")
                print(f"  ✓ Source citation available for frontend")
            
            if "safety_warning" in result:
                print(f"  ⚠️  Safety Warning: {result['safety_warning'][:80]}...")
                print(f"  ✓ Safety warning available for frontend")
            
            if "top_passages" in result:
                passages = result["top_passages"]
                print(f"  Top Passages: {len(passages)} passages")
                if passages:
                    print(f"  ✓ Low-confidence fallback available for frontend")
                    print(f"    First passage source: {passages[0].get('source', 'N/A')}")
            
            if "answer" in result and result["answer"]:
                answer_preview = result["answer"][:100]
                print(f"  Answer Preview: {answer_preview}...")
            
            print(f"  Confidence: {result.get('confidence', 0):.3f}")
            
        except requests.exceptions.ConnectionError:
            print("✗ ERROR: Backend server not running")
            print("  Start server: cd backend && python -m uvicorn api.app:app --host 0.0.0.0 --port 8000")
            sys.exit(1)
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Frontend Integration Test Complete!")
    print("\nNext Steps:")
    print("  1. Restart backend server to load improved spell correction")
    print("  2. Start frontend: npm run dev")
    print("  3. Test in browser with BERT QA model selected")
    print("  4. Verify citations appear below answers")
    print("  5. Test with typos: 'diabetis', 'symptons', 'pnemonia'")
    print("=" * 70)

if __name__ == "__main__":
    test_citation_response()

