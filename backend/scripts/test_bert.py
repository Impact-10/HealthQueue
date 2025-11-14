"""
BERT Model Testing and Inference
Test fine-tuned BERT model on medical transcriptions

Loads trained model and runs inference on test set + custom examples
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
MODEL_DIR = SCRIPT_DIR.parent / "bert-custom-model" / "final"

# Paths
TEST_PATH = DATA_DIR / "test.csv"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"

def check_model_exists():
    """Verify trained model exists."""
    if not MODEL_DIR.exists():
        print(f"❌ Model not found at: {MODEL_DIR}")
        print(f"\n📌 Please run train_bert.py first to fine-tune the model")
        return False
    
    print(f"✅ Model found at: {MODEL_DIR}")
    return True

def load_model_and_tokenizer():
    """Load fine-tuned BERT model and tokenizer."""
    print("\n🔄 Loading model...")
    
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_DIR))
    model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    return model, tokenizer, device

def load_label_map():
    """Load label mapping."""
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    return label_map

def build_label_mappings(label_map):
    """Build forward/backward label mappings in a backward-compatible way.

    Supports two schemas:
    1) Old schema with 'label_to_specialty' and 'specialty_to_label'
    2) MedQuAD schema with 'classes' (list) and 'encoding' (name->id)
    """
    if 'label_to_specialty' in label_map:
        label_to_specialty = label_map['label_to_specialty']
        specialty_to_label = label_map.get('specialty_to_label') or {v: int(k) for k, v in label_to_specialty.items()}
        classes = [label_to_specialty[str(i)] for i in range(len(label_to_specialty))]
    else:
        classes = label_map.get('classes', [])
        label_to_specialty = {str(i): name for i, name in enumerate(classes)}
        specialty_to_label = label_map.get('encoding') or {name: i for i, name in enumerate(classes)}
    num_classes = int(label_map.get('num_classes', len(classes) if classes else max([int(k) for k in label_to_specialty.keys()], default=-1) + 1))
    return label_to_specialty, specialty_to_label, classes, num_classes

def predict_text(text, model, tokenizer, device, label_map, max_length=512):
    """
    Predict medical specialty for a given text.
    
    Args:
        text: Medical transcription text
        model: Fine-tuned BERT model
        tokenizer: BERT tokenizer
        device: torch device
        label_map: Label mapping dictionary
        max_length: Maximum sequence length
    
    Returns:
        predicted_specialty, confidence, all_probabilities
    """
    # Tokenize
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Get prediction
    pred_label = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][pred_label].item()
    
    l2s, _, _, _ = build_label_mappings(label_map)
    predicted_specialty = l2s[str(pred_label)]
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities[0], k=min(3, len(probabilities[0])))
    top3_predictions = [
        (l2s[str(idx.item())], prob.item())
        for idx, prob in zip(top3_indices, top3_probs)
    ]
    
    return predicted_specialty, confidence, top3_predictions

def evaluate_test_set(model, tokenizer, device, label_map):
    """Evaluate model on test set."""
    print("\n📊 EVALUATING ON TEST SET")
    print("=" * 60)
    
    # Load test data
    test_df = pd.read_csv(TEST_PATH)
    print(f"Test samples: {len(test_df)}")
    
    predictions = []
    true_labels = test_df['label'].tolist()
    
    print("\n🔄 Running predictions...")
    l2s, s2l, classes, num_classes = build_label_mappings(label_map)
    for text in test_df['transcription']:
        pred_specialty, _, _ = predict_text(text, model, tokenizer, device, label_map)
        pred_label = s2l[pred_specialty]
        predictions.append(pred_label)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    target_names = classes if classes else [l2s[str(i)] for i in range(num_classes)]
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report_text = classification_report(true_labels, predictions, target_names=target_names, zero_division=0)
    print(report_text)

    # Save report and confusion matrix
    results_dir = Path(MODEL_DIR) / "test_results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "classification_report.txt").write_text(report_text)

    cm = confusion_matrix(true_labels, predictions, labels=list(range(len(target_names))))
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(results_dir / "confusion_matrix.csv")

    # Optional heatmap if seaborn/matplotlib are available
    try:
        import seaborn as sns  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=False, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix.png")
        plt.close()
    except Exception:
        pass

    return predictions, true_labels

def test_custom_examples(model, tokenizer, device, label_map):
    """Test model on custom medical examples."""
    print("\n🧪 TESTING CUSTOM EXAMPLES")
    print("=" * 60)
    print("Note: This model predicts MedQuAD source categories (e.g., GARD, GHR) — not clinical specialties like Cardiology.")
    
    examples = [
        {
            "text": "Patient presents with chest pain radiating to left arm. ECG shows ST elevation. "
                   "Diagnosis: Acute myocardial infarction. Started on aspirin and beta blockers.",
            "expected": "Cardiology"
        },
        {
            "text": "Patient complains of severe headache with photophobia and nausea. "
                   "Neurological examination reveals no focal deficits. Diagnosis: Migraine. "
                   "Prescribed sumatriptan for acute management.",
            "expected": "Neurology"
        },
        {
            "text": "Skin lesion on forearm showing irregular borders and color variation. "
                   "Dermoscopy performed. Biopsy recommended to rule out melanoma. "
                   "Patient counseled on sun protection.",
            "expected": "Dermatology"
        },
        {
            "text": "Patient with chronic knee pain, worse with activity. X-ray shows joint space narrowing. "
                   "Diagnosis: Osteoarthritis. Treatment plan includes physical therapy and NSAIDs.",
            "expected": "Orthopedics"
        },
        {
            "text": "Abdominal pain and bloating for 3 weeks. Endoscopy reveals gastritis. "
                   "H. pylori test positive. Started triple therapy regimen.",
            "expected": "Gastroenterology"
        }
    ]
    
    correct = 0
    for i, example in enumerate(examples, 1):
        predicted, confidence, top3 = predict_text(
            example["text"], model, tokenizer, device, label_map
        )
        
        is_correct = predicted.lower() == example["expected"].lower()
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"\n{status} Example {i}:")
        print(f"   Expected:  {example['expected']}")
        print(f"   Predicted: {predicted} ({confidence:.2%} confidence)")
        print(f"   Top 3:")
        for specialty, prob in top3:
            print(f"      - {specialty}: {prob:.2%}")
        print(f"   Text: {example['text'][:100]}...")
    
    accuracy = correct / len(examples)
    print(f"\n📊 Custom Examples Accuracy: {accuracy:.2%} ({correct}/{len(examples)})")

def interactive_mode(model, tokenizer, device, label_map):
    """Interactive prediction mode."""
    print("\n💬 INTERACTIVE MODE")
    print("=" * 60)
    print("Enter medical transcription text to get specialty prediction")
    print("Type 'quit' to exit\n")
    
    while True:
        text = input("📝 Enter text (or 'quit'): ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 Exiting interactive mode")
            break
        
        if not text:
            print("⚠️  Please enter some text")
            continue
        
        predicted, confidence, top3 = predict_text(text, model, tokenizer, device, label_map)
        
        print(f"\n🎯 Prediction: {predicted}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"\n   Top 3 predictions:")
        for specialty, prob in top3:
            print(f"      {specialty}: {prob:.2%}")
        print()

def save_test_results(predictions, true_labels, label_map):
    """Save test results to file."""
    results_dir = Path(MODEL_DIR) / "test_results"
    results_dir.mkdir(exist_ok=True)
    
    # Save predictions
    l2s, _, _, _ = build_label_mappings(label_map)
    results_df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predictions,
        'true_specialty': [l2s[str(l)] for l in true_labels],
        'predicted_specialty': [l2s[str(l)] for l in predictions],
        'correct': [t == p for t, p in zip(true_labels, predictions)]
    })
    
    results_path = results_dir / "predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Test results saved to: {results_path}")

def main():
    """Main testing pipeline."""
    print("\n" + "=" * 60)
    print("🧪 BERT MODEL TESTING & INFERENCE")
    print("=" * 60)
    
    # Check model exists
    if not check_model_exists():
        return
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer()
    label_map = load_label_map()
    
    # Test on test set
    predictions, true_labels = evaluate_test_set(model, tokenizer, device, label_map)
    
    # Save results
    save_test_results(predictions, true_labels, label_map)
    
    # Test custom examples
    test_custom_examples(model, tokenizer, device, label_map)
    
    # Interactive mode
    print("\n" + "=" * 60)
    response = input("\n🤔 Enter interactive mode? (y/n): ").lower()
    if response == 'y':
        interactive_mode(model, tokenizer, device, label_map)
    
    print("\n" + "=" * 60)
    print("✅ TESTING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
