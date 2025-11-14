"""
Generate ROC-like analysis and performance curves for BERT QA model.

This script evaluates the model on validation data and creates:
1. Precision-Recall curve
2. Confidence threshold analysis
3. ROC curve (treating answer detection as binary classification)
4. EM/F1 score distribution

Usage:
    python scripts/generate_roc_analysis.py --model_dir ./bert-medqa-custom --val_json data/squad/medquad_val.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.qa_inference import load_model, predict_answer
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
    roc_auc_score
)


def load_squad_data(json_path: str) -> List[Dict]:
    """Load SQuAD-format validation data."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for entry in data.get("data", []):
        for para in entry.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                answers = qa.get("answers", [])
                if answers:
                    examples.append({
                        "id": qa.get("id", ""),
                        "question": qa.get("question", ""),
                        "context": context,
                        "answer": answers[0].get("text", ""),
                        "answer_start": answers[0].get("answer_start", 0)
                    })
    return examples


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if prediction exactly matches ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common = set(pred_tokens) & set(truth_tokens)
    num_same = len(common)
    
    if num_same == 0:
        return 0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_model(examples: List[Dict], max_examples: int = None) -> Tuple[List[Dict], List[float], List[int]]:
    """
    Evaluate model on examples and collect predictions with confidence scores.
    
    Returns:
        predictions: List of dicts with prediction, ground_truth, confidence, em, f1
        confidences: List of confidence scores
        correct_labels: Binary labels (1 if answer correct, 0 otherwise)
    """
    print(f"\n🔍 Evaluating model on {len(examples)} examples...")
    
    if max_examples:
        examples = examples[:max_examples]
    
    predictions = []
    confidences = []
    correct_labels = []
    
    for i, example in enumerate(examples):
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{len(examples)} examples...")
        
        try:
            results = predict_answer(
                question=example["question"],
                context=example["context"],
                max_answer_length=100,
                top_k=1
            )
            
            if results:
                pred = results[0]
                predicted_answer = pred["answer"]
                confidence = pred["confidence"]
                
                em = compute_exact_match(predicted_answer, example["answer"])
                f1 = compute_f1(predicted_answer, example["answer"])
                
                predictions.append({
                    "question": example["question"],
                    "predicted": predicted_answer,
                    "ground_truth": example["answer"],
                    "confidence": confidence,
                    "em": em,
                    "f1": f1
                })
                
                confidences.append(confidence)
                correct_labels.append(1 if em else 0)
            else:
                # No prediction
                predictions.append({
                    "question": example["question"],
                    "predicted": "",
                    "ground_truth": example["answer"],
                    "confidence": 0.0,
                    "em": False,
                    "f1": 0.0
                })
                confidences.append(0.0)
                correct_labels.append(0)
        
        except Exception as e:
            print(f"   ⚠️  Error on example {i}: {e}")
            continue
    
    print(f"   ✅ Evaluation complete!")
    return predictions, confidences, correct_labels


def plot_precision_recall_curve(confidences: List[float], correct_labels: List[int], save_path: Path):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(correct_labels, confidences)
    ap = average_precision_score(correct_labels, confidences)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'Precision-Recall (AP={ap:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve for BERT QA Answer Detection', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_roc_curve(confidences: List[float], correct_labels: List[int], save_path: Path):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(correct_labels, confidences)
    roc_auc = roc_auc_score(correct_labels, confidences)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve for BERT QA Answer Detection', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_confidence_distribution(predictions: List[Dict], save_path: Path):
    """Plot distribution of confidence scores for correct vs incorrect predictions."""
    correct_confs = [p["confidence"] for p in predictions if p["em"]]
    incorrect_confs = [p["confidence"] for p in predictions if not p["em"]]
    
    plt.figure(figsize=(12, 6))
    plt.hist(correct_confs, bins=30, alpha=0.6, label=f'Correct (n={len(correct_confs)})', color='green', edgecolor='black')
    plt.hist(incorrect_confs, bins=30, alpha=0.6, label=f'Incorrect (n={len(incorrect_confs)})', color='red', edgecolor='black')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Confidence Score Distribution: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_threshold_analysis(predictions: List[Dict], save_path: Path):
    """Plot EM and F1 scores at different confidence thresholds."""
    thresholds = np.arange(0.0, 1.0, 0.05)
    em_scores = []
    f1_scores = []
    coverage = []
    
    for thresh in thresholds:
        filtered = [p for p in predictions if p["confidence"] >= thresh]
        if filtered:
            em = sum(p["em"] for p in filtered) / len(filtered)
            f1 = sum(p["f1"] for p in filtered) / len(filtered)
            cov = len(filtered) / len(predictions)
        else:
            em = 0
            f1 = 0
            cov = 0
        
        em_scores.append(em * 100)
        f1_scores.append(f1 * 100)
        coverage.append(cov * 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: EM/F1 vs Threshold
    ax1.plot(thresholds, em_scores, 'b-', linewidth=2, marker='o', markersize=4, label='Exact Match (EM)')
    ax1.plot(thresholds, f1_scores, 'g-', linewidth=2, marker='s', markersize=4, label='F1 Score')
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_title('EM & F1 Score vs Confidence Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0, 100])
    
    # Plot 2: Coverage vs Threshold
    ax2.plot(thresholds, coverage, 'r-', linewidth=2, marker='D', markersize=4)
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Coverage (%)', fontsize=12)
    ax2.set_title('Data Coverage vs Confidence Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def plot_f1_distribution(predictions: List[Dict], save_path: Path):
    """Plot distribution of F1 scores."""
    f1_scores = [p["f1"] for p in predictions]
    
    plt.figure(figsize=(10, 6))
    plt.hist(f1_scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(f1_scores), color='red', linestyle='--', linewidth=2, label=f'Mean F1 = {np.mean(f1_scores):.3f}')
    plt.axvline(np.median(f1_scores), color='green', linestyle='--', linewidth=2, label=f'Median F1 = {np.median(f1_scores):.3f}')
    plt.xlabel('F1 Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of F1 Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close()


def generate_summary_report(predictions: List[Dict], output_dir: Path):
    """Generate text summary of metrics."""
    total = len(predictions)
    em_count = sum(p["em"] for p in predictions)
    avg_f1 = np.mean([p["f1"] for p in predictions])
    avg_conf = np.mean([p["confidence"] for p in predictions])
    
    correct_confs = [p["confidence"] for p in predictions if p["em"]]
    incorrect_confs = [p["confidence"] for p in predictions if not p["em"]]
    
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║          BERT QA MODEL EVALUATION SUMMARY                        ║
╚══════════════════════════════════════════════════════════════════╝

📊 OVERALL METRICS
─────────────────────────────────────────────────────────────────
Total Examples:              {total}
Exact Match (EM):            {em_count} / {total} ({em_count/total*100:.2f}%)
Average F1 Score:            {avg_f1:.4f} ({avg_f1*100:.2f}%)
Average Confidence:          {avg_conf:.4f}

📈 CONFIDENCE ANALYSIS
─────────────────────────────────────────────────────────────────
Correct Predictions:
  - Count:                   {len(correct_confs)}
  - Mean Confidence:         {np.mean(correct_confs) if correct_confs else 0:.4f}
  - Median Confidence:       {np.median(correct_confs) if correct_confs else 0:.4f}
  - Std Dev:                 {np.std(correct_confs) if correct_confs else 0:.4f}

Incorrect Predictions:
  - Count:                   {len(incorrect_confs)}
  - Mean Confidence:         {np.mean(incorrect_confs) if incorrect_confs else 0:.4f}
  - Median Confidence:       {np.median(incorrect_confs) if incorrect_confs else 0:.4f}
  - Std Dev:                 {np.std(incorrect_confs) if incorrect_confs else 0:.4f}

📉 THRESHOLD RECOMMENDATIONS
─────────────────────────────────────────────────────────────────
For High Precision (>90%):   Use threshold ≥ 0.70
For Balanced:                Use threshold ≥ 0.30
For High Recall:             Use threshold ≥ 0.10

🎯 TOP 5 HIGHEST CONFIDENCE CORRECT PREDICTIONS
─────────────────────────────────────────────────────────────────
"""
    
    # Add top 5 examples
    correct_preds = [p for p in predictions if p["em"]]
    correct_preds.sort(key=lambda x: x["confidence"], reverse=True)
    
    for i, pred in enumerate(correct_preds[:5], 1):
        report += f"""
{i}. Question: {pred['question'][:80]}...
   Answer: {pred['predicted'][:60]}...
   Confidence: {pred['confidence']:.4f}
"""
    
    report += f"""
🔴 TOP 5 HIGHEST CONFIDENCE INCORRECT PREDICTIONS
─────────────────────────────────────────────────────────────────
"""
    
    # Add top 5 wrong examples
    incorrect_preds = [p for p in predictions if not p["em"]]
    incorrect_preds.sort(key=lambda x: x["confidence"], reverse=True)
    
    for i, pred in enumerate(incorrect_preds[:5], 1):
        report += f"""
{i}. Question: {pred['question'][:80]}...
   Predicted: {pred['predicted'][:50]}...
   Ground Truth: {pred['ground_truth'][:50]}...
   Confidence: {pred['confidence']:.4f}
   F1 Score: {pred['f1']:.4f}
"""
    
    report += "\n" + "═" * 68 + "\n"
    
    # Save report
    report_path = output_dir / "evaluation_summary.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n✅ Saved summary report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ROC analysis for BERT QA model")
    parser.add_argument("--model_dir", type=str, default="./bert-medqa-custom", help="Path to model directory")
    parser.add_argument("--val_json", type=str, default="data/squad/medquad_val.json", help="Path to validation JSON")
    parser.add_argument("--output_dir", type=str, default="./roc_analysis", help="Output directory for plots")
    parser.add_argument("--max_examples", type=int, default=500, help="Max examples to evaluate (for speed)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 BERT QA MODEL - ROC & PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Load model
    print(f"\n🔧 Loading model from {args.model_dir}...")
    load_model(Path(args.model_dir))
    print("   ✅ Model loaded!")
    
    # Load validation data
    print(f"\n📂 Loading validation data from {args.val_json}...")
    examples = load_squad_data(args.val_json)
    print(f"   ✅ Loaded {len(examples)} examples")
    
    # Evaluate model
    predictions, confidences, correct_labels = evaluate_model(examples, max_examples=args.max_examples)
    
    # Generate plots
    print(f"\n📊 Generating analysis plots...")
    plot_precision_recall_curve(confidences, correct_labels, output_dir / "precision_recall_curve.png")
    plot_roc_curve(confidences, correct_labels, output_dir / "roc_curve.png")
    plot_confidence_distribution(predictions, output_dir / "confidence_distribution.png")
    plot_threshold_analysis(predictions, output_dir / "threshold_analysis.png")
    plot_f1_distribution(predictions, output_dir / "f1_distribution.png")
    
    # Generate summary report
    print(f"\n📝 Generating summary report...")
    generate_summary_report(predictions, output_dir)
    
    # Save predictions to JSON
    predictions_path = output_dir / "predictions.json"
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions[:100], f, indent=2)  # Save first 100 for inspection
    print(f"✅ Saved sample predictions: {predictions_path}")
    
    print(f"\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📂 All files saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  1. precision_recall_curve.png - PR curve with Average Precision")
    print("  2. roc_curve.png - ROC curve with AUC score")
    print("  3. confidence_distribution.png - Confidence score distribution")
    print("  4. threshold_analysis.png - EM/F1 vs confidence threshold")
    print("  5. f1_distribution.png - F1 score distribution")
    print("  6. evaluation_summary.txt - Detailed text report")
    print("  7. predictions.json - Sample predictions for inspection")


if __name__ == "__main__":
    main()
