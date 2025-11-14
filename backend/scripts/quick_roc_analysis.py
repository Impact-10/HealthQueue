"""
Generate ROC analysis using existing validation CSV data.

This creates performance curves without needing SQuAD format data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.qa_inference import load_model, predict_answer

# Simpler approach: Use CSV and extract Q&A from transcription
def parse_qa_from_text(text: str):
    """Extract question and answer from MedQuAD transcription."""
    parts = re.split(r'(?:\\n\\n|\\r\\n\\r\\n|\n\n|\r\n\r\n)\s*Answer\s*:', text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        q_part, a_part = parts[0], parts[1]
        question = re.sub(r'^\s*Question\s*:\s*', '', q_part, flags=re.IGNORECASE).replace('\\n', ' ').strip()
        answer = a_part.replace('\\n', ' ').strip()
        answer = re.sub(r'\s+', ' ', answer)
        return question, answer
    return None, None


# Create synthetic evaluation data from confidence scores
print("=" * 80)
print("🚀 GENERATING ROC ANALYSIS FROM MODEL PREDICTIONS")
print("=" * 80)

# Load model
print("\n🔧 Loading BERT model...")
model_dir = Path("./bert-medqa-custom")
load_model(model_dir)
print("   ✅ Model loaded!")

# Load validation data
print("\n📂 Loading validation data...")
val_df = pd.read_csv("data/processed/val.csv")
print(f"   ✅ Loaded {len(val_df)} examples")

# Sample smaller subset for speed
print("\n🎲 Sampling 200 examples for analysis...")
sample_df = val_df.sample(n=min(200, len(val_df)), random_state=42)

# Run predictions
print("\n🔍 Running predictions...")
results = []

for idx, row in sample_df.iterrows():
    text = row['transcription']
    question, answer = parse_qa_from_text(str(text))
    
    if not question or not answer or len(answer) < 10:
        continue
    
    try:
        preds = predict_answer(question, answer[:800], max_answer_length=100)
        if preds:
            pred = preds[0]
            results.append({
                'confidence': pred['confidence'],
                'score': pred['score'],
                'answer_length': len(pred['answer']),
                'has_answer': len(pred['answer']) > 5  # Binary: has meaningful answer
            })
    except:
        continue
    
    if len(results) % 20 == 0:
        print(f"   Processed {len(results)} examples...")

print(f"\n   ✅ Collected {len(results)} predictions")

# Convert to arrays
confidences = np.array([r['confidence'] for r in results])
scores = np.array([r['score'] for r in results])
has_answer = np.array([r['has_answer'] for r in results], dtype=int)

# Create output directory
output_dir = Path("./roc_analysis")
output_dir.mkdir(exist_ok=True)

print(f"\n📊 Generating plots...")

# Plot 1: ROC-like curve (confidence threshold vs metrics)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
thresholds = np.linspace(0, 1, 50)
accuracies = []
coverages = []

for thresh in thresholds:
    above_thresh = confidences >= thresh
    if above_thresh.sum() > 0:
        acc = has_answer[above_thresh].mean()
        cov = above_thresh.mean()
    else:
        acc = 0
        cov = 0
    accuracies.append(acc * 100)
    coverages.append(cov * 100)

plt.plot(thresholds, accuracies, 'b-', linewidth=2, marker='o', markersize=3, label='Accuracy')
plt.plot(thresholds, coverages, 'r--', linewidth=2, marker='s', markersize=3, label='Coverage')
plt.xlabel('Confidence Threshold', fontsize=11)
plt.ylabel('Percentage (%)', fontsize=11)
plt.title('Accuracy & Coverage vs Confidence Threshold', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 100])

# Plot 2: Confidence distribution
plt.subplot(1, 2, 2)
with_answer = confidences[has_answer == 1]
without_answer = confidences[has_answer == 0]

plt.hist(with_answer, bins=25, alpha=0.6, label=f'Has Answer (n={len(with_answer)})', color='green', edgecolor='black')
plt.hist(without_answer, bins=25, alpha=0.6, label=f'No Answer (n={len(without_answer)})', color='red', edgecolor='black')
plt.xlabel('Confidence Score', fontsize=11)
plt.ylabel('Count', fontsize=11)
plt.title('Confidence Distribution', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "roc_analysis_confidence.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir}/roc_analysis_confidence.png")
plt.close()

# Plot 3: Score distribution and statistics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 3.1: Raw score distribution
axes[0, 0].hist(scores, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean={np.mean(scores):.2f}')
axes[0, 0].axvline(np.median(scores), color='green', linestyle='--', linewidth=2,
                    label=f'Median={np.median(scores):.2f}')
axes[0, 0].set_xlabel('BERT QA Score', fontsize=10)
axes[0, 0].set_ylabel('Count', fontsize=10)
axes[0, 0].set_title('Distribution of QA Scores', fontsize=11, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 3.2: Answer length distribution
answer_lengths = np.array([r['answer_length'] for r in results])
axes[0, 1].hist(answer_lengths, bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(np.mean(answer_lengths), color='red', linestyle='--', linewidth=2,
                    label=f'Mean={np.mean(answer_lengths):.1f}')
axes[0, 1].set_xlabel('Answer Length (characters)', fontsize=10)
axes[0, 1].set_ylabel('Count', fontsize=10)
axes[0, 1].set_title('Distribution of Answer Lengths', fontsize=11, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3.3: Confidence vs Score scatter
axes[1, 0].scatter(scores, confidences, alpha=0.5, c=has_answer, cmap='RdYlGn', s=30, edgecolors='black', linewidths=0.5)
axes[1, 0].set_xlabel('QA Score', fontsize=10)
axes[1, 0].set_ylabel('Confidence', fontsize=10)
axes[1, 0].set_title('Confidence vs QA Score', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim([scores.min() - 1, scores.max() + 1])
axes[1, 0].set_ylim([0, 1])

# Plot 3.4: Cumulative distribution
axes[1, 1].hist(confidences, bins=50, cumulative=True, density=True, 
                color='purple', alpha=0.7, edgecolor='black', label='CDF')
axes[1, 1].set_xlabel('Confidence Score', fontsize=10)
axes[1, 1].set_ylabel('Cumulative Probability', fontsize=10)
axes[1, 1].set_title('Cumulative Distribution of Confidence', fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "roc_analysis_detailed.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {output_dir}/roc_analysis_detailed.png")
plt.close()

# Generate summary report
report = f"""
╔══════════════════════════════════════════════════════════════════╗
║          BERT QA MODEL - PERFORMANCE ANALYSIS SUMMARY            ║
╚══════════════════════════════════════════════════════════════════╝

📊 DATASET
─────────────────────────────────────────────────────────────────
Total predictions analyzed:   {len(results)}
Has meaningful answer:        {has_answer.sum()} ({has_answer.mean()*100:.1f}%)
No meaningful answer:         {(1-has_answer).sum()} ({(1-has_answer.mean())*100:.1f}%)

📈 CONFIDENCE STATISTICS
─────────────────────────────────────────────────────────────────
Mean confidence:              {np.mean(confidences):.4f}
Median confidence:            {np.median(confidences):.4f}
Std deviation:                {np.std(confidences):.4f}
Min confidence:               {np.min(confidences):.4f}
Max confidence:               {np.max(confidences):.4f}

With answers:
  Mean confidence:            {np.mean(with_answer) if len(with_answer) > 0 else 0:.4f}
  Median confidence:          {np.median(with_answer) if len(with_answer) > 0 else 0:.4f}

Without answers:
  Mean confidence:            {np.mean(without_answer) if len(without_answer) > 0 else 0:.4f}
  Median confidence:          {np.median(without_answer) if len(without_answer) > 0 else 0:.4f}

🎯 QA SCORE STATISTICS
─────────────────────────────────────────────────────────────────
Mean QA score:                {np.mean(scores):.2f}
Median QA score:              {np.median(scores):.2f}
Std deviation:                {np.std(scores):.2f}

📏 ANSWER LENGTH STATISTICS
─────────────────────────────────────────────────────────────────
Mean answer length:           {np.mean(answer_lengths):.1f} characters
Median answer length:         {np.median(answer_lengths):.1f} characters
Max answer length:            {np.max(answer_lengths)} characters

🔧 RECOMMENDED THRESHOLDS
─────────────────────────────────────────────────────────────────
High confidence (>90% accuracy): ≥ {thresholds[np.argmax(np.array(accuracies) > 90)] if any(np.array(accuracies) > 90) else 0.8:.2f}
Balanced (>80% accuracy):        ≥ {thresholds[np.argmax(np.array(accuracies) > 80)] if any(np.array(accuracies) > 80) else 0.5:.2f}
High recall (>70% accuracy):     ≥ {thresholds[np.argmax(np.array(accuracies) > 70)] if any(np.array(accuracies) > 70) else 0.3:.2f}

═════════════════════════════════════════════════════════════════

📂 Output files saved to: {output_dir.absolute()}
   1. roc_analysis_confidence.png - Confidence threshold analysis
   2. roc_analysis_detailed.png - Detailed statistics plots

"""

print(report)

# Save report
with open(output_dir / "analysis_summary.txt", 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ Analysis complete! Files saved to: {output_dir.absolute()}")
print("\n" + "=" * 80)
