"""
BERT Fine-tuning for Medical Specialty Classification
Fine-tunes bert-base-uncased on medical transcriptions for multi-class classification

Model: bert-base-uncased (110M parameters)
Task: Medical specialty classification (40 classes)
Hardware: GPU-optimized training (RTX 3050 6GB VRAM)
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Configuration
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed"
MODEL_DIR = SCRIPT_DIR.parent / "bert-custom-model"
LOGS_DIR = SCRIPT_DIR.parent / "training_logs"

# Training hyperparameters
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 4  # Reduced to 4 for 6GB VRAM
LEARNING_RATE = 3e-5
NUM_EPOCHS = 10  # Increased for better learning
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16

# Paths
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"

def check_gpu():
    """Verify GPU availability and configuration."""
    print("\n🔍 GPU CONFIGURATION CHECK")
    print("=" * 60)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ CUDA available: YES")
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        
        # Test GPU operation
        try:
            test_tensor = torch.randn(100, 100).cuda()
            _ = test_tensor @ test_tensor
            torch.cuda.synchronize()
            print(f"   GPU Test: ✅ Working")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   GPU Test: ❌ Failed - {e}")
            return False
        
        print(f"\n🚀 Training will use GPU: {gpu_name}")
        return True
    else:
        print(f"⚠️  CUDA available: NO")
        print(f"   Training will use CPU (VERY SLOW)")
        print(f"   Recommendation: Install CUDA-enabled PyTorch")
        print(f"   Visit: https://pytorch.org/get-started/locally/")
        
        response = input("\nContinue with CPU training? This will be VERY slow (y/n): ").lower()
        if response != 'y':
            print("Exiting...")
            exit(0)
        
        return False

def load_label_map():
    """Load label mapping from preprocessing."""
    print("\n📋 Loading label mapping...")
    
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    
    num_classes = label_map['num_classes']
    print(f"   Number of classes: {num_classes}")
    
    return label_map, num_classes

def load_datasets():
    """Load preprocessed train and validation datasets."""
    print("\n📂 Loading datasets...")
    
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    
    print(f"   Train samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    
    return train_df, val_df

def tokenize_data(train_df, val_df, tokenizer):
    """Tokenize text data for BERT input."""
    print("\n🔤 Tokenizing data...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None
        )
    
    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_dict({
        'text': train_df['transcription'].tolist(),
        'label': train_df['label'].tolist()
    })
    
    val_dataset = Dataset.from_dict({
        'text': val_df['transcription'].tolist(),
        'label': val_df['label'].tolist()
    })
    
    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    print(f"   ✅ Tokenization complete")
    
    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    """Compute evaluation metrics during training."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(train_dataset, val_dataset, num_classes, gpu_available):
    """Fine-tune BERT model."""
    print("\n🚀 BERT FINE-TUNING")
    print("=" * 60)
    
    # Load tokenizer and model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        problem_type="single_label_classification"
    )
    
    # Training arguments
    device = "cuda" if gpu_available else "cpu"
    print(f"Training device: {device.upper()}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = MODEL_DIR / f"checkpoint_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_dir=str(LOGS_DIR / timestamp),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=gpu_available,  # Mixed precision training on GPU
        dataloader_num_workers=0,  # Windows compatibility
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=True,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\n🎯 Starting training...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Total steps: ~{len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS}")
    print("\n" + "=" * 60)
    
    train_result = trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    final_model_dir = MODEL_DIR / "final"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    print(f"   ✅ Model saved to: {final_model_dir}")
    
    return trainer, train_result

def evaluate_model(trainer, val_dataset, label_map):
    """Evaluate trained model on validation set."""
    print("\n📊 FINAL EVALUATION")
    print("=" * 60)
    
    # Get predictions
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Overall metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Per-class report
    print("\n📋 Classification Report:")
    print("-" * 60)
    # Build human-readable class names robustly across label_map variants
    if 'label_to_specialty' in label_map:
        target_names = [label_map['label_to_specialty'][str(i)] for i in range(len(label_map['label_to_specialty']))]
    elif 'classes' in label_map:
        target_names = label_map['classes']
    else:
        target_names = [str(i) for i in range(int(label_map.get('num_classes', max(true_labels)+1)))]
    print(classification_report(true_labels, pred_labels, target_names=target_names, zero_division=0))
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'training_loss': float(predictions.metrics.get('test_loss', 0))
    }
    
    metrics_path = MODEL_DIR / "final" / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Metrics saved to: {metrics_path}")
    
    return metrics

def display_sample_predictions(trainer, val_dataset, label_map, num_samples=5):
    """Display sample predictions for verification."""
    print(f"\n🔍 SAMPLE PREDICTIONS (First {num_samples})")
    print("=" * 60)
    
    # Get first few samples
    sample_dataset = val_dataset.select(range(min(num_samples, len(val_dataset))))
    predictions = trainer.predict(sample_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    # Build mappings in a backward-compatible way
    if 'label_to_specialty' in label_map:
        label_to_specialty = label_map['label_to_specialty']
    elif 'classes' in label_map:
        label_to_specialty = {str(i): name for i, name in enumerate(label_map['classes'])}
    else:
        label_to_specialty = {str(i): str(i) for i in range(int(label_map.get('num_classes', 0)))}
    
    for i, (pred, true) in enumerate(zip(pred_labels, sample_dataset['label'])):
        pred_specialty = label_to_specialty[str(pred)]
        true_specialty = label_to_specialty[str(true.item())]
        
        match = "✅" if pred == true else "❌"
        print(f"\nSample {i+1}: {match}")
        print(f"  True:      {true_specialty}")
        print(f"  Predicted: {pred_specialty}")

def save_training_summary(train_result, metrics, gpu_available):
    """Save training summary to README."""
    summary_path = MODEL_DIR / "TRAINING_SUMMARY.md"
    
    with open(summary_path, 'w') as f:
        f.write("# BERT Fine-Tuning Summary\n\n")
        f.write("## Model Configuration\n")
        f.write(f"- Base Model: {MODEL_NAME}\n")
        f.write(f"- Task: Medical Specialty Classification\n")
        f.write(f"- Number of Classes: {metrics.get('num_classes', 'N/A')}\n")
        f.write(f"- Training Device: {'GPU (CUDA)' if gpu_available else 'CPU'}\n\n")
        
        f.write("## Training Hyperparameters\n")
        f.write(f"- Epochs: {NUM_EPOCHS}\n")
        f.write(f"- Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})\n")
        f.write(f"- Learning Rate: {LEARNING_RATE}\n")
        f.write(f"- Max Sequence Length: {MAX_LENGTH}\n")
        f.write(f"- Weight Decay: {WEIGHT_DECAY}\n\n")
        
        f.write("## Results\n")
        f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"- Precision: {metrics['precision']:.4f}\n")
        f.write(f"- Recall: {metrics['recall']:.4f}\n")
        f.write(f"- F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"- Training Loss: {metrics.get('training_loss', 'N/A')}\n\n")
        
        f.write("## Training Time\n")
        if hasattr(train_result.metrics, 'train_runtime'):
            runtime = train_result.metrics['train_runtime']
            f.write(f"- Total: {runtime/60:.2f} minutes\n")
            f.write(f"- Samples/second: {train_result.metrics.get('train_samples_per_second', 'N/A'):.2f}\n\n")
        
        f.write("## Usage\n")
        f.write("```python\n")
        f.write("from transformers import BertTokenizer, BertForSequenceClassification\n\n")
        f.write("tokenizer = BertTokenizer.from_pretrained('./bert-custom-model/final')\n")
        f.write("model = BertForSequenceClassification.from_pretrained('./bert-custom-model/final')\n")
        f.write("```\n")
    
    print(f"\n📄 Training summary saved to: {summary_path}")

def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("🏥 BERT FINE-TUNING FOR MEDICAL SPECIALTY CLASSIFICATION")
    print("=" * 60)
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Load data
    label_map, num_classes = load_label_map()
    train_df, val_df = load_datasets()
    
    # Load tokenizer
    print(f"\n🔤 Loading tokenizer: {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenize
    train_dataset, val_dataset = tokenize_data(train_df, val_df, tokenizer)
    
    # Train
    trainer, train_result = train_model(train_dataset, val_dataset, num_classes, gpu_available)
    
    # Evaluate
    metrics = evaluate_model(trainer, val_dataset, label_map)
    metrics['num_classes'] = num_classes
    
    # Sample predictions
    display_sample_predictions(trainer, val_dataset, label_map)
    
    # Save summary
    save_training_summary(train_result, metrics, gpu_available)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Model location: {MODEL_DIR / 'final'}")
    print(f"📊 Metrics: {metrics}")
    print(f"\n🚀 Next step: Run test_bert.py for inference testing")

if __name__ == "__main__":
    main()
