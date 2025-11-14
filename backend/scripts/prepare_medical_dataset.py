"""
Medical Transcription Dataset Preparation for BERT Fine-tuning
Downloads and prepares medical transcriptions dataset for specialty classification

Dataset: Medical Transcriptions from Kaggle/MTSamples
Task: Multi-class classification (40 medical specialties)
Source: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_PATH = DATA_DIR / "mtsamples.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"

# Dataset parameters
MIN_SAMPLES_PER_CLASS = 20  # Minimum samples per specialty to include
MAX_LENGTH = 512  # BERT max sequence length
RANDOM_SEED = 42

def download_dataset():
    """
    Download medical transcriptions dataset.
    
    Note: This dataset should be manually downloaded from Kaggle:
    https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
    
    Save as: backend/data/mtsamples.csv
    """
    if not RAW_DATA_PATH.exists():
        print("❌ Dataset not found!")
        print(f"\n📥 Please download the Medical Transcriptions dataset:")
        print("   1. Visit: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions")
        print("   2. Download 'mtsamples.csv'")
        print(f"   3. Place it at: {RAW_DATA_PATH.absolute()}")
        print("\n💡 Alternative: Use the backup script to generate synthetic data")
        return False
    
    print(f"✅ Dataset found at: {RAW_DATA_PATH}")
    return True

def load_and_clean_data():
    """Load and perform initial cleaning of medical transcription data."""
    print("\n📂 Loading dataset...")
    
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   Raw dataset size: {len(df)} samples")
    print(f"   Columns: {list(df.columns)}")
    
    # Keep relevant columns
    if 'transcription' in df.columns and 'medical_specialty' in df.columns:
        df = df[['transcription', 'medical_specialty']].copy()
    elif 'description' in df.columns and 'medical_specialty' in df.columns:
        df = df[['description', 'medical_specialty']].copy()
        df.rename(columns={'description': 'transcription'}, inplace=True)
    else:
        raise ValueError("Dataset format not recognized. Expected 'transcription' and 'medical_specialty' columns")
    
    # Remove null values
    df = df.dropna()
    print(f"   After removing nulls: {len(df)} samples")
    
    # Clean text
    df['transcription'] = df['transcription'].str.strip()
    df['medical_specialty'] = df['medical_specialty'].str.strip()
    
    # Remove empty strings
    df = df[df['transcription'].str.len() > 50]
    print(f"   After removing short texts: {len(df)} samples")
    
    # Filter specialties with enough samples
    specialty_counts = df['medical_specialty'].value_counts()
    valid_specialties = specialty_counts[specialty_counts >= MIN_SAMPLES_PER_CLASS].index
    df = df[df['medical_specialty'].isin(valid_specialties)]
    
    print(f"   Specialties with >={MIN_SAMPLES_PER_CLASS} samples: {len(valid_specialties)}")
    print(f"   Final dataset size: {len(df)} samples")
    
    return df

def encode_labels(df):
    """Encode medical specialty labels to integers."""
    print("\n🏷️  Encoding labels...")
    
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['medical_specialty'])
    
    # Create label mapping
    label_map = {
        'label_to_specialty': {int(i): specialty for i, specialty in enumerate(label_encoder.classes_)},
        'specialty_to_label': {specialty: int(i) for i, specialty in enumerate(label_encoder.classes_)},
        'num_classes': len(label_encoder.classes_)
    }
    
    print(f"   Number of classes: {len(label_encoder.classes_)}")
    print(f"   Top 5 specialties: {list(label_encoder.classes_[:5])}")
    
    return df, label_map

def split_dataset(df):
    """Split dataset into train/validation/test sets."""
    print("\n✂️  Splitting dataset...")
    
    # Stratified split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=RANDOM_SEED, 
        stratify=df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=RANDOM_SEED, 
        stratify=temp_df['label']
    )
    
    print(f"   Train samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df, label_map):
    """Save processed datasets to CSV files."""
    print("\n💾 Saving processed datasets...")
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    
    # Save label mapping
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"   ✅ Train set: {TRAIN_PATH}")
    print(f"   ✅ Validation set: {VAL_PATH}")
    print(f"   ✅ Test set: {TEST_PATH}")
    print(f"   ✅ Label map: {LABEL_MAP_PATH}")

def display_statistics(train_df, val_df, test_df, label_map):
    """Display dataset statistics."""
    print("\n📊 DATASET STATISTICS")
    print("=" * 60)
    
    total_samples = len(train_df) + len(val_df) + len(test_df)
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {label_map['num_classes']}")
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_df)} ({len(train_df)/total_samples*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/total_samples*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/total_samples*100:.1f}%)")
    
    print(f"\nClass distribution in training set:")
    specialty_counts = train_df['medical_specialty'].value_counts().head(10)
    for specialty, count in specialty_counts.items():
        print(f"  {specialty}: {count} samples")
    
    print(f"\nText length statistics (characters):")
    lengths = train_df['transcription'].str.len()
    print(f"  Mean: {lengths.mean():.0f}")
    print(f"  Median: {lengths.median():.0f}")
    print(f"  Max: {lengths.max():.0f}")
    print(f"  Min: {lengths.min():.0f}")
    
    print("\n✅ Dataset preparation complete!")
    print(f"📁 Processed data location: {PROCESSED_DIR.absolute()}")

def generate_synthetic_fallback():
    """Generate small synthetic dataset as fallback if Kaggle dataset unavailable."""
    print("\n🔄 Generating synthetic dataset as fallback...")
    
    specialties = [
        "Cardiology", "Neurology", "Orthopedics", "Dermatology", 
        "Gastroenterology", "Pulmonology", "Nephrology", "Endocrinology"
    ]
    
    templates = [
        "Patient presents with symptoms of {symptom}. Physical examination reveals {finding}. "
        "Diagnosis: {diagnosis}. Treatment plan includes {treatment}.",
        
        "Chief complaint: {symptom}. History of present illness indicates {finding}. "
        "Assessment shows {diagnosis}. Recommend {treatment}.",
        
        "{symptom} noted during consultation. Clinical evaluation shows {finding}. "
        "Provisional diagnosis: {diagnosis}. Advised {treatment}.",
    ]
    
    specialty_data = {
        "Cardiology": {
            "symptoms": ["chest pain", "palpitations", "shortness of breath"],
            "findings": ["irregular heartbeat", "elevated blood pressure", "abnormal ECG"],
            "diagnoses": ["atrial fibrillation", "hypertension", "coronary artery disease"],
            "treatments": ["beta blockers", "lifestyle modification", "cardiac monitoring"]
        },
        "Neurology": {
            "symptoms": ["headache", "dizziness", "numbness"],
            "findings": ["abnormal reflexes", "cranial nerve deficit", "altered sensation"],
            "diagnoses": ["migraine", "peripheral neuropathy", "stroke risk"],
            "treatments": ["pain management", "physical therapy", "neurological follow-up"]
        },
        # Add more specialties as needed
    }
    
    data = []
    np.random.seed(RANDOM_SEED)
    
    for specialty in specialties[:2]:  # Generate for 2 specialties
        specialty_info = specialty_data.get(specialty, specialty_data["Cardiology"])
        
        for _ in range(50):  # 50 samples per specialty
            template = np.random.choice(templates)
            text = template.format(
                symptom=np.random.choice(specialty_info["symptoms"]),
                finding=np.random.choice(specialty_info["findings"]),
                diagnosis=np.random.choice(specialty_info["diagnoses"]),
                treatment=np.random.choice(specialty_info["treatments"])
            )
            data.append({"transcription": text, "medical_specialty": specialty})
    
    df = pd.DataFrame(data)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"✅ Generated {len(df)} synthetic samples")
    print(f"   Saved to: {RAW_DATA_PATH}")
    
    return True

def main():
    """Main execution pipeline."""
    print("=" * 60)
    print("🏥 MEDICAL TRANSCRIPTION DATASET PREPARATION")
    print("=" * 60)
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists, else generate synthetic
    if not download_dataset():
        response = input("\n Generate synthetic dataset for demo? (y/n): ").lower()
        if response == 'y':
            generate_synthetic_fallback()
        else:
            print("\n❌ Cannot proceed without dataset. Exiting...")
            return
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Encode labels
    df, label_map = encode_labels(df)
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(df)
    
    # Save processed data
    save_processed_data(train_df, val_df, test_df, label_map)
    
    # Display statistics
    display_statistics(train_df, val_df, test_df, label_map)
    
    print("\n🚀 Ready for BERT fine-tuning!")
    print("   Next step: Run train_bert.py")

if __name__ == "__main__":
    main()
