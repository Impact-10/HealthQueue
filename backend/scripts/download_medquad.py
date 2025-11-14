"""
Download and prepare MedQuAD dataset (47,457 medical Q&A samples)
Source: https://github.com/abachaa/MedQuAD

This dataset contains medical questions and answers from NIH sources
covering 37 different medical specialties.
"""

import os
import json
import pandas as pd
import requests
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "medquad_raw"
PROCESSED_DIR = DATA_DIR / "processed"

# MedQuAD GitHub repository
MEDQUAD_REPO = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"

def download_medquad():
    """Download MedQuAD dataset from GitHub."""
    print("\n" + "=" * 60)
    print("📥 DOWNLOADING MEDQUAD DATASET")
    print("=" * 60)
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = RAW_DIR / "medquad.zip"
    
    if zip_path.exists():
        print(f"✅ Dataset already downloaded: {zip_path}")
        # Non-interactive mode: skip re-download and proceed
        # Set MEDQUAD_FORCE_DOWNLOAD=y to force re-download if needed.
        if os.environ.get("MEDQUAD_FORCE_DOWNLOAD", "n").lower() != "y":
            return True
    
    print(f"📥 Downloading from GitHub...")
    print(f"   URL: {MEDQUAD_REPO}")
    
    try:
        response = requests.get(MEDQUAD_REPO, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {zip_path}")
        
        # Extract
        print("📦 Extracting...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        print("✅ Extraction complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        print("\nAlternative: Manual download")
        print(f"1. Visit: https://github.com/abachaa/MedQuAD")
        print(f"2. Click 'Code' → 'Download ZIP'")
        print(f"3. Extract to: {RAW_DIR}")
        return False

def parse_xml_files():
    """Parse XML files from MedQuAD dataset."""
    print("\n" + "=" * 60)
    print("🔍 PARSING XML FILES")
    print("=" * 60)
    
    # Find the extracted folder
    medquad_folders = list(RAW_DIR.glob("MedQuAD-*"))
    if not medquad_folders:
        print("❌ MedQuAD folder not found!")
        return None
    
    medquad_dir = medquad_folders[0]
    print(f"📂 Found: {medquad_dir}")
    
    # Parse all XML files
    data = []
    
    # MedQuAD has subdirectories for each source
    xml_files = list(medquad_dir.rglob("*.xml"))
    
    print(f"📄 Found {len(xml_files)} XML files")
    print("🔄 Parsing...")
    
    for xml_file in tqdm(xml_files, desc="Parsing XML"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract focus/specialty from path or filename
            focus = xml_file.parent.name
            
            for qa_pair in root.findall('.//QAPair'):
                question_elem = qa_pair.find('Question')
                answer_elem = qa_pair.find('Answer')
                
                if question_elem is not None and answer_elem is not None:
                    question = question_elem.text
                    answer = answer_elem.text
                    
                    if question and answer:
                        # Combine Q&A as text
                        text = f"Question: {question.strip()}\\n\\nAnswer: {answer.strip()}"
                        
                        data.append({
                            'text': text,
                            'question': question.strip(),
                            'answer': answer.strip(),
                            'specialty': focus,
                            'source': xml_file.parent.parent.name
                        })
        except Exception as e:
            print(f"⚠️  Error parsing {xml_file}: {e}")
            continue
    
    print(f"✅ Parsed {len(data)} Q&A pairs")
    
    return pd.DataFrame(data)

def clean_and_prepare(df):
    """Clean and prepare the dataset."""
    print("\n" + "=" * 60)
    print("🧹 CLEANING DATA")
    print("=" * 60)
    
    print(f"📊 Initial samples: {len(df)}")
    
    # Remove nulls
    df = df.dropna(subset=['text', 'specialty'])
    print(f"   After removing nulls: {len(df)}")
    
    # Remove very short texts (less than 50 characters)
    df = df[df['text'].str.len() >= 50]
    print(f"   After removing short texts: {len(df)}")
    
    # Map specialties to broader categories
    specialty_mapping = {
        'CDCQnA': 'General Medicine',
        'GHR': 'Genetics',
        'GARD': 'Rare Diseases',
        'MPlusDrugs': 'Pharmacology',
        'MPlusHealthTopics': 'General Medicine',
        'MPlusHerbsSupplements': 'Alternative Medicine',
        'NIDDK': 'Diabetes & Digestive',
        'NIHSeniorHealth': 'Geriatrics',
        'CancerGov': 'Oncology',
        'Genetics_and_Birth_Defects': 'Genetics',
        'Blood_Diseases': 'Hematology',
        'Heart_Disease': 'Cardiology',
        'Mental_Health': 'Psychiatry',
        'Nutrition': 'Nutrition',
        'Kidney_Disease': 'Nephrology',
        'Lung_Diseases': 'Pulmonology',
    }
    
    # Apply mapping
    df['specialty_clean'] = df['specialty'].replace(specialty_mapping)
    
    # Fill unmapped specialties
    df['specialty_clean'] = df['specialty_clean'].fillna(df['specialty'])
    
    # Filter specialties with at least 100 samples
    specialty_counts = df['specialty_clean'].value_counts()
    valid_specialties = specialty_counts[specialty_counts >= 100].index
    df = df[df['specialty_clean'].isin(valid_specialties)]
    
    print(f"   Specialties with >=100 samples: {len(valid_specialties)}")
    print(f"   Final dataset size: {len(df)}")
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['specialty_clean'])
    
    print(f"\n🏷️  Classes: {len(le.classes_)}")
    print(f"   Top 10 specialties:")
    for spec in specialty_counts.head(10).index[:10]:
        count = specialty_counts[spec]
        if spec in valid_specialties:
            print(f"      {spec}: {count} samples")
    
    return df, le

def split_and_save(df, label_encoder):
    """Split into train/val/test and save."""
    print("\n" + "=" * 60)
    print("✂️  SPLITTING DATASET")
    print("=" * 60)
    
    from sklearn.model_selection import train_test_split
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use 'text' as 'transcription' for compatibility with existing code
    train_df[['text', 'label']].rename(columns={'text': 'transcription'}).to_csv(
        PROCESSED_DIR / "train.csv", index=False
    )
    val_df[['text', 'label']].rename(columns={'text': 'transcription'}).to_csv(
        PROCESSED_DIR / "val.csv", index=False
    )
    test_df[['text', 'label']].rename(columns={'text': 'transcription'}).to_csv(
        PROCESSED_DIR / "test.csv", index=False
    )
    
    # Save label mapping
    label_map = {
        'num_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist(),
        'encoding': {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
    }
    
    with open(PROCESSED_DIR / "label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\n💾 Files saved:")
    print(f"   ✅ {PROCESSED_DIR / 'train.csv'}")
    print(f"   ✅ {PROCESSED_DIR / 'val.csv'}")
    print(f"   ✅ {PROCESSED_DIR / 'test.csv'}")
    print(f"   ✅ {PROCESSED_DIR / 'label_map.json'}")
    
    return train_df, val_df, test_df

def print_statistics(train_df, val_df, test_df):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("📊 DATASET STATISTICS")
    print("=" * 60)
    
    total = len(train_df) + len(val_df) + len(test_df)
    
    print(f"Total samples: {total}")
    print(f"Number of classes: {train_df['label'].nunique()}")
    print(f"\nSplit distribution:")
    print(f"  Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/total*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/total*100:.1f}%)")
    
    print(f"\nClass distribution in training set:")
    class_dist = train_df['specialty_clean'].value_counts().head(10)
    for specialty, count in class_dist.items():
        print(f"  {specialty}: {count} samples")
    
    print(f"\nText length statistics (characters):")
    text_lengths = train_df['text'].str.len()
    print(f"  Mean:   {text_lengths.mean():.0f}")
    print(f"  Median: {text_lengths.median():.0f}")
    print(f"  Max:    {text_lengths.max():.0f}")
    print(f"  Min:    {text_lengths.min():.0f}")

def main():
    """Main execution."""
    print("\n" + "=" * 60)
    print("🏥 MEDQUAD DATASET PREPARATION")
    print("=" * 60)
    
    # Step 1: Download
    if not download_medquad():
        print("\n❌ Download failed. Please download manually.")
        return
    
    # Step 2: Parse XML files
    df = parse_xml_files()
    if df is None or len(df) == 0:
        print("❌ No data parsed!")
        return
    
    # Step 3: Clean and prepare
    df, label_encoder = clean_and_prepare(df)
    
    # Step 4: Split and save
    train_df, val_df, test_df = split_and_save(df, label_encoder)
    
    # Step 5: Statistics
    print_statistics(train_df, val_df, test_df)
    
    print("\n" + "=" * 60)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"📁 Processed data location: {PROCESSED_DIR}")
    print(f"\n🚀 Ready for BERT fine-tuning!")
    print(f"   Next step: Run train_bert.py with 20 epochs")
    print(f"\n   $ python scripts/train_bert.py")

if __name__ == "__main__":
    main()
