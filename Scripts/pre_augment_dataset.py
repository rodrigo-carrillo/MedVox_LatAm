#!/usr/bin/env python3
"""
Pre-Augment Dataset Script

This script creates augmented versions of your audio files BEFORE training.
This is simpler and more reliable than real-time augmentation.

Usage:
    python pre_augment_dataset.py

Make sure to install dependencies first:
    pip install audiomentations soundfile librosa --break-system-packages
"""

import os
import pandas as pd
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import soundfile as sf
from tqdm import tqdm

##########################
### CONFIGURATION ###
##########################

# Input paths
ORIGINAL_AUDIO_DIR = "/scratch/rmcarri/FT_Whisper/Audios"
ORIGINAL_CSV_PATH = "/scratch/rmcarri/FT_Whisper/labels_latin.csv"

# Output paths
AUGMENTED_AUDIO_DIR = "/scratch/rmcarri/FT_Whisper/Audios_Augmented"
AUGMENTED_CSV_PATH = "/scratch/rmcarri/FT_Whisper/labels_augmented.csv"

# Augmentation settings
NUM_AUGMENTATIONS_PER_FILE = 2  # Create 2 augmented versions of each original file
INCLUDE_ORIGINAL = True  # Include original files in augmented dataset

##########################

print("="*70)
print("DATASET PRE-AUGMENTATION SCRIPT")
print("="*70)
print(f"\nInput audio directory: {ORIGINAL_AUDIO_DIR}")
print(f"Input CSV: {ORIGINAL_CSV_PATH}")
print(f"Output audio directory: {AUGMENTED_AUDIO_DIR}")
print(f"Output CSV: {AUGMENTED_CSV_PATH}")
print(f"Augmentations per file: {NUM_AUGMENTATIONS_PER_FILE}")
print(f"Include originals: {INCLUDE_ORIGINAL}\n")

# Create output directory
os.makedirs(AUGMENTED_AUDIO_DIR, exist_ok=True)
print(f"✓ Created output directory: {AUGMENTED_AUDIO_DIR}\n")

# Load original CSV
print("Loading original dataset...")
df = pd.read_csv(ORIGINAL_CSV_PATH, encoding='utf-8')  # FIXED: Added encoding
print(f"✓ Loaded {len(df)} samples from CSV\n")

# Create augmentation pipeline
print("Creating augmentation pipeline...")
augmentation_pipeline = Compose([
    # Add background noise
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6),
    
    # Time stretching (90%-110% speed)
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),
    
    # Pitch shifting (±2 semitones)
    PitchShift(min_semitones=-2, max_semitones=2, p=0.4),
])
print("✓ Augmentation pipeline created\n")

# Process each audio file
new_rows = []
failed_files = []

print("Starting augmentation process...")
print("This may take a while depending on dataset size...\n")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting files"):
    audio_filename = row['audio_filename']
    audio_path = os.path.join(ORIGINAL_AUDIO_DIR, audio_filename)
    
    # Check if file exists
    if not os.path.exists(audio_path):
        failed_files.append(audio_filename)
        continue
    
    try:
        # Load original audio
        audio, sample_rate = sf.read(audio_path)
        
        # Ensure audio is float32 and 1D
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono
        
        # Copy original file to augmented directory (optional)
        if INCLUDE_ORIGINAL:
            original_dest = os.path.join(AUGMENTED_AUDIO_DIR, audio_filename)
            sf.write(original_dest, audio, sample_rate)
        
        # Create augmented versions
        for aug_idx in range(NUM_AUGMENTATIONS_PER_FILE):
            # Apply augmentation
            augmented_audio = augmentation_pipeline(samples=audio, sample_rate=sample_rate)
            
            # Generate new filename
            base_name = os.path.splitext(audio_filename)[0]
            ext = os.path.splitext(audio_filename)[1]
            aug_filename = f"{base_name}_aug{aug_idx+1}{ext}"
            aug_path = os.path.join(AUGMENTED_AUDIO_DIR, aug_filename)
            
            # Save augmented audio
            sf.write(aug_path, augmented_audio, sample_rate)
            
            # Add to new dataset
            new_rows.append({
                'audio_filename': aug_filename,
                'text_original': row['text_original']
            })
    
    except Exception as e:
        print(f"\n⚠ Error processing {audio_filename}: {e}")
        failed_files.append(audio_filename)
        continue

print("\n" + "="*70)
print("AUGMENTATION COMPLETE")
print("="*70)

# Create final dataset
if INCLUDE_ORIGINAL:
    # Include both original and augmented
    df_final = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    print(f"✓ Original samples: {len(df)}")
    print(f"✓ Augmented samples: {len(new_rows)}")
    print(f"✓ Total samples: {len(df_final)}")
else:
    # Only augmented samples
    df_final = pd.DataFrame(new_rows)
    print(f"✓ Augmented samples only: {len(df_final)}")

# Save augmented dataset CSV
df_final.to_csv(AUGMENTED_CSV_PATH, index=False, encoding='utf-8')  # FIXED: Added encoding
print(f"\n✓ Saved augmented CSV to: {AUGMENTED_CSV_PATH}")

# Report failures
if failed_files:
    print(f"\n⚠ Failed to process {len(failed_files)} files:")
    for f in failed_files[:10]:
        print(f"  - {f}")
    if len(failed_files) > 10:
        print(f"  ... and {len(failed_files) - 10} more")

# Summary statistics
print("\n" + "="*70)
print("DATASET STATISTICS")
print("="*70)
print(f"Original dataset size: {len(df)} samples")
print(f"Augmented dataset size: {len(df_final)} samples")
print(f"Dataset increase: {(len(df_final)/len(df) - 1)*100:.1f}%")
print(f"Augmented files location: {AUGMENTED_AUDIO_DIR}")
print(f"Augmented CSV location: {AUGMENTED_CSV_PATH}")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Verify a few augmented audio files sound reasonable")
print("2. Update your Run_FT.py with these paths:")
print(f'   AUDIO_DIR = "{AUGMENTED_AUDIO_DIR}"')
print(f'   CSV_PATH = "{AUGMENTED_CSV_PATH}"')
print("3. Run your fine-tuning script")
print("="*70)
