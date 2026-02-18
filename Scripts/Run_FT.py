########################
### IMPORT LIBRARIES ###
########################

import os
import re
import pandas as pd
import random
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Hugging Face libraries
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,  # NEW: Added for early stopping
)

# Evaluation metrics
import jiwer

print("All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

#########################










#################################
### DATA PATHS - MODIFY THESE ###
#################################

AUDIO_DIR = "/scratch/rmcarri/FT_Whisper/Audios_Augmented"              # Directory containing your .wav audio files
CSV_PATH = "/scratch/rmcarri/FT_Whisper/labels_latin_augmented.csv"     # Path to your CSV file with transcriptions
OUTPUT_DIR = "/scratch/rmcarri/FT_Whisper/whisper-large-v3-finetuned"   # Directory where you want to save the fine-tuned model
CACHE_DIR = "/scratch/rmcarri/FT_Whisper/hf_cache"                      # Directory where Hugging Face models will be cached/downloaded

os.makedirs(OUTPUT_DIR, exist_ok = True) # Create necessary directories
os.makedirs(CACHE_DIR, exist_ok = True)

print("Configuration set!")
print(f"Audio directory: {AUDIO_DIR}")
print(f"CSV path: {CSV_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Cache directory: {CACHE_DIR}")

###########################










###########################
### MODEL CONFIGURATION ###
###########################

MODEL_ID = "openai/whisper-large-v3"
LANGUAGE = "spanish"   # Change if your data is in another language
TASK = "transcribe"    # Use "translate" if you want to translate to English

###########################










###########################
### AUDIO CONFIGURATION ###
###########################

SAMPLING_RATE = 16000  # Whisper requires 16kHz audio

###########################










##################
### DATA SPLIT ###
##################

TEST_SIZE = 0.2  # 20% of data for validation
RANDOM_SEED = 42  # For reproducibility

##################










#####################
### LOAD CSV FILE ###
#####################

print("Loading CSV file...")
df = pd.read_csv(CSV_PATH)

print(f"\nDataset shape: {df.shape}") # Display basic information
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

required_columns = ['audio_filename', 'text_original'] # Check for required columns
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV!")
print(f"\n✓ All required columns found")

#####################










####################################
### CREATE FULL AUDIO FILE PATHS ###
####################################

df['audio_path'] = df['audio_filename'].apply(
    lambda x: os.path.join(AUDIO_DIR, x)
)

print("Checking if audio files exist...") # Verify that audio files exist
missing_files = []
for idx, row in df.iterrows():
    if not os.path.exists(row['audio_path']):
        missing_files.append(row['audio_filename'])

if missing_files:
    print(f"\n⚠ Warning: {len(missing_files)} audio files not found:")
    for f in missing_files[:5]:  # Show first 5
        print(f"  - {f}")
    if len(missing_files) > 5:
        print(f"  ... and {len(missing_files) - 5} more")
    
    # Remove rows with missing files
    df = df[df['audio_path'].apply(os.path.exists)]
    print(f"\nRemoved {len(missing_files)} rows. Remaining: {len(df)}")
else:
    print(f"✓ All {len(df)} audio files found!")

df = df[['audio_path', 'text_original']] # Keep only the columns we need
df = df.rename(columns={'text_original': 'text'})

print(f"\nFinal dataset shape: {df.shape}")
print(df.head())

####################################










########################################################
### CONVERT PANDAS DATAFRAME TO HUGGING FACE DATASET ###
########################################################

print("Converting to Hugging Face Dataset format...")

# Create a dictionary format that Dataset expects
dataset_dict = {
    'audio': df['audio_path'].tolist(),
    'text': df['text'].tolist(),
}

# Create the Dataset
dataset = Dataset.from_dict(dataset_dict)

# Cast the audio column to Audio type with the correct sampling rate
dataset = dataset.cast_column("audio", Audio(sampling_rate = SAMPLING_RATE)) # This tells the dataset to load audio files at 16kHz

print(f"✓ Dataset created with {len(dataset)} examples")
print(f"\nDataset features: {dataset.features}")

########################################################










############################################
### SPLIT INTO TRAIN AND VALIDATION SETS ###
###############################################

print(f"\nSplitting dataset (test_size={TEST_SIZE})...") # Split into train and validation sets

dataset_splits = dataset.train_test_split(
    test_size=TEST_SIZE,
    seed=RANDOM_SEED,
    shuffle=True  # Shuffle before splitting
)

train_dataset = dataset_splits['train']
eval_dataset = dataset_splits['test']

print(f"✓ Training examples: {len(train_dataset)}")
print(f"✓ Validation examples: {len(eval_dataset)}")

# Show a sample
print(f"\nSample from training set:")
sample = train_dataset[0]
print(f"Text: {sample['text']}")
print(f"Audio shape: {sample['audio']['array'].shape}")
print(f"Sampling rate: {sample['audio']['sampling_rate']} Hz")

###############################################










##############################
### LOAD WHISPER PROCESSOR ###
##############################

print(f"Loading Whisper processor from {MODEL_ID}...")
print(f"Models will be cached in: {CACHE_DIR}\n")

# Load the processor
processor = WhisperProcessor.from_pretrained(
    MODEL_ID,
    cache_dir = CACHE_DIR,
    language = LANGUAGE,
    task = TASK
) # The processor combines the feature extractor (for audio) and tokenizer (for text)

# NEW: Fix extra_special_tokens to prevent tokenizer config errors
if hasattr(processor.tokenizer, 'extra_special_tokens'):
    if isinstance(processor.tokenizer.extra_special_tokens, list):
        processor.tokenizer.extra_special_tokens = {}

print("✓ Processor loaded successfully")
print(f"Language: {processor.tokenizer.language}")
print(f"Task: {processor.tokenizer.task}")

##############################










##########################
### LOAD WHISPER MODEL ###
##########################

print(f"\nLoading Whisper model from {MODEL_ID}...")
print("This may take a few minutes...\n")

# Load the model for conditional generation (sequence-to-sequence)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    cache_dir = CACHE_DIR,
    torch_dtype = torch.float32  # Explicitly use FP32
)
model = model.float()  # Ensure model is in float32

# Configure the model for fine-tuning
model.generation_config.forced_decoder_ids = None  # Allow model to predict freely
model.generation_config.suppress_tokens = []  # Don't suppress any tokens
model.config.use_cache = False  # Disable caching during training

# NEW: Enable gradient checkpointing to save GPU memory
model.gradient_checkpointing_enable()

print("✓ Model loaded successfully")
print(f"Model parameters: {model.num_parameters():,}")
print(f"Model size: ~{model.num_parameters() * 4 / 1e9:.2f} GB (fp32)")
print("✓ Gradient checkpointing enabled (saves GPU memory)")

##########################










#################################
### PREPARE DATA FOR TRAINING ###
#################################

def prepare_dataset(batch):
    """
    Preprocesses a batch of examples for Whisper training.
    
    Steps:
    1. Extract audio features using the processor's feature extractor
    2. Tokenize the text labels
    
    Args:
        batch: A batch from the dataset containing 'audio' and 'text'
    
    Returns:
        Batch with added 'input_features' and 'labels'
    """

    audio = batch["audio"]     # Get the audio array from the batch
    
    # Compute input features from audio
    # The processor converts audio to log-mel spectrogram features
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Tokenize the text
    # This converts the text transcript to token IDs
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch

print("Preprocessing training dataset...")
train_dataset = train_dataset.map(
    prepare_dataset,
    remove_columns=train_dataset.column_names,  # Remove original columns
    desc="Preparing training data"
)

print("Preprocessing validation dataset...")
eval_dataset = eval_dataset.map(
    prepare_dataset,
    remove_columns=eval_dataset.column_names,
    desc="Preparing validation data"
)

print("\n✓ Preprocessing complete")
print(f"Training set features: {train_dataset.features}")
print(f"Validation set features: {eval_dataset.features}")

#################################










#####################
### DATA COLLATOR ###
#####################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for speech-to-text tasks.
    
    This collator:
    1. Pads input features to the same length
    2. Pads labels to the same length
    3. Replaces padding token IDs in labels with -100 (ignored in loss)
    """
    
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        # Input features are the log-mel spectrograms
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        
        # Pad input features to max length in batch
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # Labels are the tokenized transcripts
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad labels to max length in batch
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # Replace padding token IDs with -100
        # -100 is ignored by PyTorch's CrossEntropyLoss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove decoder start token if present
        # Whisper adds this automatically during generation
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# Create the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print("✓ Data collator created")

#####################










##########################
### EVALUATION METRICS ###
##########################

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces()
])


def compute_metrics(pred):
    """
    Compute Word Error Rate (WER) for evaluation.
    
    Args:
        pred: Predictions from the model containing:
            - predictions: Model output logits
            - label_ids: Ground truth labels
    
    Returns:
        Dictionary with WER score
    """
    # Get predicted token IDs (take argmax over vocabulary dimension)
    pred_ids = pred.predictions
    
    # Get ground truth labels
    label_ids = pred.label_ids

    # Replace -100 (padding) with pad token ID for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels to text
    pred_str = processor.tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )
    label_str = processor.tokenizer.batch_decode(
        label_ids,
        skip_special_tokens=True
    )

    # Normalize text before computing WER
    pred_str_norm = [transform(s) for s in pred_str]
    label_str_norm = [transform(s) for s in label_str]

    # Compute WER & CER
    wer = jiwer.wer(label_str_norm, pred_str_norm)
    cer = jiwer.cer(label_str_norm, pred_str_norm)

    return {"wer": wer,
            "cer": cer}

print("✓ Evaluation functions defined")
print("\nExample normalization:")
print(f"Original: 'Hello, World! How are you?'")
print(f"Normalized: '{transform('Hello, World! How are you?')}'")

##########################










##############################
### TRAINING CONFIGURATION ###
##############################

# UPDATED: Improved training configuration to prevent overfitting
training_args = Seq2SeqTrainingArguments(
    # Output and logging
    output_dir = OUTPUT_DIR,           # Where to save checkpoints
    per_device_train_batch_size = 2,   # Batch size per GPU - reduce if OOM
    per_device_eval_batch_size = 2,    # Evaluation batch size
    
    # Training duration - REDUCED from 50 to prevent overfitting
    num_train_epochs = 50,             # Reduced from 50 - will stop earlier with early stopping
    
    # Optimization
    gradient_accumulation_steps = 2,   # Accumulate gradients over 2 steps
    gradient_checkpointing = True,     # NEW: Save GPU memory
    learning_rate = 1e-5,              # Learning rate - Whisper works well with small LR
    warmup_steps = 200,                # INCREASED: Linear warmup steps (was 20)
    weight_decay = 0.01,               # NEW: L2 regularization to prevent overfitting
    max_grad_norm = 1.0,               # NEW: Gradient clipping
    
    # Mixed precision training (much faster on modern GPUs)
    fp16 = False,                      # Completely disabled to avoid type mismatches
    fp16_full_eval = False,            # Use FP32 for evaluation to avoid type errors
    
    # UPDATED: Evaluation and saving - more frequent to catch best model early
    eval_strategy = "epoch",           # Evaluate every N steps (not epochs)
    eval_steps = 100,                  # Evaluate every 50 steps (was 10, too frequent)
    save_strategy = "epoch",           # Save checkpoint every N steps
    save_steps = 50,                   # Save every 50 steps (was 10)
    save_total_limit = 3,              # INCREASED: Keep 3 best checkpoints (was 2)
    load_best_model_at_end = True,     # Load best model at end of training
    metric_for_best_model = "wer",     # Use WER to determine best model
    greater_is_better = False,         # Lower WER is better
    
    # Logging
    logging_dir = f"{OUTPUT_DIR}/logs",
    logging_steps = 25,                # INCREASED: Log every 25 steps (was 10, less verbose)
    logging_first_step = True,
    
    # Generation settings (for evaluation)
    predict_with_generate = True,      # Use generation for evaluation
    generation_max_length = 250,       # Max generation length
    
    # Performance optimizations
    dataloader_num_workers = 0,        # INCREASED: More workers for faster data loading (was 0)
    dataloader_pin_memory = True,      # CHANGED: Enable pin_memory for faster GPU transfer (was False)
    remove_unused_columns = False,     # Keep all columns
    
    # Reproducibility
    seed = RANDOM_SEED,
    
    # NEW: Report to none to avoid wandb/tensorboard issues
    report_to = "none",
)

print("✓ Training configuration set")
print(f"\nTraining settings:")
print(f"  - Epochs: {training_args.num_train_epochs}")
print(f"  - Batch size: {training_args.per_device_train_batch_size}")
print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  - Learning rate: {training_args.learning_rate}")
print(f"  - Weight decay: {training_args.weight_decay}")
print(f"  - Warmup steps: {training_args.warmup_steps}")
print(f"  - Gradient checkpointing: {training_args.gradient_checkpointing}")
print(f"  - Max grad norm: {training_args.max_grad_norm}")
print(f"  - Eval every: {training_args.eval_steps} steps")
print(f"  - Early stopping patience: 5 evaluations")
print(f"  - FP16: {training_args.fp16}")
print(f"  - Output dir: {training_args.output_dir}")

##############################










##########################
### INITIALIZE TRAINER ###
##########################

print("\nInitializing trainer with early stopping callback...")

# NEW: Create early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5,   # Stop if no improvement for 5 consecutive evaluations
    early_stopping_threshold=0.0  # Any improvement counts
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
    callbacks = [early_stopping_callback],  # NEW: Add early stopping
)

print("✓ Trainer initialized with early stopping")
print(f"  - Will stop if WER doesn't improve for {early_stopping_callback.early_stopping_patience} evaluations")
print(f"\nEstimated training steps: {len(train_dataset) // training_args.per_device_train_batch_size // training_args.gradient_accumulation_steps * training_args.num_train_epochs}")

##########################










###########################
### EVALUATE BASE MODEL ###
###########################

print("\nEvaluating base model performance...")
print("This may take a few minutes...\n")

base_model_metrics = trainer.evaluate()

print("\n" + "="*50)
print("BASE MODEL PERFORMANCE")
print("="*50)
print(f"WER: {base_model_metrics['eval_wer']:.4f} ({base_model_metrics['eval_wer']*100:.2f}%)")
print(f"CER: {base_model_metrics['eval_cer']:.4f} ({base_model_metrics['eval_cer']*100:.2f}%)")
print(f"Loss: {base_model_metrics['eval_loss']:.4f}")
print("="*50)

###########################










###################
### TRAIN MODEL ###
###################

print("\nStarting training with early stopping...")
print("Monitor the progress below.\n")
print("IMPORTANT NOTES:")
print("  - Training will stop automatically if validation WER doesn't improve for 5 evaluations")
print("  - The best model (lowest WER) will be automatically loaded at the end")
print("  - Check logs to see when the best checkpoint was saved\n")
print(f"TIP: You can view detailed logs in TensorBoard:")
print(f"  tensorboard --logdir {OUTPUT_DIR}/logs\n")

# Start training!
train_result = trainer.train()

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
if trainer.state.best_metric is not None:
    print(f"Best WER achieved: {trainer.state.best_metric:.4f}")
    print(f"Best model from step: {trainer.state.best_model_checkpoint}")

###################










############################
### EVALUATE FINAL MODEL ###
############################

print("\nEvaluating fine-tuned model...\n")

final_metrics = trainer.evaluate()

print("\n" + "="*50)
print("RESULTS COMPARISON")
print("="*50)
print(f"\nBase Model:")
print(f"  WER: {base_model_metrics['eval_wer']:.4f} ({base_model_metrics['eval_wer']*100:.2f}%)")
print(f"  CER: {base_model_metrics['eval_cer']:.4f} ({base_model_metrics['eval_cer']*100:.2f}%)")
print(f"  Loss: {base_model_metrics['eval_loss']:.4f}")

print(f"\nFine-tuned Model:")
print(f"  WER: {final_metrics['eval_wer']:.4f} ({final_metrics['eval_wer']*100:.2f}%)")
print(f"  CER: {final_metrics['eval_cer']:.4f} ({final_metrics['eval_cer']*100:.2f}%)")
print(f"  Loss: {final_metrics['eval_loss']:.4f}")

# Calculate improvement
wer_improvement = (base_model_metrics['eval_wer'] - final_metrics['eval_wer']) / base_model_metrics['eval_wer'] * 100
cer_improvement = (base_model_metrics['eval_cer'] - final_metrics['eval_cer']) / base_model_metrics['eval_cer'] * 100
print(f"\nImprovement:")
print(f"  WER reduced by: {wer_improvement:.1f}%")
print(f"  CER reduced by: {cer_improvement:.1f}%")
print("="*50)

############################










############################
### SAVE THE FINAL MODEL ###
############################

final_model_dir = f"{OUTPUT_DIR}/final_model" # Save the final model

print(f"\nSaving model & processor to {final_model_dir}...")
trainer.save_model(final_model_dir) # Save model
processor.save_pretrained(final_model_dir) # Save processor (needed for inference)

# NEW: Fix tokenizer_config.json to prevent loading errors
import json
tokenizer_config_path = os.path.join(final_model_dir, "tokenizer_config.json")
if os.path.exists(tokenizer_config_path):
    with open(tokenizer_config_path, 'r') as f:
        config = json.load(f)
    
    if "extra_special_tokens" in config and isinstance(config["extra_special_tokens"], list):
        config["extra_special_tokens"] = {}
        with open(tokenizer_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("✓ Fixed tokenizer_config.json format")

print("\n✓ Model saved successfully!")
print(f"\nTo load this model later:")
print(f"  from transformers import WhisperForConditionalGeneration, WhisperProcessor")
print(f"  model = WhisperForConditionalGeneration.from_pretrained('{final_model_dir}')")
print(f"  processor = WhisperProcessor.from_pretrained('{final_model_dir}')")

############################










########################################
### TEST INFERENCE IN A FEW EXAMPLES ###
########################################

random.seed(42) # Set random seed for reproducibility

# Select a few random examples from validation set
num_examples = 3
sample_indices = random.sample(range(len(dataset_splits['test'])), num_examples)

print("\nTesting model on random validation examples...\n")
print("="*70)

# Put model in evaluation mode
model.eval()

for i, idx in enumerate(sample_indices, 1):
    # Get the example
    example = dataset_splits['test'][idx]

    # Ground truth text
    ground_truth = example['text']

    # Prepare input
    input_features = processor(
        example['audio']['array'],
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt"
    ).input_features

    # Move to GPU if available
    if torch.cuda.is_available():
        input_features = input_features.cuda()

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # Decode prediction
    prediction = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    # Calculate WER for this example
    example_wer = jiwer.wer(
        [transform(ground_truth)],
        [transform(prediction)]
    )
    example_cer = jiwer.cer(
        [transform(ground_truth)],
        [transform(prediction)]
    )



    # Print results
    print(f"Example {i}:")
    print(f"Ground Truth: {ground_truth}")
    print(f"Prediction:   {prediction}")
    print(f"WER: {example_wer:.4f} ({example_wer*100:.2f}%)")
    print(f"CER: {example_cer:.4f} ({example_cer*100:.2f}%)")
    print("-" * 70)

print("\n✓ Testing complete!")

########################################










###############
### SUMMARY ###
###############

print("\n" + "="*70)
print("FINE-TUNING SUMMARY")
print("="*70)
print(f"\nModel: {MODEL_ID}")
print(f"Language: {LANGUAGE}")
print(f"Task: {TASK}")

print(f"\nDataset:")
print(f"  Training examples: {len(train_dataset)}")
print(f"  Validation examples: {len(eval_dataset)}")

print(f"\nTraining:")
print(f"  Max epochs: {training_args.num_train_epochs}")
print(f"  Actual epochs trained: {train_result.metrics.get('epoch', 'N/A')}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Weight decay: {training_args.weight_decay}")
print(f"  Early stopping: Enabled (patience=5)")

print(f"\nPerformance:")
print(f"  Base model WER: {base_model_metrics['eval_wer']*100:.2f}%")
print(f"  Fine-tuned WER: {final_metrics['eval_wer']*100:.2f}%")
print(f"  Improvement: {wer_improvement:.1f}%")
print(f"  Base model CER: {base_model_metrics['eval_cer']*100:.2f}%")
print(f"  Fine-tuned CER: {final_metrics['eval_cer']*100:.2f}%")
print(f"  Improvement: {cer_improvement:.1f}%")

print(f"\nSaved model location:")
print(f"  {final_model_dir}")
print("\n" + "="*70)

###############


