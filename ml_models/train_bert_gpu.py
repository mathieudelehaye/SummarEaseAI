"""
Train BERT model for intent classification with GPU support
"""

import os
import sys
import logging
import json
import shutil
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm

import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'training.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Default hyperparameters
DEFAULT_CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'epochs': 10,
    'max_length': 128,
    'model_name': 'bert-base-uncased',
    'hidden_size': 768,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'intermediate_size': 3072
}

class IntentDataset(Dataset):
    """Custom dataset for intent classification"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTIntentClassifier:
    """BERT-based intent classifier"""
    def __init__(
        self,
        model_path: str,
        num_labels: int = None,
        config: dict = None,
    ):
        self.model_path = Path(model_path)
        self.num_labels = num_labels
        self.config = config or DEFAULT_CONFIG.copy()
        
        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            logger.info("🚀 Using GPU for training")
        else:
            logger.info("💻 Using CPU for training")
        
        # Set up signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._validate_config()
        self._build_model()
        self._log_config()

    def _validate_config(self):
        """Validate hyperparameters"""
        logger.info("\n🔍 Validating configuration...")
        
        # Batch size validation
        if self.config['batch_size'] <= 0:
            raise ValueError("Batch size must be positive")
        if self.config['batch_size'] > 64:
            logger.warning("⚠️ Large batch size may cause memory issues")
        
        # Learning rate validation
        if not (0 < self.config['learning_rate'] < 1):
            raise ValueError("Learning rate should be between 0 and 1")
        if self.config['learning_rate'] > 0.01:
            logger.warning("⚠️ Learning rate seems high")
        
        # Epochs validation
        if self.config['epochs'] <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.config['epochs'] < 3:
            logger.warning("⚠️ Very few epochs might lead to underfitting")
        if self.config['epochs'] > 50:
            logger.warning("⚠️ Many epochs might lead to overfitting")
        
        # Max length validation
        if not (16 <= self.config['max_length'] <= 512):
            raise ValueError("Max length should be between 16 and 512")
        
        logger.info("✅ Configuration validated")

    def _signal_handler(self, signum, frame):
        """Handle interruption signals"""
        logger.info("\n⚠️ Training interrupted! Cleaning up...")
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                logger.info("Saving model state before exit...")
                self.model.save_pretrained(self.model_path)
                logger.info("✅ Model saved successfully")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def _log_config(self):
        """Log training configuration"""
        logger.info("\n📋 Training Configuration:")
        for key, value in self.config.items():
            logger.info(f"   - {key}: {value}")
        logger.info(f"   - Device: {self.device}")

    def _build_model(self):
        """Initialize BERT model and tokenizer"""
        logger.info("\n🔧 Building model...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
            
            # Create model directory if it doesn't exist
            os.makedirs(self.model_path, exist_ok=True)
            
            # Check if we have a valid existing model
            if is_valid_model_dir(self.model_path):
                logger.info("Loading existing model...")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    num_labels=self.num_labels
                )
                logger.info("✅ Model loaded successfully")
            else:
                logger.info("Initializing new model...")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config['model_name'],
                    num_labels=self.num_labels
                )
                logger.info("✅ Model initialized successfully")
            
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"❌ Error building model: {e}")
            raise

    def train(self, X_train, X_test, y_train, y_test):
        """Train the model"""
        logger.info("\n🚀 Starting training...")
        
        try:
            # Create datasets
            train_dataset = IntentDataset(
                X_train, 
                y_train, 
                self.tokenizer, 
                self.config['max_length']
            )
            test_dataset = IntentDataset(
                X_test, 
                y_test, 
                self.tokenizer, 
                self.config['max_length']
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['batch_size']
            )
            
            # Setup training
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.config['learning_rate']
            )
            num_training_steps = len(train_loader) * self.config['epochs']
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps
            )
            
            # Training loop
            best_accuracy = 0.0
            for epoch in range(self.config['epochs']):
                logger.info(f"\n📊 Epoch {epoch + 1}/{self.config['epochs']}")
                
                # Training phase
                self.model.train()
                total_loss = 0
                
                progress_bar = tqdm(train_loader, desc="Training")
                for batch in progress_bar:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    progress_bar.set_postfix({'loss': loss.item()})
                
                avg_train_loss = total_loss / len(train_loader)
                logger.info(f"Average training loss: {avg_train_loss:.4f}")
                
                # Evaluation phase
                self.model.eval()
                total_eval_loss = 0
                correct_predictions = 0
                total_predictions = 0
                
                with torch.no_grad():
                    for batch in tqdm(test_loader, desc="Evaluating"):
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        total_eval_loss += loss.item()
                        
                        predictions = torch.argmax(outputs.logits, dim=1)
                        correct_predictions += (predictions == labels).sum().item()
                        total_predictions += labels.shape[0]
                
                avg_eval_loss = total_eval_loss / len(test_loader)
                accuracy = correct_predictions / total_predictions
                
                logger.info(f"Validation Loss: {avg_eval_loss:.4f}")
                logger.info(f"Accuracy: {accuracy:.4f}")
                
                # Save best model
                if accuracy > best_accuracy:
                    logger.info("🎯 New best model! Saving...")
                    best_accuracy = accuracy
                    self.model.save_pretrained(self.model_path)
                    self.tokenizer.save_pretrained(self.model_path)
            
            logger.info("\n✅ Training completed!")
            logger.info(f"Best accuracy: {best_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error during training: {e}")
            self._cleanup()
            raise

def check_existing_model(model_path: str) -> bool:
    """Check if model exists and ask for confirmation to overwrite"""
    if os.path.exists(model_path):
        logger.warning(f"\n⚠️ Model already exists at {model_path}")
        logger.warning("Please remove the existing model directory before training.")
        logger.warning("This helps prevent accidental model overwrites.")
        return True
    return False

def is_valid_model_dir(model_path: str) -> bool:
    """Check if directory contains a valid model"""
    model_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "model.ckpt.index",
        "flax_model.msgpack"
    ]
    config_file = "config.json"
    
    # Check for config file first
    if not os.path.exists(os.path.join(model_path, config_file)):
        return False
        
    # Check for at least one model file
    return any(os.path.exists(os.path.join(model_path, f)) for f in model_files)

def handle_existing_model(model_path: str) -> bool:
    """Handle existing model directory
    Returns:
        bool: True if training should proceed, False if it should be cancelled
    """
    if not os.path.exists(model_path):
        return True
        
    # Check if it's a valid model
    is_valid = is_valid_model_dir(model_path)
    if not is_valid:
        logger.info("🔄 Starting fresh training...")
        shutil.rmtree(model_path)
        return True
        
    # Valid model exists
    logger.warning(f"\n⚠️ Valid model found at {model_path}")
    logger.info("\nChoose an option:")
    logger.info("1. Delete existing model and train new one")
    logger.info("2. Continue training existing model")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-2): ").strip()
            if choice == "1":
                logger.info("🗑️ Deleting existing model...")
                shutil.rmtree(model_path)
                return True
            elif choice == "2":
                logger.info("🔄 Continuing with existing model...")
                return True
            else:
                logger.warning("Invalid choice. Please enter 1 or 2.")
        except Exception as e:
            logger.error(f"Error handling input: {e}")
            return False
    
def check_gpu_availability() -> bool:
    """Check GPU availability and log details"""
    if not torch.cuda.is_available():
        logger.warning("\n⚠️ No GPU detected!")
        logger.warning("Training on CPU will be significantly slower.")
        response = input("Do you want to continue with CPU training? [y/N]: ").lower()
        return response == 'y'
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"\n🎮 Found {gpu_count} GPU(s):")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        logger.info(f"   - GPU {i}: {gpu_name}")
    return True

def main():
    """Main training function"""
    try:
        # Check GPU availability first
        if not check_gpu_availability():
            logger.info("Training cancelled by user.")
            sys.exit(0)
            
        # Data path
        data_path = "ml_models/training_data/wikipedia_training_data.csv"
        model_path = "ml_models/bert_gpu_model"
        
        # Handle existing model
        if not handle_existing_model(model_path):
            sys.exit(0)
        
        logger.info(f"📂 Loading data from {data_path}")
        
        # Load and prepare data
        df = pd.read_csv(data_path)
        
        # Use clean text for better training
        X = df['text_clean']
        y = df['intent']
        
        # Encode labels to numerical values
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        
        # Save label encoder with the model
        os.makedirs(model_path, exist_ok=True)
        encoder_path = os.path.join(model_path, "label_encoder.pkl")
        pd.to_pickle(label_encoder, encoder_path)
        logger.info(f"✅ Saved label encoder to {encoder_path}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        # Initialize classifier with default or custom config
        classifier = BERTIntentClassifier(
            model_path=model_path,
            num_labels=len(df['intent'].unique()),
            config=DEFAULT_CONFIG
        )
        
        # Train model
        classifier.train(X_train, X_test, y_train, y_test)
        
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 