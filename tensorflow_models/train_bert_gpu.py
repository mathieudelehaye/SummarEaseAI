"""
Train BERT model for intent classification with PyTorch GPU support
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
import psutil
try:
    import GPUtil  # Package name is GPUtil but we use it in lowercase
    gputil = GPUtil  # Alias for consistent lowercase usage
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("\n❌ GPUtil package not found!")
    logger.error("   Please install it with: pip install GPUtil")
    sys.exit(1)

import signal
import sys
import time  # Add time import for GPU testing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def setup_gpu():
    """Setup and verify GPU availability"""
    if not torch.cuda.is_available():
        logger.warning("\n🚨 CUDA is not available. Training on CPU will be SIGNIFICANTLY slower!")
        logger.warning("   A single epoch could take hours instead of minutes.")
        logger.info("\n💡 To enable GPU support, make sure you have:")
        logger.info("   1. An NVIDIA GPU")
        logger.info("   2. CUDA Toolkit 12.1 installed")
        logger.info("   3. Proper PyTorch CUDA version (current torch: %s)", torch.__version__)
        logger.info("\n📝 Installation Guide:")
        logger.info("   1. Install CUDA Toolkit 12.1 from NVIDIA website")
        logger.info("   2. Install PyTorch with CUDA: pip install torch==2.1.1+cu121 -f https://download.pytorch.org/whl/cu121")
        return False
        
    # Get GPU information
    gpu_id = 0  # Use first GPU by default
    try:
        gpus = gputil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Get primary GPU
            logger.info("\n🎮 GPU Information:")
            logger.info(f"   - Name: {gpu.name}")
            logger.info(f"   - Memory Total: {gpu.memoryTotal}MB")
            logger.info(f"   - Memory Free: {gpu.memoryFree}MB")
            logger.info(f"   - Memory Used: {gpu.memoryUsed}MB")
            logger.info(f"   - GPU Load: {gpu.load*100:.1f}%")
            
            # Check if GPU has enough memory (need at least 2GB free)
            if gpu.memoryFree < 2000:  # 2000MB = 2GB
                logger.warning("\n⚠️ Less than 2GB GPU memory available!")
                logger.warning("   This may cause out-of-memory errors during training.")
                logger.warning("   Consider closing other GPU applications or reducing batch size.")
                return False
                
        else:
            logger.warning("\n⚠️ No NVIDIA GPUs detected!")
            logger.warning("   This could mean:")
            logger.warning("   1. You don't have an NVIDIA GPU")
            logger.warning("   2. GPU drivers are not properly installed")
            logger.warning("   3. CUDA is not properly installed")
            return False
            
    except Exception as e:
        logger.warning(f"\n⚠️ Error getting GPU information: {e}")
        logger.warning("   This might indicate driver or CUDA installation issues.")
        return False
        
    # Set up PyTorch device
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    # Test CUDA memory allocation
    try:
        logger.info("\n🧪 Testing CUDA memory allocation...")
        test_tensor = torch.zeros((1000, 1000), device=device)  # Allocate ~4MB
        del test_tensor
        torch.cuda.empty_cache()
        logger.info("✅ CUDA memory test successful!")
    except Exception as e:
        logger.error(f"\n❌ CUDA memory test failed: {e}")
        logger.error("   This indicates a problem with CUDA setup or GPU access.")
        return False
        
    # Log CUDA configuration
    logger.info("\n🚀 CUDA Configuration:")
    logger.info(f"   - CUDA Version: {torch.version.cuda}")
    logger.info(f"   - PyTorch Version: {torch.__version__}")
    logger.info(f"   - GPU Device: {torch.cuda.get_device_name(gpu_id)}")
    logger.info(f"   - GPU Architecture: {torch.cuda.get_device_capability(gpu_id)}")
    logger.info(f"   - Max Memory Allocated: {torch.cuda.max_memory_allocated(device)/1e9:.2f}GB")
    
    return True

class MemoryTracker:
    """Track GPU memory usage during training"""
    def __init__(self):
        self.baseline_mb = 0
        self.peak_mb = 0
        
    def _get_memory_mb(self):
        """Get current GPU memory usage in MB"""
        try:
            memory_used = torch.cuda.memory_allocated() / (1024 * 1024)
            return memory_used
        except:
            return 0
            
    def log_memory(self, checkpoint: str):
        """Log memory usage at checkpoint"""
        try:
            current_mb = self._get_memory_mb()
            
            # Set baseline on first measurement
            if self.baseline_mb == 0:
                self.baseline_mb = current_mb
                
            # Update peak
            self.peak_mb = max(self.peak_mb, current_mb)
            
            # Calculate increase
            increase = current_mb - self.baseline_mb
            
            logging.info(f"💾 Memory at {checkpoint}:")
            logging.info(f"   Current: {current_mb:.1f}MB")
            logging.info(f"   Peak: {self.peak_mb:.1f}MB")
            logging.info(f"   Increase: {increase:.1f}MB")
            
        except Exception as e:
            logging.warning(f"Unable to log memory usage: {e}")

class IntentDataset(Dataset):
    """Custom dataset for intent classification"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Convert pandas Series to list to avoid indexing issues
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
    """BERT-based intent classifier using PyTorch"""
    def __init__(
        self,
        model_path: str,
        num_labels: int = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        epochs: int = 10,
        max_length: int = 128,
        model_name: str = "bert-base-uncased",
        gradient_checkpointing: bool = False
    ):
        self.model_path = Path(model_path)
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_length = max_length
        self.model_name = model_name
        self.gradient_checkpointing = gradient_checkpointing
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            # Log GPU info
            gpu = gputil.getGPUs()[0]
            logger.info(f"🎮 Training on GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   - Memory Free: {gpu.memoryFree}MB")
            
            # Verify CUDA is actually being used
            logger.info("\n🔍 Verifying CUDA Setup:")
            logger.info(f"   - CUDA Available: {torch.cuda.is_available()}")
            logger.info(f"   - Current Device: {torch.cuda.current_device()}")
            logger.info(f"   - Device Count: {torch.cuda.device_count()}")
            
            # Test tensor operations on GPU
            logger.info("\n🧪 Testing GPU Tensor Operations:")
            try:
                x = torch.randn(1000, 1000, device=self.device)
                y = torch.randn(1000, 1000, device=self.device)
                start_time = time.time()
                z = torch.matmul(x, y)
                del z  # Clean up
                torch.cuda.synchronize()  # Wait for GPU
                end_time = time.time()
                logger.info(f"   - Matrix multiplication time: {(end_time - start_time)*1000:.2f}ms")
                logger.info("   ✅ GPU tensor operations working correctly")
            except Exception as e:
                logger.error(f"   ❌ GPU tensor test failed: {e}")
        else:
            logger.warning("⚠️ Training on CPU - this will be slow!")
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        self._interrupt_requested = False
        
        # Initialize memory tracking
        self.memory_tracker = MemoryTracker()
        
        # Initialize model and configuration
        self._log_config()
        if not self._build_model():
            raise RuntimeError("Failed to initialize model")

    def _signal_handler(self, signum, frame):
        """Handle interrupt signal (Ctrl+C)"""
        logger.info("\n\n⚠️ Interrupt received! Cleaning up...")
        self._interrupt_requested = True
        
    def _cleanup(self):
        """Cleanup resources"""
        if self.device.type == "cuda":
            try:
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Reset device
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Error during CUDA cleanup: {e}")
        
        # Close progress bars if any are open
        try:
            sys.stdout.write('\n')
            sys.stdout.flush()
        except:
            pass

    def _monitor_gpu_usage(self, description=""):
        """Monitor GPU usage during training"""
        if self.device.type == "cuda":
            try:
                gpu = gputil.getGPUs()[0]
                logger.info(f"\n📊 GPU Usage {description}:")
                logger.info(f"   - Memory Used: {gpu.memoryUsed}MB")
                logger.info(f"   - Memory Free: {gpu.memoryFree}MB")
                logger.info(f"   - GPU Load: {gpu.load*100:.1f}%")
                logger.info(f"   - GPU Temperature: {gpu.temperature}°C")
            except Exception as e:
                logger.warning(f"Could not monitor GPU usage: {e}")

    def _log_config(self):
        """Log training configuration"""
        self.logger.info("🚀 BERT Classifier initialized")
        self.logger.info(f"🖥️ Using device: {self.device}")
        self.logger.info(f"🤗 Model: {self.model_name}")
        self.logger.info(f"📊 Batch size: {self.batch_size}")
        self.logger.info(f"📈 Learning rate: {self.learning_rate}")
        self.logger.info(f"🔄 Epochs: {self.epochs}")
        self.logger.info(f"📏 Max sequence length: {self.max_length}")
        self.logger.info(f"💾 Save directory: {self.model_path}")
        self.logger.info("⚙️ Memory optimizations:")
        self.logger.info(f"   - Gradient checkpointing: {self.gradient_checkpointing}")
         
    def _build_model(self):
        """Build or load the model"""
        try:
            if (self.model_path / 'model.safetensors').exists():
                self.logger.info(f"Loading model from {self.model_path}")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_path),
                    use_safetensors=True  # Explicitly use safetensors
                )
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            else:
                self.logger.info("Creating new model")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=self.num_labels,
                    use_safetensors=True  # Explicitly use safetensors
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
            # Enable gradient checkpointing if requested
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                
            # Move model to device and enable training mode
            self.model = self.model.to(self.device)
            self.model.train()
            
            # Log model size
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"📊 Model size:")
            logger.info(f"   - Total parameters: {total_params:,}")
            logger.info(f"   - Trainable parameters: {trainable_params:,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            return False

    def train(self, X_train, X_test, y_train, y_test):
        """Train the model"""
        try:
            # Create datasets
            train_dataset = IntentDataset(X_train, y_train, self.tokenizer, self.max_length)
            test_dataset = IntentDataset(X_test, y_test, self.tokenizer, self.max_length)
            
            # Create data loaders with appropriate num_workers for GPU
            num_workers = 4 if self.device.type == "cuda" else 0
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if self.device.type == "cuda" else False
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == "cuda" else False
            )
            
            # Initialize optimizer and scheduler
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01
            )
            
            num_training_steps = len(train_loader) * self.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_training_steps // 10,
                num_training_steps=num_training_steps
            )
            
            # Training loop
            best_val_loss = float('inf')
            for epoch in range(self.epochs):
                if self._interrupt_requested:
                    logger.info("Training interrupted by user")
                    break
                    
                self.logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
                
                # Training phase
                self.model.train()
                train_loss = 0
                train_steps = 0
                
                # Monitor GPU at start of epoch
                self._monitor_gpu_usage(f"Start of Epoch {epoch + 1}")
                
                progress_bar = tqdm(train_loader, desc="Training")
                try:
                    for batch_idx, batch in enumerate(progress_bar):
                        if self._interrupt_requested:
                            break
                            
                        # Move batch to device
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        # Forward pass
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        train_loss += loss.item()
                        train_steps += 1
                        
                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            'loss': f"{train_loss/train_steps:.4f}"
                        })
                        
                        # Monitor GPU every 5 batches
                        if batch_idx % 5 == 0:
                            self._monitor_gpu_usage(f"Batch {batch_idx}")
                        
                        # Clear GPU memory if needed
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                            
                finally:
                    progress_bar.close()
                
                if self._interrupt_requested:
                    break
                    
                # Calculate average training loss
                avg_train_loss = train_loss / train_steps if train_steps > 0 else float('inf')
                
                # Monitor GPU before validation
                self._monitor_gpu_usage("Before Validation")
                
                # Validation phase
                self.model.eval()
                val_loss = 0
                val_steps = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in tqdm(test_loader, desc="Validation"):
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        val_loss += outputs.loss.item()
                        val_steps += 1
                        
                        # Calculate accuracy
                        predictions = torch.argmax(outputs.logits, dim=1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
                        
                        # Clear GPU memory if needed
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                
                avg_val_loss = val_loss / val_steps
                accuracy = correct / total
                
                self.logger.info(f"Train Loss: {avg_train_loss:.4f}")
                self.logger.info(f"Val Loss: {avg_val_loss:.4f}")
                self.logger.info(f"Accuracy: {accuracy:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss and not self._interrupt_requested:
                    best_val_loss = avg_val_loss
                    self.logger.info(f"Saving best model to {self.model_path}")
                    self.model.save_pretrained(
                        str(self.model_path),
                        safe_serialization=True  # Use safetensors for saving
                    )
                    self.tokenizer.save_pretrained(str(self.model_path))
                
                # Monitor GPU at end of epoch
                self._monitor_gpu_usage(f"End of Epoch {epoch + 1}")
                
                # Log memory usage
                self.memory_tracker.log_memory(f"Epoch {epoch + 1}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            return False
            
        finally:
            self._cleanup()
            if self._interrupt_requested:
                logger.info("Training stopped gracefully. Partial progress has been saved.")
                # Save final model state if requested
                try:
                    save_path = self.model_path / "interrupted_checkpoint"
                    self.model.save_pretrained(str(save_path))
                    self.tokenizer.save_pretrained(str(save_path))
                    logger.info(f"Saved interrupted model state to {save_path}")
                except Exception as e:
                    logger.warning(f"Could not save interrupted model state: {e}")

def main():
    """Main training function"""
    # Setup GPU first
    gpu_available = setup_gpu()
    if not gpu_available:
        logger.warning("\n⚠️ GPU acceleration is not available!")
        logger.warning("   Training on CPU will be VERY slow (10-20x slower than GPU)")
        logger.warning("   A single epoch could take hours instead of minutes")
        response = input("\n❓ Do you want to continue with CPU training anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Training cancelled.")
            return
        logger.info("\n⏳ Continuing with CPU training... (this will be slow)")
    
    # Load and preprocess data
    try:
        data_path = Path("tensorflow_models/training_data/enhanced_wikipedia_training_data.csv")
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return
            
        logger.info(f"Loading data from {data_path}")
        try:
            df = pd.read_csv(data_path)
            if 'text' not in df.columns or 'intent' not in df.columns:
                logger.error("Data file must contain 'text' and 'intent' columns")
                return
        except Exception as e:
            logger.error(f"Error reading data file: {e}")
            return
            
        # Encode labels
        label_encoder = LabelEncoder()
        try:
            y = label_encoder.fit_transform(df['intent'])
            X = df['text'].fillna("")  # Handle any NaN values
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return
            
        logger.info(f"Found {len(label_encoder.classes_)} unique labels")
        logger.info(f"Total samples: {len(df)}")
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return
            
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Testing samples: {len(X_test)}")
        
        # Model parameters
        model_params = {
            'model_path': "tensorflow_models/bert_gpu_model",
            'num_labels': len(label_encoder.classes_),
            'batch_size': 32,
            'learning_rate': 2e-5,
            'epochs': 5,
            'max_length': 128,
            'model_name': "prajjwal1/bert-tiny",  # Tiny 4-layer BERT model (~4.4M params)
            'gradient_checkpointing': True
        }
        
        # Print model parameters and size info
        print("\n🤖 Model Parameters:")
        for key, value in model_params.items():
            print(f"   {key}: {value}")
        print("\n📊 Model Size:")
        print("   - Architecture: TinyBERT")
        print("   - Layers: 4")
        print("   - Hidden size: 312")
        print("   - Parameters: ~4.4M")  # Fixed from incorrect 14M
        print("   - Disk size: ~17MB")  # Fixed from incorrect 53MB
            
        # Check if model exists
        model_path = Path(model_params['model_path'])
        if model_path.exists():
            response = input("\n⚠️ Model directory exists. Delete and retrain? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(model_path)
                logger.info(f"Deleted existing model at {model_path}")
            else:
                print("Exiting without training.")
                return
                
        # Confirm training
        response = input("\n🚀 Start training with these parameters? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
            
        # Initialize and train model
        classifier = BERTIntentClassifier(**model_params)
        if not classifier._build_model():
            logger.error("Failed to build model")
            return
            
        if classifier.train(X_train, X_test, y_train, y_test):
            logger.info("Training completed successfully!")
        else:
            logger.error("Training failed")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 