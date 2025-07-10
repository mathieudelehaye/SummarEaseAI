"""
Train BERT model for intent classification with GPU support
"""

import os
import sys
import logging
import traceback
import json
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Change logging level to show warnings (0=all, 1=info, 2=warning, 3=error)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Required imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification
)

# Don't suppress TF logging
logging.getLogger('tensorflow').setLevel(logging.INFO)

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

class SimpleTokenizer:
    """A basic tokenizer for our BERT model"""
    def __init__(self, vocab_size=30522, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
    def encode(self, texts, padding=True, truncation=True):
        """Simple encoding - just use character indices"""
        if isinstance(texts, str):
            texts = [texts]
            
        input_ids = []
        attention_masks = []
        
        for text in texts:
            # Convert to lowercase and split into words
            words = text.lower().split()
            
            # Truncate if needed
            if truncation and len(words) > self.max_length - 2:  # -2 for [CLS] and [SEP]
                words = words[:self.max_length - 2]
                
            # Create tokens - simple hash function to get consistent IDs
            tokens = [1]  # [CLS]
            for word in words:
                token_id = hash(word) % (self.vocab_size - 3) + 3  # Leave room for special tokens
                tokens.append(token_id)
            tokens.append(2)  # [SEP]
            
            # Create attention mask
            attention_mask = [1] * len(tokens)
            
            # Pad if needed
            if padding and len(tokens) < self.max_length:
                pad_length = self.max_length - len(tokens)
                tokens.extend([0] * pad_length)  # 0 is [PAD]
                attention_mask.extend([0] * pad_length)
                
            input_ids.append(tokens)
            attention_masks.append(attention_mask)
            
        return {
            'input_ids': np.array(input_ids, dtype=np.int32),
            'attention_mask': np.array(attention_masks, dtype=np.int32)
        }
        
    def save_pretrained(self, path):
        """Save tokenizer configuration"""
        config = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'tokenizer_config.json'), 'w') as f:
            json.dump(config, f)
            
    @classmethod
    def from_pretrained(cls, path):
        """Load tokenizer from configuration"""
        with open(os.path.join(path, 'tokenizer_config.json'), 'r') as f:
            config = json.load(f)
        return cls(**config)

class MemoryTracker:
    """Track memory usage during training"""
    def __init__(self):
        self.baseline_mb = 0
        self.peak_mb = 0
        
    def _get_memory_mb(self):
        """Get current GPU memory usage in MB"""
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            return memory_info['current'] / (1024 * 1024)
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

class BERTIntentClassifier:
    """BERT-based intent classifier"""
    def __init__(
        self,
        model_path,
        batch_size=16,
        learning_rate=1e-4,
        epochs=10,
        max_length=128,
        use_gradient_checkpointing=False
    ):
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_length = max_length
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self._log_config()
        
        # Initialize memory tracking
        self.memory_tracker = MemoryTracker()
        
    def _log_config(self):
        """Log training configuration"""
        self.logger.info("🚀 BERT Classifier initialized")
        self.logger.info(f"📊 Batch size: {self.batch_size}")
        self.logger.info(f"📈 Learning rate: {self.learning_rate}")
        self.logger.info(f"🔄 Epochs: {self.epochs}")
        self.logger.info(f"📏 Max sequence length: {self.max_length}")
        self.logger.info(f"💾 Save directory: {self.model_path}")
        self.logger.info("⚙️ Memory optimizations:")
        self.logger.info(f"   - Gradient checkpointing: {self.use_gradient_checkpointing}")
        self.logger.info(f"   - Dynamic padding: True")
        self.logger.info(f"   - Optimized attention: True")
        
    def _get_callbacks(self):
        """Get training callbacks"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=1,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_path / 'checkpoints' / 'best_model'),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
    def _build_model(self):
        """Build or load the model"""
        try:
            if (self.model_path / 'saved_model.pb').exists():
                self.logger.info(f"Loading model from {self.model_path}")
                self.model = tf.keras.models.load_model(str(self.model_path))
            else:
                self.logger.info("Creating new model")
                self.model = create_simple_bert_model()
                
            # Load or create tokenizer
            tokenizer_config = self.model_path / 'tokenizer_config.json'
            if tokenizer_config.exists():
                self.logger.info(f"Loading tokenizer from {self.model_path}")
                self.tokenizer = SimpleTokenizer.from_pretrained(str(self.model_path))
            else:
                self.logger.info("Creating new tokenizer")
                self.tokenizer = SimpleTokenizer(max_length=self.max_length)
                self.tokenizer.save_pretrained(str(self.model_path))
                
            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            return False
            
    def _prepare_data(self, texts, labels=None):
        """Prepare data for training or inference"""
        # Tokenize texts
        encoded = self.tokenizer.encode(
            texts,
            padding=True,
            truncation=True
        )
        
        if labels is not None:
            return encoded, labels
        return encoded
        
    def train(self, X_train, X_test, y_train, y_test):
        """Train the model"""
        try:
            self.logger.info("Starting training...")
            self.memory_tracker.log_memory("Before training")
            
            # Initialize model
            self.logger.info("Initializing model...")
            if not self._build_model():
                raise RuntimeError("Failed to build model")
                
            # Prepare data
            train_data, train_labels = self._prepare_data(X_train, y_train)
            test_data, test_labels = self._prepare_data(X_test, y_test)
            
            # Create dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': train_data['input_ids'],
                    'attention_mask': train_data['attention_mask']
                },
                train_labels
            )).shuffle(1000).batch(self.batch_size)
            
            test_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': test_data['input_ids'],
                    'attention_mask': test_data['attention_mask']
                },
                test_labels
            )).batch(self.batch_size)
            
            # Train
            self.logger.info("Training model...")
            history = self.model.fit(
                train_dataset,
                validation_data=test_dataset,
                epochs=self.epochs,
                callbacks=self._get_callbacks()
            )
            
            # Save final model
            self.model.save(str(self.model_path))
            self.logger.info(f"Model saved to {self.model_path}")
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.logger.error("Stack trace:", exc_info=True)
            raise

def main():
    """Main training function"""
    try:
        # Configure TensorFlow for DirectML
        os.environ['TF_DIRECTML_KERNEL_CACHE'] = '1'  # Enable kernel caching
        os.environ['TF_DIRECTML_ENABLE_TELEMETRY'] = '0'  # Disable telemetry
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Enable memory growth
        
        # Force float32 for DirectML
        tf.keras.mixed_precision.set_global_policy('float32')
        
        # Configure GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            # Use first GPU (should be the NVIDIA one)
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            logger.info(f"Using DirectML GPU: {physical_devices[0].name}")
            
            # Configure memory growth
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            except Exception as e:
                logger.info(f"Memory growth already configured: {e}")
        else:
            logger.warning("No DirectML GPU found, falling back to CPU")
            
        # Create a simple local BERT model
        logger.info("\nCreating local BERT model...")
        temp_model = create_simple_bert_model(max_length=128)  # Set consistent max_length
        num_params = temp_model.count_params()
        
        logger.info(f"\nModel Information:")
        logger.info(f"- Model: Simple BERT (2-layer)")
        logger.info(f"- Parameters: {num_params:,}")
        logger.info(f"- Size on disk: ~{num_params * 4 / (1024*1024):.1f}MB")
        
        # Ask for confirmation
        response = input("\nWould you like to proceed with training? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Training cancelled by user")
            return
            
        # Use model from bert_gpu_model directory
        model_path = Path("tensorflow_models/bert_gpu_model")
        
        # Check if old model exists and verify its location
        if model_path.exists():
            try:
                abs_path = model_path.resolve()
                logger.info("\nFound existing model:")
                logger.info(f"- Relative path: {model_path}")
                logger.info(f"- Absolute path: {abs_path}")
                
                # Check if it contains model files
                saved_model_pb = model_path / "saved_model.pb"
                if saved_model_pb.exists():
                    logger.info("- Verified: Contains saved_model.pb")
                else:
                    logger.info("- Warning: No saved_model.pb found")
                    
                variables_dir = model_path / "variables"
                if variables_dir.exists():
                    logger.info("- Verified: Contains variables directory")
                else:
                    logger.info("- Warning: No variables directory found")
                
                logger.info("\nThe model needs to be recreated with correct sequence length.")
                delete_response = input("Would you like to delete the old model and create a new one? (yes/no): ")
                if delete_response.lower() != 'yes':
                    logger.info("Training cancelled - cannot proceed without recreating model")
                    return
                    
                import shutil
                logger.info("Removing old model...")
                shutil.rmtree(model_path)
            except Exception as e:
                logger.error(f"Error verifying model directory: {e}")
                return
            
        # Create new model directory
        model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating and saving local BERT model...")
        model = create_simple_bert_model(max_length=128)  # Set consistent max_length
        
        # Compile model before saving
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        model.save(str(model_path))
        
        # Create and save tokenizer
        tokenizer = SimpleTokenizer(vocab_size=30522, max_length=128)  # Match model's max_length
        tokenizer.save_pretrained(str(model_path))
        logger.info("Saved model and tokenizer")
            
        logger.info(f"\nUsing model at: {model_path}")
        
        # Initialize classifier with new parameters
        classifier = BERTIntentClassifier(
            model_path=str(model_path),
            batch_size=16,
            learning_rate=1e-4,
            epochs=10,
            max_length=128,  # Match model's max_length
            use_gradient_checkpointing=False
        )
        
        # Load and preprocess data
        data_path = "tensorflow_models/training_data/enhanced_wikipedia_training_data.csv"
        df = pd.read_csv(data_path)
        texts = df['text'].values
        labels = df['intent'].values
        
        # Convert to binary classification
        binary_labels = np.array([1 if label == 'Finance' else 0 for label in labels])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, binary_labels,
            test_size=0.2,
            random_state=42,
            stratify=binary_labels
        )
        
        # Train model
        history = classifier.train(X_train, X_test, y_train, y_test)
        
        logger.info("✅ Training completed successfully!")
        return history
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

def create_simple_bert_model(vocab_size=30522, hidden_size=128, num_layers=2, max_length=128):
    """Create a simple BERT-like model locally without downloading"""
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
    
    # Embedding layer
    embedding_layer = tf.keras.layers.Embedding(vocab_size, hidden_size)(input_ids)
    
    # Transformer layers
    x = embedding_layer
    for _ in range(num_layers):
        # Self-attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=hidden_size // 4
        )(x, x, attention_mask=tf.cast(attention_mask[:, tf.newaxis, tf.newaxis, :], tf.float32))
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size * 4, activation='gelu'),
            tf.keras.layers.Dense(hidden_size)
        ])
        x = tf.keras.layers.Add()([x, ffn(x)])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Pooler
    pooled_output = tf.keras.layers.Dense(hidden_size, activation='tanh')(x[:, 0, :])
    
    # Classification head
    outputs = tf.keras.layers.Dense(2)(pooled_output)  # 2 classes
    
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=outputs
    )
    return model

if __name__ == "__main__":
    main() 