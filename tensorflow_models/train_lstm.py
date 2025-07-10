"""
Train LSTM Intent Classifier
Uses enhanced Wikipedia data to train a text classification model
Supports DirectML GPU acceleration
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF debug/info messages

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

import tensorflow as tf

# Configure DirectML for GPU support
try:
    import tensorflow_directml
    physical_devices = tf.config.list_physical_devices()
    dml_visible_devices = tf.config.list_physical_devices('DML')
    if dml_visible_devices:
        logger.info(f"DirectML devices found: {len(dml_visible_devices)}")
        for device in dml_visible_devices:
            logger.info(f"  {device.name}")
    else:
        logger.warning("No DirectML devices found. Running on CPU only.")
    
    logger.info(f"All available devices:")
    for device in physical_devices:
        logger.info(f"  {device.name} ({device.device_type})")
except ImportError:
    logger.warning("tensorflow-directml-plugin not found. Install with: pip install tensorflow-directml-plugin")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# Training parameters
MAX_WORDS = 10000  # Maximum vocabulary size
MAX_LEN = 100     # Maximum sequence length
EMBED_DIM = 100   # Word embedding dimension
LSTM_UNITS = 100  # LSTM layer units
EPOCHS = 10
BATCH_SIZE = 32

def check_gpu_support():
    """Check and log GPU/DirectML support status"""
    logger.info("Checking GPU support...")
    
    # Check TensorFlow version
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Check if CUDA is available (for systems with NVIDIA GPUs)
    cuda_available = len(tf.config.list_physical_devices('GPU')) > 0
    if cuda_available:
        logger.info("CUDA GPU devices found:")
        for device in tf.config.list_physical_devices('GPU'):
            logger.info(f"  {device.name}")
    
    # Check DirectML support
    dml_available = len(tf.config.list_physical_devices('DML')) > 0
    if dml_available:
        logger.info("DirectML devices found:")
        for device in tf.config.list_physical_devices('DML'):
            logger.info(f"  {device.name}")
    
    if not (cuda_available or dml_available):
        logger.warning("No GPU acceleration available. Running on CPU only.")
    
    return cuda_available or dml_available

def load_data(data_path):
    """Load and preprocess data from CSV file."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Filter to use same categories as Wikipedia training
    valid_categories = ['Biography', 'History', 'Music', 'Science', 'Sports', 'Technology', 'Finance']
    df = df[df['intent'].isin(valid_categories)]
    logger.info(f"Dataset size after category filtering: {len(df)} samples")
    logger.info(f"Intents: {sorted(df['intent'].unique().tolist())}")

    # Prepare data
    texts = df['text_clean'].values
    labels = df['intent'].values

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Testing samples: {len(X_test)}")
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")
    logger.info(f"Classes: {sorted(label_encoder.classes_.tolist())}")

    return X_train, X_test, y_train, y_test, label_encoder

def prepare_tokenizer(texts):
    """Create and fit tokenizer"""
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    
    logger.info(f"Vocabulary size: {len(tokenizer.word_index)}")
    logger.info(f"Using top {MAX_WORDS} words")
    
    return tokenizer

def build_model(num_classes, vocab_size):
    """Create LSTM model"""
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)
    
    return model

def save_model(model, tokenizer, label_encoder, save_dir, training_history=None):
    """Save model and associated files"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_dir = save_dir / "lstm_model"
    model.save(model_dir)
    logger.info(f"Model saved to {model_dir}")
    
    # Save tokenizer
    tokenizer_path = save_dir / "tokenizer.json"
    tokenizer_json = tokenizer.to_json()
    tokenizer_path.write_text(tokenizer_json)
    logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    # Save label encoder
    encoder_path = save_dir / "label_encoder.pkl"
    import pickle
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Label encoder saved to {encoder_path}")
    
    # Save training history if available
    if training_history:
        history_path = save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy values to float for JSON serialization
            history_dict = {}
            for key, values in training_history.history.items():
                history_dict[key] = [float(val) for val in values]
            json.dump(history_dict, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    
    # Save metadata
    metadata = {
        'max_length': MAX_LEN,
        'vocab_size': MAX_WORDS,
        'embed_dim': EMBED_DIM,
        'lstm_units': LSTM_UNITS,
        'labels': label_encoder.classes_.tolist(),
        'gpu_support': check_gpu_support()
    }
    metadata_path = save_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main training function"""
    try:
        # Check GPU support first
        gpu_available = check_gpu_support()
        
        # Setup paths
        current_dir = Path(__file__).resolve().parent
        data_path = current_dir / "training_data" / "enhanced_training_data.csv"
        save_dir = current_dir / "lstm_model"
        logs_dir = current_dir / "logs" / "lstm_training"
        
        # Create directories
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        X_train, X_test, y_train, y_test, label_encoder = load_data(data_path)
        logger.info(f"Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")
        
        # Prepare text data
        tokenizer = prepare_tokenizer(X_train)
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences
        X_train = pad_sequences(X_train, maxlen=MAX_LEN)
        X_test = pad_sequences(X_test, maxlen=MAX_LEN)
        
        # Build and train model
        num_classes = len(label_encoder.classes_)
        vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
        
        model = build_model(num_classes, vocab_size)
        logger.info("Model built successfully")
        
        # Setup callbacks
        callbacks = [
            TensorBoard(
                log_dir=str(logs_dir),
                histogram_freq=1,
                write_graph=True
            ),
            ModelCheckpoint(
                str(save_dir / "checkpoints" / "model_{epoch:02d}_{val_accuracy:.4f}.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Save everything
        save_model(model, tokenizer, label_encoder, save_dir, history)
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    main() 