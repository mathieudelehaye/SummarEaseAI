#!/usr/bin/env python3
"""
Retrain TensorFlow Intent Classifier with Wikipedia Portal Data
Fix Beatles → Science misclassification by adding more Music training data
"""

import sys
import os
sys.path.append('..')

import numpy as np
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from utils.wikipedia_training_fetcher import WikipediaPortalFetcher
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaTrainedIntentClassifier:
    """Intent classifier trained with Wikipedia portal data"""
    
    def __init__(self):
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.max_sequence_length = 100
        self.vocab_size = 10000
        
    def load_existing_training_data(self) -> pd.DataFrame:
        """Load existing training data if available"""
        try:
            # Try to load existing training data
            existing_data = pd.read_csv('training_data.csv')
            logger.info(f"📊 Loaded existing training data: {len(existing_data)} samples")
            return existing_data
        except FileNotFoundError:
            logger.info("📊 No existing training data found, starting fresh")
            return pd.DataFrame(columns=['text', 'intent'])
    
    def collect_wikipedia_training_data(self, pages_per_portal: int = 50) -> pd.DataFrame:
        """Collect training data from Wikipedia portals"""
        logger.info(f"🌍 Collecting Wikipedia training data...")
        
        fetcher = WikipediaPortalFetcher()
        
        # Collect data from all portals
        all_portal_data = fetcher.collect_all_portals_data(pages_per_portal)
        
        # Convert to flat DataFrame
        training_samples = []
        
        for portal_name, portal_data in all_portal_data.items():
            for sample in portal_data:
                training_samples.append({
                    'text': sample['text'],
                    'intent': sample['intent'],
                    'source': f"Wikipedia:{portal_name}",
                    'title': sample['title']
                })
        
        # Add Beatles-specific samples to fix misclassification
        beatles_samples = fetcher.create_beatles_specific_samples()
        for sample in beatles_samples:
            training_samples.append({
                'text': sample['text'],
                'intent': sample['intent'],
                'source': "Manual:Beatles",
                'title': sample['title']
            })
        
        # Create DataFrame
        df = pd.DataFrame(training_samples)
        
        logger.info(f"✅ Collected {len(df)} training samples from Wikipedia")
        
        # Print distribution
        intent_counts = df['intent'].value_counts()
        logger.info("📊 Intent distribution:")
        for intent, count in intent_counts.items():
            logger.info(f"   {intent}: {count} samples")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for training"""
        logger.info("🔄 Preparing training data...")
        
        # Clean text data
        df['text_clean'] = df['text'].str.lower().str.strip()
        
        # Remove duplicates
        df_clean = df.drop_duplicates(subset=['text_clean', 'intent'])
        logger.info(f"📊 After removing duplicates: {len(df_clean)} samples")
        
        # Prepare features and labels
        texts = df_clean['text_clean'].tolist()
        labels = df_clean['intent'].tolist()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Tokenize texts
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        y = labels_encoded
        
        logger.info(f"✅ Prepared {len(X)} samples for training")
        logger.info(f"📊 Vocabulary size: {len(self.tokenizer.word_index)}")
        logger.info(f"📊 Label classes: {list(self.label_encoder.classes_)}")
        
        return X, y, df_clean
    
    def create_model(self, num_classes: int) -> Sequential:
        """Create LSTM model architecture"""
        model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=128, input_length=self.max_sequence_length),
            LSTM(64, dropout=0.3, recurrent_dropout=0.3),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("🏗️ Created LSTM model architecture")
        return model
    
    def train_model(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model"""
        logger.info("🚀 Starting model training...")
        
        num_classes = len(np.unique(y))
        self.model = self.create_model(num_classes)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Training samples: {len(X_train)}")
        logger.info(f"📊 Validation samples: {len(X_val)}")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info("✅ Model training completed!")
        return history
    
    def evaluate_model(self, X, y):
        """Evaluate model performance"""
        logger.info("📊 Evaluating model...")
        
        # Overall accuracy
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        logger.info(f"✅ Model accuracy: {accuracy:.4f}")
        
        # Predictions
        predictions = self.model.predict(X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Test Beatles-specific queries
        beatles_queries = [
            "Who were the Beatles?",
            "Tell me about The Beatles",
            "Beatles music",
            "British rock band Beatles"
        ]
        
        logger.info("🎵 Testing Beatles queries:")
        for query in beatles_queries:
            intent, confidence = self.predict_intent(query)
            status = "✅" if intent == "Music" else "❌"
            logger.info(f"   {status} '{query}' → {intent} (confidence: {confidence:.3f})")
        
        return accuracy, predicted_classes
    
    def predict_intent(self, text: str) -> tuple:
        """Predict intent for a single text"""
        if not self.model or not self.tokenizer or not self.label_encoder:
            return "Unknown", 0.0
        
        # Preprocess text
        text_clean = text.lower().strip()
        sequence = self.tokenizer.texts_to_sequences([text_clean])
        padded = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
        
        # Predict
        prediction = self.model.predict(padded, verbose=0)[0]
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[predicted_class_idx]
        
        # Decode label
        intent = self.label_encoder.classes_[predicted_class_idx]
        
        return intent, confidence
    
    def save_model(self, base_path="saved_model"):
        """Save the trained model and preprocessing objects"""
        os.makedirs(base_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(base_path, "model.h5")
        self.model.save(model_path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(base_path, "tokenizer.pickle")
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save label encoder
        label_encoder_path = os.path.join(base_path, "label_encoder.pickle")
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info(f"💾 Model saved to: {base_path}")
        logger.info(f"   - Model: {model_path}")
        logger.info(f"   - Tokenizer: {tokenizer_path}")
        logger.info(f"   - Label encoder: {label_encoder_path}")
    
    def plot_training_history(self, history, save_path="training_history.png"):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        
        logger.info(f"📊 Training history plot saved to: {save_path}")

def main():
    """Main training pipeline"""
    logger.info("🚀 Starting Wikipedia-enhanced intent classifier training")
    
    # Initialize classifier
    classifier = WikipediaTrainedIntentClassifier()
    
    # Collect Wikipedia training data
    wikipedia_df = classifier.collect_wikipedia_training_data(pages_per_portal=30)
    
    # Load existing data and combine
    existing_df = classifier.load_existing_training_data()
    
    if not existing_df.empty:
        # Ensure columns match
        if 'source' not in existing_df.columns:
            existing_df['source'] = 'Original'
        if 'title' not in existing_df.columns:
            existing_df['title'] = 'Original Data'
        
        # Combine datasets
        combined_df = pd.concat([existing_df, wikipedia_df], ignore_index=True)
    else:
        combined_df = wikipedia_df
    
    logger.info(f"📊 Total training data: {len(combined_df)} samples")
    
    # Prepare training data
    X, y, df_clean = classifier.prepare_training_data(combined_df)
    
    # Train model
    history = classifier.train_model(X, y, epochs=30)
    
    # Evaluate model
    accuracy, predictions = classifier.evaluate_model(X, y)
    
    # Save model
    classifier.save_model()
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Save the enhanced training data
    df_clean.to_csv('enhanced_training_data.csv', index=False)
    logger.info("💾 Enhanced training data saved to: enhanced_training_data.csv")
    
    logger.info("🎉 Training pipeline completed successfully!")
    logger.info(f"🎯 Final model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 