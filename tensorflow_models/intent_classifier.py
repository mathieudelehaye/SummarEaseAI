# TensorFlow intent classifier model
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import logging
from typing import List, Dict, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self, vocab_size: int = 10000, max_length: int = 100, embedding_dim: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.intent_categories = [
            'History', 'Science', 'Biography', 'Technology', 
            'Arts', 'Sports', 'Politics', 'Geography', 'General'
        ]
        
    def create_model(self, num_categories: int) -> tf.keras.Model:
        """Create the neural network model for intent classification"""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=self.vocab_size, 
                output_dim=self.embedding_dim, 
                input_length=self.max_length
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.3)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_categories, activation='softmax')
        ])
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_training_data(self) -> Tuple[List[str], List[str]]:
        """Generate training data for intent classification"""
        # Sample training data - in practice, you'd load from Wikipedia or other sources
        training_data = [
            # History
            ("World War II", "History"),
            ("American Revolution", "History"),
            ("French Revolution", "History"),
            ("Ancient Rome", "History"),
            ("Medieval period", "History"),
            
            # Science
            ("Quantum mechanics", "Science"),
            ("Theory of relativity", "Science"),
            ("DNA structure", "Science"),
            ("Climate change", "Science"),
            ("Photosynthesis", "Science"),
            
            # Biography
            ("Albert Einstein", "Biography"),
            ("Marie Curie", "Biography"),
            ("Leonardo da Vinci", "Biography"),
            ("Winston Churchill", "Biography"),
            ("Nelson Mandela", "Biography"),
            
            # Technology
            ("Artificial intelligence", "Technology"),
            ("Machine learning", "Technology"),
            ("Internet", "Technology"),
            ("Blockchain", "Technology"),
            ("Renewable energy", "Technology"),
            
            # Arts
            ("Renaissance art", "Arts"),
            ("Classical music", "Arts"),
            ("Modern literature", "Arts"),
            ("Abstract painting", "Arts"),
            ("Ballet", "Arts"),
            
            # Sports
            ("Olympic Games", "Sports"),
            ("Football World Cup", "Sports"),
            ("Tennis championships", "Sports"),
            ("Basketball", "Sports"),
            ("Swimming", "Sports"),
            
            # Politics
            ("Democracy", "Politics"),
            ("United Nations", "Politics"),
            ("Presidential election", "Politics"),
            ("Constitution", "Politics"),
            ("Diplomacy", "Politics"),
            
            # Geography
            ("Mountain ranges", "Geography"),
            ("Ocean currents", "Geography"),
            ("Capital cities", "Geography"),
            ("Climate zones", "Geography"),
            ("Continents", "Geography"),
        ]
        
        texts, labels = zip(*training_data)
        return list(texts), list(labels)
    
    def train_model(self, texts: List[str], labels: List[str], epochs: int = 10, validation_split: float = 0.2):
        """Train the intent classification model"""
        logger.info("Starting model training...")
        
        # Prepare tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        # Prepare labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            padded_sequences, encoded_labels, 
            test_size=validation_split, 
            random_state=42, 
            stratify=encoded_labels
        )
        
        # Create model
        num_categories = len(self.label_encoder.classes_)
        self.model = self.create_model(num_categories)
        
        logger.info(f"Model architecture:\n{self.model.summary()}")
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        logger.info("Model training completed!")
        return history
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Predict intent for a given text"""
        if not self.model or not self.tokenizer or not self.label_encoder:
            return "General", 0.0
        
        try:
            # Preprocess text
            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_length, padding='post')
            
            # Make prediction
            prediction = self.model.predict(padded_sequence, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            # Convert back to label
            intent = self.label_encoder.inverse_transform([predicted_class])[0]
            
            logger.info(f"Predicted intent: {intent} (confidence: {confidence:.2f})")
            return intent, confidence
            
        except Exception as e:
            logger.error(f"Error predicting intent: {str(e)}")
            return "General", 0.0
    
    def save_model(self, model_dir: str = "tensorflow_models/saved_model"):
        """Save the trained model and components"""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model:
            self.model.save(os.path.join(model_dir, "intent_model.h5"))
        
        if self.tokenizer:
            with open(os.path.join(model_dir, "tokenizer.pkl"), "wb") as f:
                pickle.dump(self.tokenizer, f)
        
        if self.label_encoder:
            with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as f:
                pickle.dump(self.label_encoder, f)
        
        logger.info(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir: str = "tensorflow_models/saved_model"):
        """Load a pre-trained model and components"""
        try:
            model_path = os.path.join(model_dir, "intent_model.h5")
            tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
            encoder_path = os.path.join(model_dir, "label_encoder.pkl")
            
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
            
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, "rb") as f:
                    self.tokenizer = pickle.load(f)
            
            if os.path.exists(encoder_path):
                with open(encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
            
            logger.info(f"Model loaded from {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

# Global classifier instance
intent_classifier = IntentClassifier()

def create_model(vocab_size, max_length, num_categories):
    """Legacy function for backward compatibility"""
    classifier = IntentClassifier(vocab_size, max_length)
    return classifier.create_model(num_categories)

def get_intent_classifier():
    """Get the global intent classifier instance"""
    return intent_classifier
