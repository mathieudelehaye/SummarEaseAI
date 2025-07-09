"""
Intent Classification using TensorFlow Neural Networks

This module provides intent classification using a bidirectional LSTM model.
It includes fallback behavior when no trained model is available.
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle
from pathlib import Path

# Suppress TensorFlow logging before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    # Additional TensorFlow logging suppression
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Intent classification will use fallback.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    Intent classification using TensorFlow bidirectional LSTM
    """
    
    def __init__(self):
        """Initialize the intent classifier"""
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_sequence_length = 100
        self.vocab_size = 10000
        self.embedding_dim = 100
        self.lstm_units = 64
        
        # Intent categories
        self.intent_categories = [
            'History', 'Science', 'Biography', 'Technology', 
            'Arts', 'Sports', 'Politics', 'Geography', 'General'
        ]
        
        # Fallback rules for when model is not available
        self.fallback_rules = {
            'History': ['war', 'battle', 'ancient', 'historical', 'century', 'empire', 'revolution', 'medieval', 'dynasty', 'timeline'],
            'Science': ['physics', 'chemistry', 'biology', 'quantum', 'molecule', 'experiment', 'theory', 'scientific', 'research', 'discovery'],
            'Biography': ['biography', 'life', 'born', 'died', 'person', 'people', 'who was', 'tell me about', 'famous', 'leader'],
            'Technology': ['technology', 'computer', 'internet', 'software', 'digital', 'artificial intelligence', 'machine learning', 'programming', 'innovation'],
            'Arts': ['art', 'painting', 'music', 'literature', 'poetry', 'novel', 'artist', 'composer', 'renaissance', 'culture'],
            'Sports': ['sports', 'olympic', 'game', 'football', 'basketball', 'tennis', 'soccer', 'athlete', 'championship', 'tournament'],
            'Politics': ['politics', 'government', 'democracy', 'election', 'president', 'congress', 'law', 'constitution', 'policy', 'diplomatic'],
            'Geography': ['geography', 'mountain', 'river', 'ocean', 'continent', 'country', 'climate', 'location', 'region', 'geological'],
            'General': ['general', 'information', 'knowledge', 'facts', 'trivia', 'miscellaneous', 'various', 'different', 'multiple']
        }
        
        logger.info(f"Intent classifier initialized (TensorFlow available: {TF_AVAILABLE})")
    
    def prepare_training_data(self) -> Tuple[List[str], List[str]]:
        """
        Generate comprehensive training data for intent classification
        """
        training_data = [
            # History
            ("World War II battles and strategies", "History"),
            ("American Revolution timeline", "History"),
            ("Ancient Rome empire expansion", "History"),
            ("French Revolution causes and effects", "History"),
            ("Medieval period lifestyle", "History"),
            ("Cold War events and consequences", "History"),
            ("What happened in 1969?", "History"),
            ("Historical events of the 20th century", "History"),
            ("Civil War battle outcomes", "History"),
            ("Viking exploration routes", "History"),
            
            # Science
            ("Quantum mechanics principles", "Science"),
            ("Theory of relativity explained", "Science"),
            ("DNA structure and function", "Science"),
            ("Climate change effects on environment", "Science"),
            ("Photosynthesis process in plants", "Science"),
            ("How does gravity work?", "Science"),
            ("Chemical bonding types", "Science"),
            ("Evolution by natural selection", "Science"),
            ("Solar system formation", "Science"),
            ("Electromagnetic wave properties", "Science"),
            ("Explain quantum physics concepts", "Science"),
            ("What is thermodynamics?", "Science"),
            ("Nuclear fusion process", "Science"),
            ("Particle physics experiments", "Science"),
            ("Biochemistry reactions", "Science"),
            ("Molecular biology research", "Science"),
            ("Physics laws and principles", "Science"),
            ("Scientific method explanation", "Science"),
            ("Chemistry periodic table", "Science"),
            ("Biology cell structures", "Science"),
            
            # Biography
            ("Albert Einstein biography and discoveries", "Biography"),
            ("Marie Curie life story", "Biography"),
            ("Leonardo da Vinci achievements", "Biography"),
            ("Winston Churchill leadership", "Biography"),
            ("Nelson Mandela legacy", "Biography"),
            ("Who was William Shakespeare?", "Biography"),
            ("Gandhi's philosophy and methods", "Biography"),
            ("Nikola Tesla inventions", "Biography"),
            ("Charles Darwin discoveries", "Biography"),
            ("Cleopatra's reign in Egypt", "Biography"),
            
            # Technology
            ("Artificial intelligence applications", "Technology"),
            ("Machine learning algorithms", "Technology"),
            ("Internet development history", "Technology"),
            ("Blockchain technology explained", "Technology"),
            ("Renewable energy sources", "Technology"),
            ("How do computers work?", "Technology"),
            ("Smartphone technology advances", "Technology"),
            ("Space exploration technology", "Technology"),
            ("Medical technology innovations", "Technology"),
            ("Robotics in manufacturing", "Technology"),
            
            # Arts
            ("Renaissance art movement", "Arts"),
            ("Classical music composers", "Arts"),
            ("Modern literature trends", "Arts"),
            ("Abstract painting techniques", "Arts"),
            ("Ballet performance styles", "Arts"),
            ("What is impressionism in art?", "Arts"),
            ("Contemporary sculpture", "Arts"),
            ("Jazz music origins", "Arts"),
            ("Theater history and development", "Arts"),
            ("Photography as artistic medium", "Arts"),
            
            # Sports
            ("Olympic Games history", "Sports"),
            ("Football World Cup records", "Sports"),
            ("Tennis grand slam tournaments", "Sports"),
            ("Basketball rules and gameplay", "Sports"),
            ("Swimming competitive events", "Sports"),
            ("How is soccer played?", "Sports"),
            ("Marathon running techniques", "Sports"),
            ("Baseball statistics and records", "Sports"),
            ("Winter Olympics events", "Sports"),
            ("Golf professional tours", "Sports"),
            
            # Politics
            ("Democracy principles and values", "Politics"),
            ("United Nations formation", "Politics"),
            ("Presidential election process", "Politics"),
            ("Constitution amendments", "Politics"),
            ("International diplomacy", "Politics"),
            ("What is federalism?", "Politics"),
            ("Political party systems", "Politics"),
            ("Voting rights movements", "Politics"),
            ("Government branches and functions", "Politics"),
            ("International law principles", "Politics"),
            
            # Geography
            ("Mountain formation processes", "Geography"),
            ("Ocean current patterns", "Geography"),
            ("Capital cities of Europe", "Geography"),
            ("Climate zones classification", "Geography"),
            ("Continental drift theory", "Geography"),
            ("Where is the Amazon rainforest?", "Geography"),
            ("Desert ecosystem characteristics", "Geography"),
            ("River systems worldwide", "Geography"),
            ("Volcanic activity causes", "Geography"),
            ("Population distribution patterns", "Geography"),
            
            # General
            ("General knowledge questions", "General"),
            ("Random facts and trivia", "General"),
            ("Miscellaneous information topics", "General"),
            ("Various subjects overview", "General"),
            ("Mixed topic discussions", "General")
        ]
        
        texts, labels = zip(*training_data)
        return list(texts), list(labels)
    
    def predict_intent_fallback(self, text: str) -> Tuple[str, float]:
        """
        Fallback intent prediction using keyword matching when TensorFlow model is not available
        """
        text_lower = text.lower()
        scores = {}
        
        # Calculate scores for each intent category
        for intent, keywords in self.fallback_rules.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight longer keywords more heavily
                    score += len(keyword.split()) * 0.3
            scores[intent] = score
        
        # Find the best match
        if scores and max(scores.values()) > 0:
            best_intent = max(scores, key=scores.get)
            confidence = min(scores[best_intent] / 3.0, 0.95)  # Cap confidence at 95%
        else:
            best_intent = "General"
            confidence = 0.5
        
        logger.info(f"Fallback prediction - Text: '{text}' -> Intent: {best_intent} (confidence: {confidence:.3f})")
        return best_intent, confidence
    
    def build_model(self) -> bool:
        """Build the neural network model"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot build model.")
            return False
        
        try:
            # Build bidirectional LSTM model
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_length),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, dropout=0.3, recurrent_dropout=0.3)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.intent_categories), activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info("Neural network model built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return False
    
    def train_model(self, texts: List[str], labels: List[str], epochs: int = 20) -> bool:
        """Train the intent classification model"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot train model.")
            return False
        
        try:
            # Prepare tokenizer
            tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
            tokenizer.fit_on_texts(texts)
            
            # Convert texts to sequences
            sequences = tokenizer.texts_to_sequences(texts)
            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                sequences, maxlen=self.max_sequence_length, truncating='post'
            )
            
            # Prepare labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                padded_sequences, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
            )
            
            # Build model if not already built
            if self.model is None:
                if not self.build_model():
                    return False
            
            # Train model
            logger.info("Starting model training...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
            
            # Store tokenizer and label encoder
            self.tokenizer = tokenizer
            self.label_encoder = label_encoder
            
            # Evaluate model
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
            logger.info(f"Training completed - Validation accuracy: {val_accuracy:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """Save the trained model and associated components"""
        if not TF_AVAILABLE or self.model is None:
            logger.warning("No model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(model_path, exist_ok=True)
            
            # Save model
            self.model.save(os.path.join(model_path, "model.h5"))
            
            # Save tokenizer
            with open(os.path.join(model_path, "tokenizer.pickle"), 'wb') as f:
                pickle.dump(self.tokenizer, f)
            
            # Save label encoder
            with open(os.path.join(model_path, "label_encoder.pickle"), 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model and associated components"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Using fallback prediction.")
            return False
        
        try:
            model_file = os.path.join(model_path, "model.h5")
            tokenizer_file = os.path.join(model_path, "tokenizer.pickle")
            label_encoder_file = os.path.join(model_path, "label_encoder.pickle")
            
            # Check if all files exist
            if not all(os.path.exists(f) for f in [model_file, tokenizer_file, label_encoder_file]):
                logger.warning(f"Model files not found in {model_path}. Using fallback prediction.")
                return False
            
            # Load model
            self.model = tf.keras.models.load_model(model_file)
            
            # Load tokenizer
            with open(tokenizer_file, 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load label encoder
            with open(label_encoder_file, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """
        Predict intent for given text
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted_intent, confidence_score)
        """
        if not text or len(text.strip()) == 0:
            return "General", 0.5
        
        # Use fallback if TensorFlow model is not available
        if not TF_AVAILABLE or self.model is None or self.tokenizer is None or self.label_encoder is None:
            return self.predict_intent_fallback(text)
        
        try:
            # Preprocess text
            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
                sequence, maxlen=self.max_sequence_length, truncating='post'
            )
            
            # Make prediction
            prediction = self.model.predict(padded_sequence, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            # Convert to intent label
            intent = self.label_encoder.inverse_transform([predicted_class])[0]
            
            logger.info(f"TF prediction - Text: '{text}' -> Intent: {intent} (confidence: {confidence:.3f})")
            return intent, confidence
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Fall back to rule-based prediction
            return self.predict_intent_fallback(text)
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            "tensorflow_available": TF_AVAILABLE,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "label_encoder_loaded": self.label_encoder is not None,
            "intent_categories": self.intent_categories,
            "max_sequence_length": self.max_sequence_length,
            "vocab_size": self.vocab_size,
            "fallback_available": True
        }

# Global instance
_global_classifier = None

def get_intent_classifier() -> IntentClassifier:
    """Get or create global intent classifier instance"""
    global _global_classifier
    
    if _global_classifier is None:
        _global_classifier = IntentClassifier()
    
    return _global_classifier
