"""
TensorFlow LSTM Intent Classifier
Handles basic text classification using LSTM
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntentClassifier:
    """LSTM-based intent classifier for text categorization."""

    def __init__(self, model_path=None):
        """Initialize the classifier with model parameters."""
        self.max_sequence_length = 100
        self.vocab_size = 10000
        self.model = None
        self.tokenizer = None
        self.label_encoder = None

        # If model path provided, load the saved model
        if model_path:
            self.load_model(model_path)

        logger.info("LSTM Intent Classifier initialized")

    def load_model(self, model_path):
        """Load the saved model and associated files."""
        logger.info(f"Loading model from {model_path}")

        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found at {model_path}")

        # Load tokenizer
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        with open(tokenizer_path) as f:
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

        # Load label encoder
        encoder_path = model_dir / "label_encoder.pkl"
        if not encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found at {encoder_path}")

        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                self.max_sequence_length = metadata.get("max_length", 100)
                self.vocab_size = metadata.get("vocab_size", 10000)

        # Load model
        model_path = model_dir / "lstm_model"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = tf.keras.models.load_model(str(model_path))
        logger.info("Model loaded successfully")

    def predict_intent(self, text: str) -> tuple:
        """Predict intent for given text."""
        try:
            if not self.model or not self.tokenizer or not self.label_encoder:
                raise ValueError("Model, tokenizer, or label encoder not loaded")

            # Tokenize and pad
            sequence = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=self.max_sequence_length)

            # Get prediction
            prediction = self.model.predict(padded, verbose=0)
            predicted_index = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_index])

            # Get predicted category
            predicted_category = self.label_encoder.inverse_transform(
                [predicted_index]
            )[0]

            # Return prediction info
            return {
                "intent": predicted_category,
                "confidence": confidence,
                "categories_available": self.label_encoder.classes_.tolist(),
                "model_type": "TensorFlow LSTM",
                "model_loaded": True,
                "text": text,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error predicting intent: {e}")
        return {
            "error": str(e),
            "model_loaded": False,
            "text": text,
            "timestamp": datetime.now().isoformat(),
        }


def get_classifier(model_path: str = None) -> IntentClassifier:
    """Get a loaded classifier instance"""
    try:
        classifier = IntentClassifier(model_path)
        return classifier
    except Exception as e:
        logger.error(f"Error initializing classifier: {e}")
        return None
