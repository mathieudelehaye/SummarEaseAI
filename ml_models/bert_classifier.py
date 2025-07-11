"""
BERT Intent Classifier for SummarEaseAI
Uses PyTorch for inference (CPU version)
"""

import os
import logging
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

# Configure logging with UTF-8 encoding for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class IntentDataset(Dataset):
    """Dataset for intent classification"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
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
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier:
    """
    BERT intent classifier using PyTorch (CPU version)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize BERT classifier
        
        Args:
            model_path: Path to saved model directory
        """
        # Model paths - handle both absolute and relative paths
        if model_path:
            self.model_dir = Path(model_path)
        else:
            # Get the directory where this file is located
            current_file = Path(__file__).resolve()
            self.model_dir = current_file.parent / "bert_model"
        
        # Intent categories - matches categories in api.py
        self.intent_categories = [
            'History', 'Science', 'Technology', 'Arts', 'Sports', 'Politics'
        ]
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.metadata = None
        self.max_length = 128
        
        # Always use CPU
        self.device = torch.device("cpu")
        
        # Performance tracking
        self.inference_times = []
        
        logger.info("BERT Classifier initialized")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info("Using CPU for inference")

    def load_model(self) -> bool:
        """
        Load pre-trained BERT model and components
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Loading BERT model...")
            start_time = time.time()
            
            # Load metadata
            metadata_path = self.model_dir / "metadata.json"
            logger.info(f"Looking for metadata at: {metadata_path}")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.max_length = self.metadata.get('max_length', 128)
                    logger.info(f"Loaded metadata: {self.metadata}")
            
            # Load tokenizer first - look directly in model directory
            tokenizer_path = self.model_dir
            logger.info(f"Looking for tokenizer at: {tokenizer_path}")
            if tokenizer_path.exists():
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
                    logger.info("Tokenizer loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load tokenizer: {e}")
                    return False
            else:
                logger.error(f"Tokenizer not found at {tokenizer_path}")
                return False
            
            # Load label encoder
            label_encoder_path = self.model_dir / "label_encoder.pkl"
            logger.info(f"Looking for label encoder at: {label_encoder_path}")
            if label_encoder_path.exists():
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info(f"Label encoder loaded with classes: {self.label_encoder.classes_}")
            else:
                logger.info("Creating new label encoder with predefined categories")
                from sklearn.preprocessing import LabelEncoder
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(self.intent_categories)
                # Save the label encoder
                with open(label_encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                logger.info(f"Created and saved label encoder with classes: {self.label_encoder.classes_}")
            
            # Load model
            model_path = self.model_dir
            logger.info(f"Looking for model at: {model_path}")
            if model_path.exists():
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        str(model_path),
                        use_safetensors=True  # Use safetensors for secure loading
                    )
                    self.model = self.model.to(self.device)
                    self.model.eval()  # Set to evaluation mode
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    return False
            else:
                logger.error(f"Model not found at {model_path}")
                return False
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict intent for input text
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted_intent, confidence_score)
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Tokenize input
            logger.info(f"Tokenizing input text: {text[:50]}...")
            inputs = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            predictions = predictions.cpu().numpy()
            predicted_class_id = np.argmax(predictions, axis=1)[0]
            confidence = float(predictions[0][predicted_class_id])
            predicted_intent = self.label_encoder.inverse_transform([predicted_class_id])[0]
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            logger.info(f"Predicted intent: {predicted_intent} (confidence: {confidence:.3f})")
            
            # Return NO DETECTED if confidence is too low
            if confidence < 0.1:  # 10% confidence threshold
                logger.info("Confidence too low, returning NO DETECTED")
                return "NO DETECTED", confidence
                
            return predicted_intent, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "NO DETECTED", 0.0  # Return special category for errors

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict intents for a batch of texts"""
        if not self.is_loaded():
            raise ValueError("Model not loaded. Call load_model() first.")
            
        try:
            # Tokenize all texts
            inputs = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            predictions = predictions.cpu().numpy()
            predicted_classes = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)
            predicted_intents = self.label_encoder.inverse_transform(predicted_classes)
            
            return list(zip(predicted_intents, confidences.tolist()))
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return [("NO DETECTED", 0.0)] * len(texts)  # Return special category for errors

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return (self.model is not None and 
                self.tokenizer is not None and 
                self.label_encoder is not None)

    def get_performance_stats(self) -> Dict:
        """Get model performance statistics"""
        if not self.inference_times:
            return {'avg_inference_time': 0, 'total_predictions': 0}
            
        return {
            'avg_inference_time': sum(self.inference_times) / len(self.inference_times),
            'total_predictions': len(self.inference_times)
        }

def get_classifier(model_path: Optional[str] = None) -> Optional[BERTClassifier]:
    """
    Get initialized BERT classifier
    
    Args:
        model_path: Optional path to model directory
        
    Returns:
        Initialized classifier or None if initialization fails
    """
    try:
        classifier = BERTClassifier(model_path)
        if classifier.load_model():
            return classifier
    except Exception as e:
        logger.error(f"Error initializing classifier: {e}")
    return None

def classify_intent(text: str) -> Tuple[str, float]:
    """
    Classify intent for given text using global classifier
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (intent, confidence)
    """
    global_classifier = get_classifier()
    if global_classifier:
        return global_classifier.predict(text)
    return "NO DETECTED", 0.0  # Return special category when classifier not available 