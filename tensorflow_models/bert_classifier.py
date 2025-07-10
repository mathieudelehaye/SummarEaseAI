"""
BERT Intent Classifier for SummarEaseAI
Supports GPU training and CPU inference
"""

import os
import logging
import time
from typing import List, Dict, Tuple, Optional
import numpy as np

# Suppress TensorFlow logging before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
# Additional TensorFlow logging suppression
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from transformers import AutoTokenizer
from pathlib import Path
import json
import pickle

# Configure logging with UTF-8 encoding for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class BERTClassifier:
    """
    BERT intent classifier
    - Supports GPU training for faster model updates
    - Uses CPU for inference for consistent deployment
    """
    
    def __init__(self, model_path: Optional[str] = None, use_gpu_for_training: bool = True):
        """
        Initialize BERT classifier
        
        Args:
            model_path: Path to saved model directory
            use_gpu_for_training: Whether to use GPU for training (if available)
        """
        self.use_gpu_for_training = use_gpu_for_training
        
        # Model paths - handle both absolute and relative paths
        if model_path:
            self.model_dir = Path(model_path)
        else:
            # Get the directory where this file is located
            current_file = Path(__file__).resolve()
            self.model_dir = current_file.parent / "bert_gpu_models"
        
        # Intent categories
        self.intent_categories = [
            'History', 'Science', 'Biography', 'Technology', 
            'Arts', 'Sports', 'Politics', 'Geography', 'General'
        ]
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.metadata = None
        self.max_length = 128
        
        # Performance tracking
        self.inference_times = []
        
        logger.info("BERT Classifier initialized")
        logger.info(f"Model directory: {self.model_dir}")
        
        # Log GPU availability for training
        gpus = tf.config.list_physical_devices('GPU')
        if self.use_gpu_for_training and gpus:
            logger.info(f"Found {len(gpus)} GPU(s) available for training")
        else:
            logger.info("Training will use CPU")

    def load_model(self) -> bool:
        """
        Load pre-trained BERT model and components
        Forces CPU for inference regardless of GPU availability
        
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
            
            # Load tokenizer first
            tokenizer_path = self.model_dir / "tokenizer"
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
                logger.error(f"Label encoder not found at {label_encoder_path}")
                return False
            
            # Force CPU for model loading and inference
            with tf.device('/CPU:0'):
                model_path = self.model_dir / "bert_gpu_model"
                logger.info(f"Looking for model at: {model_path}")
                if model_path.exists():
                    try:
                        # Try Keras loading first
                        logger.info("Attempting to load with Keras...")
                        self.model = tf.keras.models.load_model(str(model_path))
                        self.inference_fn = self.model
                        logger.info("Model loaded with Keras successfully")
                    except Exception as e:
                        logger.warning(f"Keras loading failed: {e}, trying SavedModel")
                        try:
                            # Fallback to SavedModel loading
                            logger.info("Attempting to load with SavedModel...")
                            self.model = tf.saved_model.load(str(model_path))
                            if hasattr(self.model, 'signatures'):
                                self.inference_fn = self.model.signatures['serving_default']
                                logger.info("Model loaded with serving signature")
                            else:
                                self.inference_fn = self.model
                                logger.info("Model loaded as callable")
                        except Exception as e2:
                            logger.error(f"All model loading attempts failed: {e2}")
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
        Predict intent for input text (CPU-only inference)
        
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
                return_tensors='tf'
            )
            logger.info(f"Tokenized input shape: {inputs['input_ids'].shape}")
            
            # Force CPU for inference
            with tf.device('/CPU:0'):
                if hasattr(self, 'inference_fn') and self.inference_fn is not None:
                    try:
                        logger.info("Using inference function...")
                        predictions = self.inference_fn(**inputs)
                        if isinstance(predictions, dict):
                            logger.info(f"Predictions is a dict with keys: {list(predictions.keys())}")
                            predictions = list(predictions.values())[0]
                        predictions = predictions.numpy()
                        logger.info(f"Raw predictions shape: {predictions.shape}")
                    except Exception as e:
                        logger.warning(f"Signature inference failed: {e}, using direct model")
                        predictions = self.model.predict(inputs, verbose=0)
                else:
                    logger.info("Using direct model prediction...")
                    predictions = self.model.predict(inputs, verbose=0)
            
            # Process results
            logger.info(f"Processing predictions: {predictions}")
            predicted_class_id = np.argmax(predictions, axis=1)[0]
            logger.info(f"Predicted class ID: {predicted_class_id}")
            confidence = float(predictions[0][predicted_class_id])
            logger.info(f"Confidence: {confidence}")
            predicted_intent = self.label_encoder.inverse_transform([predicted_class_id])[0]
            logger.info(f"Predicted intent: {predicted_intent}")
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return predicted_intent, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            return "General", 0.0

    def train(self, train_texts: List[str], train_labels: List[str], **kwargs) -> Dict:
        """
        Train or fine-tune the model (uses GPU if available and enabled)
        
        Args:
            train_texts: List of training texts
            train_labels: List of corresponding labels
            **kwargs: Additional training arguments (epochs, batch_size, etc.)
            
        Returns:
            Training history
        """
        if self.use_gpu_for_training and tf.config.list_physical_devices('GPU'):
            logger.info("Training on GPU")
            # Training code will automatically use GPU
        else:
            logger.info("Training on CPU")
            # Force CPU if GPU training is disabled
            with tf.device('/CPU:0'):
                pass  # Training code here
        
        # TODO: Implement actual training logic
        return {"message": "Training not implemented yet"}

    def save_model(self, save_path: Optional[str] = None) -> bool:
        """
        Save the model and all necessary components
        
        Args:
            save_path: Optional path to save the model (uses model_dir if not provided)
            
        Returns:
            bool: True if successful
        """
        save_dir = Path(save_path) if save_path else self.model_dir
        try:
            # Save model
            model_path = save_dir / "bert_model"
            self.model.save(str(model_path))
            
            # Save tokenizer
            tokenizer_path = save_dir / "tokenizer"
            self.tokenizer.save_pretrained(str(tokenizer_path))
            
            # Save label encoder
            with open(save_dir / "label_encoder.pkl", 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Save metadata
            metadata = {
                'max_length': self.max_length,
                'intent_categories': self.intent_categories,
                'version': '1.0'
            }
            with open(save_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f)
                
            logger.info(f"Model saved to {save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict intents for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of (predicted_intent, confidence_score) tuples
        """
        return [self.predict(text) for text in texts]

    def get_intent_with_details(self, text: str) -> Dict:
        """
        Get detailed prediction results
        
        Args:
            text: Input text
            
        Returns:
            Dict with prediction details
        """
        intent, confidence = self.predict(text)
        return {
            'text': text,
            'intent': intent,
            'confidence': confidence,
            'model_type': 'BERT',
            'inference_time': self.inference_times[-1] if self.inference_times else None
        }

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return (self.model is not None and 
                self.tokenizer is not None and 
                self.label_encoder is not None)

    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {}
            
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_predictions': len(self.inference_times)
        }

def get_classifier(model_path: Optional[str] = None) -> Optional[BERTClassifier]:
    """
    Get initialized BERT classifier
    
    Args:
        model_path: Optional path to model directory
        
    Returns:
        Initialized classifier or None if loading fails
    """
    try:
        classifier = BERTClassifier(model_path)
        if classifier.load_model():
            return classifier
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
    return None

def classify_intent(text: str) -> Tuple[str, float]:
    """
    Convenience function for single text classification
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (predicted_intent, confidence_score)
    """
    classifier = get_classifier()
    if classifier:
        return classifier.predict(text)
    return "General", 0.0 