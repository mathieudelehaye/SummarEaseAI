"""
GPU-Optimized BERT Intent Classifier for SummarEaseAI
Production-ready inference module with RTX 4070 acceleration
"""

import os
import logging
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
from pathlib import Path
import json
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUBERTClassifier:
    """
    Production GPU-accelerated BERT intent classifier
    Optimized for fast inference on RTX 4070
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize GPU BERT classifier for inference
        
        Args:
            model_path: Path to saved model directory
        """
        # GPU Configuration
        self._setup_gpu()
        
        # Model paths
        self.model_dir = Path(model_path) if model_path else Path("tensorflow_models/bert_gpu_models")
        
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
        
        logger.info("🚀 GPU BERT Classifier initialized")
        logger.info(f"📁 Model directory: {self.model_dir}")

    def _setup_gpu(self):
        """Configure GPU for optimal inference performance"""
        # Enable GPU memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                    logger.info(f"✅ GPU memory growth enabled: {device}")
            except RuntimeError as e:
                logger.info(f"🔧 GPU already configured: {e}")
        else:
            logger.warning("⚠️  No GPU found, using CPU")
        
        # Mixed precision for faster inference
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("⚡ Mixed precision inference enabled")
        except:
            logger.info("⚡ Using float32 precision")

    def load_model(self) -> bool:
        """
        Load pre-trained BERT model and components with signature compatibility
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("📂 Loading GPU BERT model...")
            start_time = time.time()
            
            # Load metadata
            metadata_path = self.model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.max_length = self.metadata.get('max_length', 128)
                    logger.info(f"📋 Loaded metadata: {self.metadata}")
            
            # Load tokenizer first
            tokenizer_path = self.model_dir / "tokenizer"
            if tokenizer_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                logger.info("🔤 Tokenizer loaded")
            else:
                logger.error(f"❌ Tokenizer not found at {tokenizer_path}")
                return False
            
            # Load label encoder
            label_encoder_path = self.model_dir / "label_encoder.pkl"
            if label_encoder_path.exists():
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("🏷️  Label encoder loaded")
            else:
                logger.error(f"❌ Label encoder not found at {label_encoder_path}")
                return False
            
            # Load TensorFlow model with custom loading to handle signature mismatch
            model_path = self.model_dir / "bert_gpu_model"
            if model_path.exists():
                # Try loading with signature compatibility
                try:
                    # Load model with flexible signature
                    self.model = tf.saved_model.load(str(model_path))
                    
                    # Get the inference function
                    if hasattr(self.model, 'signatures'):
                        # Use default serving signature
                        self.inference_fn = self.model.signatures['serving_default']
                        logger.info("🤖 Model loaded with serving signature")
                    else:
                        # Fallback to direct callable
                        self.inference_fn = self.model
                        logger.info("🤖 Model loaded as callable")
                    
                except Exception as e:
                    logger.warning(f"⚠️  SavedModel loading failed: {e}")
                    # Fallback to Keras loading
                    self.model = tf.keras.models.load_model(model_path)
                    self.inference_fn = self.model
                    logger.info("🤖 Model loaded with Keras (fallback)")
                    
            else:
                logger.error(f"❌ Model not found at {model_path}")
                return False
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            
            # Warm up model with dummy prediction
            self._warmup_model()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

    def _warmup_model(self):
        """Warm up the GPU model for faster initial predictions"""
        try:
            logger.info("🔥 Warming up GPU model...")
            warmup_text = "This is a warmup prediction for GPU optimization"
            
            # Perform dummy prediction
            _ = self.predict(warmup_text)
            logger.info("✅ Model warmup completed")
            
        except Exception as e:
            logger.warning(f"⚠️  Warmup failed: {e}")

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict intent for input text with GPU acceleration
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted_intent, confidence_score)
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Tokenize input - match training format exactly
            inputs = self.tokenizer(
                text,
                padding='max_length',  # Force exact length padding like training
                truncation=True,
                max_length=self.max_length,
                return_tensors='tf'
            )
            
            # GPU prediction - use inference function
            with tf.device('/GPU:0'):
                if hasattr(self, 'inference_fn') and self.inference_fn is not None:
                    # Use serving signature or callable
                    try:
                        predictions = self.inference_fn(**inputs)
                        # Extract predictions from signature output
                        if isinstance(predictions, dict):
                            predictions = list(predictions.values())[0]
                        predictions = predictions.numpy()
                    except Exception as e:
                        logger.warning(f"⚠️  Signature inference failed: {e}, using direct model")
                        predictions = self.model.predict(inputs, verbose=0)
                else:
                    # Direct model prediction
                    predictions = self.model.predict(inputs, verbose=0)
            
            # Process results
            predicted_class_id = np.argmax(predictions, axis=1)[0]
            confidence = float(predictions[0][predicted_class_id])
            
            # Decode label
            predicted_intent = self.label_encoder.inverse_transform([predicted_class_id])[0]
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return predicted_intent, confidence
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            logger.error(f"❌ Error details: {type(e).__name__}: {str(e)}")
            return "General", 0.0

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict intents for multiple texts efficiently
        
        Args:
            texts: List of input texts
            
        Returns:
            List of (predicted_intent, confidence_score) tuples
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Tokenize all inputs - match training format exactly
            inputs = self.tokenizer(
                texts,
                padding='max_length',  # Force exact length padding like training
                truncation=True,
                max_length=self.max_length,
                return_tensors='tf'
            )
            
            # GPU batch prediction - use inference function
            with tf.device('/GPU:0'):
                if hasattr(self, 'inference_fn') and self.inference_fn is not None:
                    # Use serving signature or callable
                    try:
                        predictions = self.inference_fn(**inputs)
                        # Extract predictions from signature output
                        if isinstance(predictions, dict):
                            predictions = list(predictions.values())[0]
                        predictions = predictions.numpy()
                    except Exception as e:
                        logger.warning(f"⚠️  Signature inference failed: {e}, using direct model")
                        predictions = self.model.predict(inputs, verbose=0)
                else:
                    # Direct model prediction
                    predictions = self.model.predict(inputs, verbose=0)
            
            # Process results
            results = []
            for i, pred in enumerate(predictions):
                predicted_class_id = np.argmax(pred)
                confidence = float(pred[predicted_class_id])
                predicted_intent = self.label_encoder.inverse_transform([predicted_class_id])[0]
                results.append((predicted_intent, confidence))
            
            batch_time = time.time() - start_time
            logger.info(f"⚡ Batch prediction ({len(texts)} texts) completed in {batch_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch prediction error: {e}")
            logger.error(f"❌ Error details: {type(e).__name__}: {str(e)}")
            return [("General", 0.0)] * len(texts)

    def get_intent_with_details(self, text: str) -> Dict:
        """
        Get detailed intent prediction with confidence and performance metrics
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with detailed prediction information
        """
        start_time = time.time()
        intent, confidence = self.predict(text)
        prediction_time = time.time() - start_time
        
        return {
            'text': text,
            'predicted_intent': intent,
            'confidence': confidence,
            'prediction_time_ms': prediction_time * 1000,
            'model_type': 'GPU-BERT',
            'categories': self.intent_categories
        }

    def is_loaded(self) -> bool:
        """Check if model is fully loaded"""
        return all([
            self.model is not None,
            self.tokenizer is not None,
            self.label_encoder is not None
        ])

    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {"message": "No predictions made yet"}
        
        times_ms = [t * 1000 for t in self.inference_times]
        
        return {
            'total_predictions': len(self.inference_times),
            'avg_inference_time_ms': np.mean(times_ms),
            'min_inference_time_ms': np.min(times_ms),
            'max_inference_time_ms': np.max(times_ms),
            'std_inference_time_ms': np.std(times_ms),
            'predictions_per_second': 1 / np.mean(self.inference_times) if self.inference_times else 0
        }

    def benchmark(self, test_texts: Optional[List[str]] = None, num_runs: int = 100) -> Dict:
        """
        Benchmark GPU performance
        
        Args:
            test_texts: Optional list of test texts
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if test_texts is None:
            test_texts = [
                "Who were the Beatles and their musical impact?",
                "What is quantum mechanics in physics?",
                "Tell me about Albert Einstein's discoveries",
                "How do modern computers work?",
                "What happened during World War II?"
            ]
        
        logger.info(f"🏃 Starting GPU benchmark ({num_runs} runs)...")
        
        # Clear previous times
        self.inference_times = []
        
        # Benchmark single predictions
        single_times = []
        for i in range(num_runs):
            text = test_texts[i % len(test_texts)]
            start = time.time()
            _ = self.predict(text)
            single_times.append(time.time() - start)
        
        # Benchmark batch prediction
        batch_start = time.time()
        _ = self.predict_batch(test_texts * 10)  # 50 texts
        batch_time = time.time() - batch_start
        
        results = {
            'single_prediction': {
                'avg_time_ms': np.mean(single_times) * 1000,
                'min_time_ms': np.min(single_times) * 1000,
                'max_time_ms': np.max(single_times) * 1000,
                'predictions_per_second': 1 / np.mean(single_times)
            },
            'batch_prediction': {
                'batch_size': len(test_texts) * 10,
                'total_time_ms': batch_time * 1000,
                'avg_time_per_item_ms': (batch_time / (len(test_texts) * 10)) * 1000,
                'throughput_per_second': (len(test_texts) * 10) / batch_time
            },
            'gpu_info': self._get_gpu_info()
        }
        
        logger.info("✅ Benchmark completed")
        return results

    def _get_gpu_info(self) -> Dict:
        """Get GPU information"""
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            return {
                'gpu_count': len(gpu_devices),
                'gpu_names': [device.name for device in gpu_devices],
                'tensorflow_version': tf.__version__
            }
        except:
            return {'gpu_info': 'Not available'}

# Global instance for SummarEaseAI integration
_gpu_classifier = None

def get_gpu_classifier() -> GPUBERTClassifier:
    """Get global GPU classifier instance"""
    global _gpu_classifier
    if _gpu_classifier is None:
        _gpu_classifier = GPUBERTClassifier()
        if not _gpu_classifier.load_model():
            logger.warning("⚠️  Failed to load GPU model, falling back to CPU")
            return None
    return _gpu_classifier

def classify_intent_gpu(text: str) -> Tuple[str, float]:
    """
    Classify intent using GPU BERT model
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (intent, confidence)
    """
    classifier = get_gpu_classifier()
    if classifier:
        return classifier.predict(text)
    else:
        # Fallback to default classification
        return "General", 0.5

def test_gpu_classifier():
    """Test function for GPU classifier"""
    logger.info("🧪 Testing GPU BERT Classifier")
    logger.info("=" * 50)
    
    classifier = GPUBERTClassifier()
    
    if not classifier.load_model():
        logger.error("❌ Failed to load model")
        return
    
    # Test queries
    test_queries = [
        "Who were the Beatles?",
        "What is quantum mechanics?",
        "Tell me about Albert Einstein",
        "How do computers work?",
        "What happened in World War II?",
        "What are the Olympic Games?",
        "Explain democracy and voting",
        "Where is the Amazon rainforest?",
        "Random interesting facts"
    ]
    
    logger.info("🔍 Testing individual predictions:")
    for query in test_queries:
        result = classifier.get_intent_with_details(query)
        logger.info(f"Query: '{query}'")
        logger.info(f"Intent: {result['predicted_intent']} ({result['confidence']:.3f})")
        logger.info(f"Time: {result['prediction_time_ms']:.1f}ms")
        logger.info("")
    
    # Performance stats
    stats = classifier.get_performance_stats()
    logger.info("📊 Performance Statistics:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    # Benchmark
    benchmark_results = classifier.benchmark(test_queries[:5], num_runs=20)
    logger.info("\n🏃 Benchmark Results:")
    logger.info(f"Single prediction avg: {benchmark_results['single_prediction']['avg_time_ms']:.1f}ms")
    logger.info(f"Predictions/sec: {benchmark_results['single_prediction']['predictions_per_second']:.1f}")
    logger.info(f"Batch throughput: {benchmark_results['batch_prediction']['throughput_per_second']:.1f} items/sec")
    
    logger.info("✅ GPU classifier test completed!")

if __name__ == "__main__":
    test_gpu_classifier() 