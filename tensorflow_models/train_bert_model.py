"""
BERT Model Training Script for Intent Classification

This script fine-tunes a pre-trained BERT model on intent classification data.
It provides a complete training pipeline with evaluation and model saving.
"""

import os
import sys
import logging
from pathlib import Path
import tensorflow as tf

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tensorflow_models.bert_intent_classifier import BERTIntentClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check GPU availability and get user confirmation if no GPU detected"""
    print("🚀 SummarEaseAI GPU Detection for BERT Training")
    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")
    
    # Detect available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    gpu_count = len(gpus)
    
    print(f"🔍 GPU Detection Results:")
    if gpu_count > 0:
        print(f"✅ {gpu_count} GPU(s) detected!")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        print("🚀 BERT training will use GPU acceleration!")
        print("⚡ Expected training time: ~5-15 minutes")
        return True
    else:
        print("❌ No GPU detected!")
        print("⚠️ BERT training will run on CPU (this will be VERY slow)")
        print("⏰ Expected training time: ~30-90 minutes")
        print("\nPossible reasons for no GPU detection:")
        print("  - No compatible GPU installed") 
        print("  - GPU drivers not installed")
        print("  - DirectML plugin not properly configured")
        print("  - CUDA/ROCm not set up (for other GPU types)")
        
        # Prompt user for confirmation
        print("\n" + "="*50)
        print("⚠️ WARNING: BERT training without GPU is extremely slow!")
        response = input("❓ Do you want to continue BERT training WITHOUT GPU acceleration? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            print("✅ Continuing with CPU training (this will take a long time)...")
            return False
        else:
            print("❌ Training cancelled by user.")
            print("💡 To enable GPU acceleration:")
            print("   1. Ensure your GPU is compatible (DirectX 12)")  
            print("   2. Install tensorflow-directml-plugin")
            print("   3. Run: pip install tensorflow-directml-plugin")
            print("   4. Or try the regular neural network training instead")
            sys.exit(0)

def main():
    """Main training function"""
    print("🤗 BERT Intent Classification Training")
    print("=" * 50)
    
    # Check GPU availability first
    gpu_available = check_gpu_availability()
    
    print("\n" + "="*50)
    print("🧠 TRAINING CONFIGURATION")
    print("="*50)
    print(f"Hardware Acceleration: {'🚀 GPU' if gpu_available else '🐌 CPU Only'}")
    
    # Model selection
    available_models = {
        "1": ("bert-base-uncased", "Standard BERT (110MB, best quality)"),
        "2": ("distilbert-base-uncased", "DistilBERT (66MB, faster)"),
        "3": ("roberta-base", "RoBERTa (125MB, very high quality)")
    }
    
    print("\nAvailable BERT models:")
    for key, (model_name, description) in available_models.items():
        print(f"{key}. {model_name} - {description}")
    
    if not gpu_available:
        print("\n💡 Recommendation: Use DistilBERT (option 2) for CPU training")
    
    choice = input("\nSelect model (1-3, default=1): ").strip() or "1"
    
    if choice not in available_models:
        print("Invalid choice, using default BERT-base-uncased")
        choice = "1"
    
    model_name, description = available_models[choice]
    print(f"\nSelected: {model_name} - {description}")
    
    # Training configuration
    print("\nTraining Configuration:")
    epochs = input("Number of epochs (default=3): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 3
    
    print(f"Epochs: {epochs}")
    print(f"Model: {model_name}")
    print(f"Hardware: {'🚀 GPU Accelerated' if gpu_available else '🐌 CPU Only'}")
    
    # Confirm training
    confirm = input(f"\nStart training? This will download and fine-tune {model_name} (y/N): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Initialize classifier
    logger.info(f"Initializing BERT classifier with {model_name}")
    classifier = BERTIntentClassifier(model_name=model_name)
    
    # Prepare training data
    logger.info("Preparing training data...")
    texts, labels = classifier.prepare_training_data()
    
    print(f"\nTraining Data Statistics:")
    print(f"Total samples: {len(texts)}")
    print(f"Unique labels: {len(set(labels))}")
    print(f"Labels: {sorted(set(labels))}")
    
    # Start training
    print(f"\n🚀 Starting BERT fine-tuning...")
    if gpu_available:
        print(f"🚀 Using GPU acceleration - training should be fast!")
    else:
        print(f"🐌 Using CPU only - this will take a while, please be patient...")
    print(f"This may take {'10-30 minutes' if gpu_available else '30-90 minutes'} depending on your hardware.")
    print(f"GPU acceleration: {'✅ Available' if gpu_available else '❌ CPU only'}")
    
    success = classifier.train_model(texts, labels, epochs=epochs)
    
    if success:
        print("\n✅ Training completed successfully!")
        
        # Test the model
        print("\n🧪 Testing the trained model...")
        test_queries = [
            "Tell me about World War II",
            "How does photosynthesis work?",
            "Who was Albert Einstein?",
            "What is artificial intelligence?",
            "Olympic Games history"
        ]
        
        print("\nTest Results:")
        print("-" * 50)
        for query in test_queries:
            intent, confidence = classifier.predict_intent(query)
            print(f"Query: {query}")
            print(f"Intent: {intent} (confidence: {confidence:.1%})")
            print()
        
        # Model information
        model_info = classifier.get_model_info()
        print("📊 Model Information:")
        print(f"Model: {model_info['model_name']}")
        print(f"Type: {model_info['model_type']}")
        print(f"Hardware Used: {'🚀 GPU Accelerated' if gpu_available else '🐌 CPU Only'}")
        print(f"Categories: {len(model_info['intent_categories'])}")
        
        print("\n🎉 BERT model training completed successfully!")
        print("You can now use the BERT model in the SummarEaseAI application.")
        
    else:
        print("\n❌ Training failed. Please check the logs for errors.")
        print("Common issues:")
        print("- Insufficient memory (try distilbert-base-uncased)")
        print("- CUDA out of memory (reduce batch size)")
        print("- Network issues (check internet connection)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        print(f"\n❌ Training failed: {str(e)}")
        print("Please check the logs and try again.") 