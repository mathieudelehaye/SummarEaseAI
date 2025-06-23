#!/usr/bin/env python3
"""
SummarEaseAI - Intent Classification Model Training Script

This script trains a TensorFlow neural network to classify user intents
for the SummarEaseAI chatbot and provides model evaluation metrics.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow_models.intent_classifier import IntentClassifier

def setup_gpu():
    """Configure GPU settings for optimal performance"""
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found, training will use CPU")

def plot_training_history(history):
    """Plot training history with accuracy and loss curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('tensorflow_models/training_history.png')
    plt.show()
    
    # Print final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    return final_val_acc

def test_model_predictions(classifier):
    """Test the trained model with sample predictions"""
    test_queries = [
        "Tell me about World War I",
        "How does photosynthesis work?",
        "Who was Albert Einstein?",
        "What is artificial intelligence?",
        "Olympic Games history",
        "Democracy principles",
        "Mountain formation",
        "Renaissance paintings",
        "What happened on July 20, 1969?",
        "Explain quantum mechanics",
        "Biography of Marie Curie",
        "Latest technology trends"
    ]
    
    print("\nTesting model predictions:")
    print("=" * 50)
    
    predictions = []
    for query in test_queries:
        intent, confidence = classifier.predict_intent(query)
        predictions.append({
            'Query': query, 
            'Predicted Intent': intent, 
            'Confidence': f"{confidence:.3f}"
        })
        print(f"Query: '{query}'")
        print(f"Predicted Intent: {intent} (Confidence: {confidence:.3f})")
        print("-" * 30)
    
    # Create DataFrame for better visualization
    predictions_df = pd.DataFrame(predictions)
    print("\nPredictions Summary:")
    print(predictions_df.to_string(index=False))
    
    return predictions_df

def main():
    """Main training function"""
    print("🚀 SummarEaseAI Intent Classification Model Training")
    print("=" * 60)
    
    # Setup GPU
    setup_gpu()
    
    # Initialize the intent classifier
    classifier = IntentClassifier(vocab_size=10000, max_length=100, embedding_dim=128)
    
    # Generate training data
    print("\n📊 Preparing training data...")
    texts, labels = classifier.prepare_training_data()
    
    print(f"Training data prepared:")
    print(f"Number of samples: {len(texts)}")
    print(f"Number of unique intents: {len(set(labels))}")
    print(f"Intent categories: {sorted(set(labels))}")
    
    # Visualize label distribution
    plt.figure(figsize=(12, 6))
    pd.Series(labels).value_counts().plot(kind='bar')
    plt.title('Distribution of Intent Categories')
    plt.xlabel('Intent Category')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tensorflow_models/label_distribution.png')
    plt.show()
    
    # Train the model
    print("\n🧠 Starting model training...")
    history = classifier.train_model(texts, labels, epochs=20, validation_split=0.2)
    print("Training completed!")
    
    # Plot training history
    print("\n📈 Plotting training history...")
    final_val_acc = plot_training_history(history)
    
    # Test the model
    print("\n🔍 Testing model predictions...")
    predictions_df = test_model_predictions(classifier)
    
    # Save the trained model
    print("\n💾 Saving trained model...")
    model_save_path = "tensorflow_models/saved_model"
    classifier.save_model(model_save_path)
    print(f"Model saved successfully to: {model_save_path}")
    
    # Test loading the model
    print("\n🔄 Testing model loading...")
    test_classifier = IntentClassifier()
    if test_classifier.load_model(model_save_path):
        print("Model loaded successfully!")
        
        # Test a prediction with the loaded model
        test_query = "Tell me about quantum physics"
        intent, confidence = test_classifier.predict_intent(test_query)
        print(f"Test prediction with loaded model:")
        print(f"Query: '{test_query}'")
        print(f"Predicted Intent: {intent} (Confidence: {confidence:.3f})")
    else:
        print("Failed to load model!")
    
    # Model evaluation summary
    print("\n" + "="*60)
    print("🎯 MODEL TRAINING SUMMARY")
    print("="*60)
    print(f"Architecture: Bidirectional LSTM with Embedding")
    print(f"Vocabulary Size: {classifier.vocab_size}")
    print(f"Max Sequence Length: {classifier.max_length}")
    print(f"Embedding Dimension: {classifier.embedding_dim}")
    print(f"Number of Intent Categories: {len(classifier.intent_categories)}")
    print(f"Training Samples: {len(texts)}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    # Calculate model size
    model_path = os.path.join(model_save_path, 'intent_model.h5')
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / 1024 / 1024
        print(f"Model Size: {model_size:.2f} MB")
    
    print("\n✅ Model ready for deployment! 🚀")
    print("You can now use the trained model in your SummarEaseAI chatbot.")

if __name__ == "__main__":
    main() 