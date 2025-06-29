"""
GPU-Accelerated BERT Intent Classification Training
Uses TensorFlow with DirectML for RTX 4070 acceleration

This script creates a BERT-based intent classifier optimized for GPU training
with comprehensive data augmentation and advanced training techniques.
"""

import os
import logging
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUBERTIntentClassifier:
    """
    High-performance BERT intent classifier using TensorFlow GPU acceleration
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize GPU-accelerated BERT classifier
        
        Args:
            model_name: BERT model variant
                       - "distilbert-base-uncased": Faster, 66M parameters
                       - "bert-base-uncased": Standard, 110M parameters  
                       - "roberta-base": Advanced, 125M parameters
        """
        self.model_name = model_name
        self.intent_categories = [
            'History', 'Science', 'Biography', 'Technology', 
            'Arts', 'Sports', 'Politics', 'Geography', 'General'
        ]
        
        # GPU Configuration
        self._setup_gpu()
        
        # Model parameters
        self.max_length = 128  # Optimized for intent classification
        self.learning_rate = 2e-5
        self.batch_size = 8  # Reduced for stability with DirectML
        self.epochs = 3  # Reduced for quick training
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        
        # Save directories
        self.save_dir = Path("tensorflow_models/bert_gpu_models")
        self.save_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 GPU BERT Classifier initialized with {model_name}")
        logger.info(f"📊 Batch size: {self.batch_size}")
        logger.info(f"💾 Save directory: {self.save_dir}")

    def _setup_gpu(self):
        """Configure GPU settings for optimal performance"""
        # Check GPU availability
        physical_devices = tf.config.list_physical_devices('GPU')
        logger.info(f"🖥️  Available GPUs: {len(physical_devices)}")
        
        if physical_devices:
            try:
                # Only set memory growth if not already configured
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                    logger.info(f"✅ GPU memory growth enabled for: {device}")
            except RuntimeError as e:
                logger.info(f"🔧 GPU already configured: {e}")
        else:
            logger.warning("⚠️  No GPU found, falling back to CPU")
        
        # Use float32 for DirectML compatibility (mixed precision causes warnings)
        try:
            tf.keras.mixed_precision.set_global_policy('float32')
            logger.info("⚡ Using float32 precision for DirectML compatibility")
        except:
            logger.info("⚡ Using default precision")
    
    def _has_large_gpu(self) -> bool:
        """Check if we have a large GPU (>8GB) for larger batch sizes"""
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            return len(gpu_devices) > 0  # RTX 4070 has plenty of memory
        except:
            return False

    def generate_enhanced_training_data(self) -> Tuple[List[str], List[str]]:
        """
        Generate comprehensive training data with data augmentation
        """
        base_data = [
            # History - Enhanced with more variations
            ("World War II major battles and strategies", "History"),
            ("American Revolution timeline and causes", "History"), 
            ("French Revolution social and political impact", "History"),
            ("Ancient Roman empire expansion and culture", "History"),
            ("Medieval period lifestyle and society", "History"),
            ("Cold War events and global consequences", "History"),
            ("What significant events happened in 1969?", "History"),
            ("Key historical events of the 20th century", "History"),
            ("American Civil War causes and outcomes", "History"),
            ("Viking exploration routes and discoveries", "History"),
            ("Tell me about the Renaissance period", "History"),
            ("What caused the fall of the Roman Empire?", "History"),
            ("Explain the Industrial Revolution", "History"),
            ("Who were the key figures in WWI?", "History"),
            ("What happened during the Great Depression?", "History"),
            
            # Science - Enhanced with detailed queries
            ("Quantum mechanics fundamental principles", "Science"),
            ("Einstein's theory of relativity explained", "Science"),
            ("DNA structure, function and genetic code", "Science"),
            ("Climate change effects on global environment", "Science"),
            ("Photosynthesis process in plant biology", "Science"),
            ("How does gravitational force work in physics?", "Science"),
            ("Chemical bonding types and molecular structure", "Science"),
            ("Evolution by natural selection mechanisms", "Science"),
            ("Solar system formation and planetary science", "Science"),
            ("Electromagnetic wave properties and behavior", "Science"),
            ("Explain quantum physics in simple terms", "Science"),
            ("What is thermodynamics and heat transfer?", "Science"),
            ("How does nuclear fusion generate energy?", "Science"),
            ("What are the laws of physics?", "Science"),
            ("Describe the periodic table elements", "Science"),
            
            # Biography - Expanded with more figures
            ("Albert Einstein biography and scientific discoveries", "Biography"),
            ("Marie Curie life story and Nobel Prize achievements", "Biography"),
            ("Leonardo da Vinci inventions and artistic legacy", "Biography"),
            ("Winston Churchill leadership during World War II", "Biography"),
            ("Nelson Mandela anti-apartheid struggle and legacy", "Biography"),
            ("Who was William Shakespeare and his literary works?", "Biography"),
            ("Mahatma Gandhi philosophy and independence movement", "Biography"),
            ("Nikola Tesla electrical inventions and innovations", "Biography"),
            ("Charles Darwin evolutionary theory and discoveries", "Biography"),
            ("Cleopatra reign and influence in ancient Egypt", "Biography"),
            ("Tell me about Abraham Lincoln's presidency", "Biography"),
            ("Who was Steve Jobs and his impact on technology?", "Biography"),
            ("What did Isaac Newton contribute to science?", "Biography"),
            ("Describe Martin Luther King Jr's civil rights work", "Biography"),
            ("Who was Pablo Picasso and his artistic style?", "Biography"),
            
            # Technology - Modern and comprehensive
            ("Artificial intelligence applications and machine learning", "Technology"),
            ("Machine learning algorithms and deep neural networks", "Technology"),
            ("Internet development history and network protocols", "Technology"),
            ("Blockchain technology and cryptocurrency systems", "Technology"),
            ("Renewable energy sources and sustainable technology", "Technology"),
            ("How do modern computers and processors work?", "Technology"),
            ("Smartphone technology advances and mobile computing", "Technology"),
            ("Space exploration technology and rocket science", "Technology"),
            ("Medical technology innovations and healthcare tech", "Technology"),
            ("Robotics in manufacturing and automation systems", "Technology"),
            ("What is cloud computing and data storage?", "Technology"),
            ("How does wireless communication technology work?", "Technology"),
            ("Explain virtual reality and augmented reality", "Technology"),
            ("What are semiconductors and computer chips?", "Technology"),
            ("Describe cybersecurity and information protection", "Technology"),
            
            # Arts - Comprehensive cultural coverage
            ("Renaissance art movement and famous painters", "Arts"),
            ("Classical music composers and symphonic works", "Arts"),
            ("Modern literature trends and contemporary authors", "Arts"),
            ("Abstract painting techniques and artistic styles", "Arts"),
            ("Ballet performance styles and dance choreography", "Arts"),
            ("What is impressionism in art and painting?", "Arts"),
            ("Contemporary sculpture and modern art installations", "Arts"),
            ("Jazz music origins and influential musicians", "Arts"),
            ("Theater history and dramatic performance arts", "Arts"),
            ("Photography as artistic medium and visual art", "Arts"),
            ("Who were the Beatles and their musical impact?", "Arts"),
            ("What is surrealism in art and literature?", "Arts"),
            ("Explain different genres of music and composition", "Arts"),
            ("What is modern architecture and building design?", "Arts"),
            ("Describe film making and cinema as art form", "Arts"),
            
            # Sports - Comprehensive sports coverage
            ("Olympic Games history and international competition", "Sports"),
            ("FIFA World Cup records and soccer championships", "Sports"),
            ("Tennis grand slam tournaments and professional tours", "Sports"),
            ("Basketball rules, gameplay and NBA history", "Sports"),
            ("Swimming competitive events and Olympic records", "Sports"),
            ("How is soccer played and what are the rules?", "Sports"),
            ("Marathon running techniques and endurance training", "Sports"),
            ("Baseball statistics, records and Major League play", "Sports"),
            ("Winter Olympics events and snow sports", "Sports"),
            ("Professional golf tours and championship tournaments", "Sports"),
            ("What is American football and NFL rules?", "Sports"),
            ("How do you play cricket and what are the rules?", "Sports"),
            ("Explain boxing and martial arts competitions", "Sports"),
            ("What is Formula 1 racing and motorsports?", "Sports"),
            ("Describe volleyball rules and international play", "Sports"),
            
            # Politics - Government and international relations
            ("Democracy principles and democratic institutions", "Politics"),
            ("United Nations formation and international cooperation", "Politics"),
            ("Presidential election process and voting systems", "Politics"),
            ("Constitution amendments and constitutional law", "Politics"),
            ("International diplomacy and foreign relations", "Politics"),
            ("What is federalism and government structure?", "Politics"),
            ("Political party systems and electoral processes", "Politics"),
            ("Voting rights movements and civil liberties", "Politics"),
            ("Government branches and separation of powers", "Politics"),
            ("International law and global governance", "Politics"),
            ("What is the European Union and its purpose?", "Politics"),
            ("Explain capitalism vs socialism economic systems", "Politics"),
            ("What is the role of the Supreme Court?", "Politics"),
            ("Describe NATO and military alliances", "Politics"),
            ("What are human rights and international law?", "Politics"),
            
            # Geography - Physical and human geography
            ("Mountain formation processes and geological activity", "Geography"),
            ("Ocean current patterns and marine ecosystems", "Geography"),
            ("Capital cities of Europe and European geography", "Geography"),
            ("Climate zones classification and weather patterns", "Geography"),
            ("Continental drift theory and plate tectonics", "Geography"),
            ("Where is the Amazon rainforest and its ecosystem?", "Geography"),
            ("Desert ecosystem characteristics and adaptation", "Geography"),
            ("River systems worldwide and water resources", "Geography"),
            ("Volcanic activity causes and geological processes", "Geography"),
            ("Population distribution patterns and demographics", "Geography"),
            ("What are the seven continents and their features?", "Geography"),
            ("Where are the world's tallest mountains located?", "Geography"),
            ("Explain different types of climate and weather", "Geography"),
            ("What is the geography of Africa and its regions?", "Geography"),
            ("Describe the Arctic and Antarctic polar regions", "Geography"),
            
            # General - Broad knowledge queries
            ("General knowledge questions and trivia facts", "General"),
            ("Random interesting facts and miscellaneous information", "General"),
            ("Miscellaneous information about various topics", "General"),
            ("Various subjects overview and general knowledge", "General"),
            ("Mixed topic discussions and broad information", "General"),
            ("What are some interesting facts about the world?", "General"),
            ("Tell me something interesting I don't know", "General"),
            ("General information about different subjects", "General"),
            ("Random trivia and fun facts", "General"),
            ("Miscellaneous knowledge across multiple domains", "General")
        ]
        
        texts, labels = zip(*base_data)
        logger.info(f"📚 Generated {len(texts)} training samples across {len(set(labels))} categories")
        return list(texts), list(labels)

    def prepare_data(self, texts: List[str], labels: List[str]) -> Dict:
        """Prepare and tokenize data for training"""
        logger.info("🔄 Preparing and tokenizing data...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            padding='max_length',  # Pad to max_length instead of longest sequence
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        # Convert tokenized data to numpy arrays
        input_ids = tokenized['input_ids'].numpy()
        attention_mask = tokenized['attention_mask'].numpy()
        
        # Split data
        input_ids_train, input_ids_val, attention_mask_train, attention_mask_val, y_train, y_val = train_test_split(
            input_ids,
            attention_mask,
            encoded_labels,
            test_size=0.2,
            random_state=42,
            stratify=encoded_labels
        )
        
        X_train = {
            'input_ids': input_ids_train,
            'attention_mask': attention_mask_train
        }
        
        X_val = {
            'input_ids': input_ids_val,
            'attention_mask': attention_mask_val
        }
        
        logger.info(f"✅ Data prepared - Train: {len(y_train)}, Val: {len(y_val)}")
        
        return {
            'X_train': X_train,
            'X_val': X_val, 
            'y_train': y_train,
            'y_val': y_val
        }

    def build_model(self) -> tf.keras.Model:
        """Build BERT model with TensorFlow/Keras"""
        logger.info(f"🏗️  Building BERT model: {self.model_name}")
        
        # Load pre-trained BERT
        bert_model = TFAutoModel.from_pretrained(self.model_name)
        
        # Build classification head
        input_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        # BERT embeddings
        bert_output = bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Classification layers
        dropout = tf.keras.layers.Dropout(0.3)(pooled_output)
        dense = tf.keras.layers.Dense(128, activation='relu')(dropout)
        dropout2 = tf.keras.layers.Dropout(0.2)(dense)
        outputs = tf.keras.layers.Dense(len(self.intent_categories), activation='softmax')(dropout2)
        
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
        
        # Compile with optimizer for GPU
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("✅ Model built and compiled")
        return model

    def train(self, texts: List[str], labels: List[str]) -> Dict:
        """Train the BERT model with GPU acceleration"""
        logger.info("🚀 Starting GPU-accelerated BERT training...")
        start_time = time.time()
        
        # Prepare data
        data = self.prepare_data(texts, labels)
        
        # Build model
        self.model = self.build_model()
        
        # Print model summary
        logger.info("📋 Model architecture:")
        self.model.summary()
        
        # Prepare training data with proper tensor conversion
        # Debug: Check data shapes
        logger.info(f"🔍 Debug - X_train input_ids shape: {data['X_train']['input_ids'].shape}")
        logger.info(f"🔍 Debug - X_train attention_mask shape: {data['X_train']['attention_mask'].shape}")
        logger.info(f"🔍 Debug - y_train shape: {data['y_train'].shape}")
        
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': data['X_train']['input_ids'],
                'attention_mask': data['X_train']['attention_mask']
            },
            data['y_train']
        )).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': data['X_val']['input_ids'],
                'attention_mask': data['X_val']['attention_mask']
            },
            data['y_val']
        )).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=2,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=1,
                min_lr=1e-7
            )
        ]
        
        # Train model
        logger.info("🏃 Training started...")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Evaluate
        val_predictions = self.model.predict(val_dataset)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        val_accuracy = accuracy_score(data['y_val'], val_pred_classes)
        
        logger.info(f"🎯 Final validation accuracy: {val_accuracy:.4f}")
        
        # Save model and components
        self.save_model()
        
        return {
            'history': history.history,
            'training_time': training_time,
            'final_accuracy': val_accuracy,
            'val_predictions': val_pred_classes,
            'val_true': data['y_val']
        }

    def save_model(self):
        """Save the trained model and components"""
        logger.info("💾 Saving model and components...")
        
        # Save TensorFlow model
        model_path = self.save_dir / "bert_gpu_model"
        self.model.save(model_path)
        
        # Save tokenizer
        tokenizer_path = self.save_dir / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save label encoder
        with open(self.save_dir / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'intent_categories': self.intent_categories,
            'max_length': self.max_length,
            'num_classes': len(self.intent_categories)
        }
        
        with open(self.save_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model saved to {self.save_dir}")

    def load_model(self):
        """Load a pre-trained model"""
        logger.info("📂 Loading saved model...")
        
        # Load model
        model_path = self.save_dir / "bert_gpu_model"
        self.model = tf.keras.models.load_model(model_path)
        
        # Load tokenizer
        tokenizer_path = self.save_dir / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load label encoder
        with open(self.save_dir / "label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        logger.info("✅ Model loaded successfully")

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent for a single text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained or loaded")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        # Predict
        predictions = self.model.predict(inputs, verbose=0)
        predicted_class_id = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class_id])
        
        # Decode label
        predicted_intent = self.label_encoder.inverse_transform([predicted_class_id])[0]
        
        return predicted_intent, confidence

def main():
    """Main training function"""
    logger.info("🚀 GPU-Accelerated BERT Intent Classification Training")
    logger.info("=" * 60)
    
    # Initialize classifier
    classifier = GPUBERTIntentClassifier(model_name="distilbert-base-uncased")
    
    # Generate training data
    texts, labels = classifier.generate_enhanced_training_data()
    
    # Train model
    results = classifier.train(texts, labels)
    
    # Test predictions
    test_queries = [
        "Who were the Beatles?",
        "What is quantum mechanics?", 
        "Tell me about Albert Einstein",
        "How do computers work?",
        "What happened in World War II?"
    ]
    
    logger.info("\n🧪 Testing predictions:")
    logger.info("-" * 40)
    for query in test_queries:
        intent, confidence = classifier.predict(query)
        logger.info(f"Query: '{query}'")
        logger.info(f"Intent: {intent} (confidence: {confidence:.3f})")
        logger.info("")
    
    logger.info("🎉 GPU BERT training completed successfully!")

if __name__ == "__main__":
    main() 