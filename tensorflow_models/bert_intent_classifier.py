"""
BERT-based Intent Classification using Hugging Face Transformers

This module provides intent classification using pre-trained BERT models
as an alternative to the custom TensorFlow LSTM approach.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTIntentClassifier:
    """
    BERT-based intent classification using Hugging Face transformers
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize BERT intent classifier
        
        Args:
            model_name: HuggingFace BERT model name
                       Options: "bert-base-uncased", "distilbert-base-uncased", "roberta-base"
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.classifier_pipeline = None
        self.label_encoder = None
        self.intent_categories = [
            'History', 'Science', 'Biography', 'Technology', 
            'Arts', 'Sports', 'Politics', 'Geography', 'General'
        ]
        
        # Model configurations
        self.model_configs = {
            "bert-base-uncased": {
                "max_length": 512,
                "description": "Standard BERT model for classification",
                "performance": "High quality"
            },
            "distilbert-base-uncased": {
                "max_length": 512,
                "description": "Faster, smaller BERT variant",
                "performance": "Fast"
            },
            "roberta-base": {
                "max_length": 512,
                "description": "RoBERTa model with improved training",
                "performance": "Very high quality"
            }
        }
        
        self.save_dir = Path("tensorflow_models/bert_models")
        self.save_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initializing BERT Intent Classifier with model: {model_name}")
        logger.info(f"Device: {self.device}")
    
    def prepare_training_data(self) -> Tuple[List[str], List[str]]:
        """
        Generate comprehensive training data for intent classification
        """
        training_data = [
            # History
            ("World War II battles", "History"),
            ("American Revolution timeline", "History"),
            ("French Revolution causes", "History"),
            ("Ancient Rome empire", "History"),
            ("Medieval period life", "History"),
            ("Cold War events", "History"),
            ("What happened in 1969?", "History"),
            ("Historical events of the 20th century", "History"),
            ("Civil War consequences", "History"),
            ("Viking exploration", "History"),
            
            # Science
            ("Quantum mechanics principles", "Science"),
            ("Theory of relativity explained", "Science"),
            ("DNA structure and function", "Science"),
            ("Climate change effects", "Science"),
            ("Photosynthesis process", "Science"),
            ("How does gravity work?", "Science"),
            ("Chemical bonding types", "Science"),
            ("Evolution by natural selection", "Science"),
            ("Solar system formation", "Science"),
            ("Electromagnetic waves", "Science"),
            
            # Biography
            ("Albert Einstein biography", "Biography"),
            ("Marie Curie life story", "Biography"),
            ("Leonardo da Vinci achievements", "Biography"),
            ("Winston Churchill leadership", "Biography"),
            ("Nelson Mandela legacy", "Biography"),
            ("Who was Shakespeare?", "Biography"),
            ("Gandhi's philosophy", "Biography"),
            ("Tesla's inventions", "Biography"),
            ("Darwin's discoveries", "Biography"),
            ("Cleopatra's reign", "Biography"),
            
            # Technology
            ("Artificial intelligence applications", "Technology"),
            ("Machine learning algorithms", "Technology"),
            ("Internet development history", "Technology"),
            ("Blockchain technology", "Technology"),
            ("Renewable energy sources", "Technology"),
            ("How do computers work?", "Technology"),
            ("Smartphone technology", "Technology"),
            ("Space exploration technology", "Technology"),
            ("Medical technology advances", "Technology"),
            ("Robotics in manufacturing", "Technology"),
            
            # Arts
            ("Renaissance art movement", "Arts"),
            ("Classical music composers", "Arts"),
            ("Modern literature trends", "Arts"),
            ("Abstract painting techniques", "Arts"),
            ("Ballet performance styles", "Arts"),
            ("What is impressionism?", "Arts"),
            ("Contemporary sculpture", "Arts"),
            ("Jazz music origins", "Arts"),
            ("Theater history", "Arts"),
            ("Photography as art", "Arts"),
            
            # Sports
            ("Olympic Games history", "Sports"),
            ("Football World Cup records", "Sports"),
            ("Tennis grand slam tournaments", "Sports"),
            ("Basketball rules and gameplay", "Sports"),
            ("Swimming competitive events", "Sports"),
            ("How is soccer played?", "Sports"),
            ("Marathon running", "Sports"),
            ("Baseball statistics", "Sports"),
            ("Winter Olympics events", "Sports"),
            ("Golf professional tours", "Sports"),
            
            # Politics
            ("Democracy principles", "Politics"),
            ("United Nations formation", "Politics"),
            ("Presidential election process", "Politics"),
            ("Constitution amendments", "Politics"),
            ("Diplomacy in international relations", "Politics"),
            ("What is federalism?", "Politics"),
            ("Political party systems", "Politics"),
            ("Voting rights movements", "Politics"),
            ("Government branches", "Politics"),
            ("International law", "Politics"),
            
            # Geography
            ("Mountain formation processes", "Geography"),
            ("Ocean current patterns", "Geography"),
            ("Capital cities of Europe", "Geography"),
            ("Climate zones classification", "Geography"),
            ("Continental drift theory", "Geography"),
            ("Where is the Amazon rainforest?", "Geography"),
            ("Desert ecosystem features", "Geography"),
            ("River systems worldwide", "Geography"),
            ("Volcanic activity causes", "Geography"),
            ("Population distribution patterns", "Geography"),
            
            # General
            ("General knowledge questions", "General"),
            ("Random facts and trivia", "General"),
            ("Miscellaneous information", "General"),
            ("Various topics overview", "General"),
            ("Mixed subject matter", "General")
        ]
        
        texts, labels = zip(*training_data)
        return list(texts), list(labels)
    
    def load_tokenizer_and_model(self, num_labels: int = None) -> bool:
        """Load tokenizer and model"""
        try:
            logger.info(f"Loading tokenizer and model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if num_labels:
                # For training - create new model with specific number of labels
                config = AutoConfig.from_pretrained(
                    self.model_name,
                    num_labels=num_labels,
                    problem_type="single_label_classification"
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    config=config
                )
            else:
                # For inference - try to load fine-tuned model first
                model_path = self.save_dir / f"{self.model_name.replace('/', '_')}_finetuned"
                if model_path.exists():
                    logger.info(f"Loading fine-tuned model from {model_path}")
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                else:
                    # Use pre-trained model
                    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            logger.info("Model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def create_classifier_pipeline(self) -> bool:
        """Create classification pipeline"""
        try:
            if not self.model or not self.tokenizer:
                if not self.load_tokenizer_and_model():
                    return False
            
            self.classifier_pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            logger.info("Classification pipeline created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating pipeline: {str(e)}")
            return False
    
    def train_model(self, texts: List[str], labels: List[str], epochs: int = 3) -> bool:
        """
        Fine-tune BERT model for intent classification
        
        Args:
            texts: Training texts
            labels: Training labels
            epochs: Number of training epochs
            
        Returns:
            Success status
        """
        try:
            logger.info("Starting BERT model fine-tuning...")
            
            # Prepare label encoder
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            num_labels = len(self.label_encoder.classes_)
            
            # Load model with correct number of labels
            if not self.load_tokenizer_and_model(num_labels):
                return False
            
            # Prepare dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=self.model_configs[self.model_name]["max_length"]
                )
            
            # Split data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
            )
            
            # Create datasets
            train_dataset = Dataset.from_dict({
                "text": train_texts,
                "labels": train_labels
            })
            
            val_dataset = Dataset.from_dict({
                "text": val_texts,
                "labels": val_labels
            })
            
            # Tokenize datasets
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            val_dataset = val_dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.save_dir / "training_output"),
                num_train_epochs=epochs,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=str(self.save_dir / "logs"),
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_accuracy"
            )
            
            # Metrics function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                return {"accuracy": accuracy_score(labels, predictions)}
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )
            
            # Train model
            logger.info("Starting training...")
            trainer.train()
            
            # Save fine-tuned model
            model_save_path = self.save_dir / f"{self.model_name.replace('/', '_')}_finetuned"
            trainer.save_model(str(model_save_path))
            self.tokenizer.save_pretrained(str(model_save_path))
            
            # Save label encoder
            label_encoder_path = self.save_dir / f"{self.model_name.replace('/', '_')}_label_encoder.pkl"
            with open(label_encoder_path, "wb") as f:
                pickle.dump(self.label_encoder, f)
            
            logger.info(f"Model fine-tuning completed and saved to {model_save_path}")
            
            # Create pipeline for inference
            self.create_classifier_pipeline()
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """
        Predict intent for given text
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (predicted_intent, confidence)
        """
        if not self.classifier_pipeline:
            # Try to load existing model
            if not self.create_classifier_pipeline():
                return "General", 0.0
        
        if not self.label_encoder:
            # Try to load label encoder
            label_encoder_path = self.save_dir / f"{self.model_name.replace('/', '_')}_label_encoder.pkl"
            if label_encoder_path.exists():
                with open(label_encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
        
        try:
            # Get predictions
            results = self.classifier_pipeline(text)
            
            if not results or not isinstance(results, list) or not results[0]:
                return "General", 0.0
            
            # Find highest scoring prediction
            best_result = max(results[0], key=lambda x: x['score'])
            
            # Convert label back to intent if we have label encoder
            if self.label_encoder:
                try:
                    label_idx = int(best_result['label'].split('_')[-1])
                    intent = self.label_encoder.inverse_transform([label_idx])[0]
                except:
                    intent = "General"
            else:
                intent = "General"
            
            confidence = best_result['score']
            
            logger.info(f"BERT predicted intent: {intent} (confidence: {confidence:.3f})")
            return intent, confidence
            
        except Exception as e:
            logger.error(f"Error predicting intent: {str(e)}")
            return "General", 0.0
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        config = self.model_configs.get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "model_type": "BERT-based",
            "device": self.device,
            "description": config.get("description", "Unknown model"),
            "max_length": config.get("max_length", "Unknown"),
            "cuda_available": torch.cuda.is_available(),
            "model_loaded": self.model is not None,
            "pipeline_ready": self.classifier_pipeline is not None,
            "intent_categories": self.intent_categories
        }

# Global instance
_global_bert_classifier = None

def get_bert_intent_classifier(model_name: str = "bert-base-uncased") -> BERTIntentClassifier:
    """Get or create global BERT intent classifier instance"""
    global _global_bert_classifier
    
    if _global_bert_classifier is None or _global_bert_classifier.model_name != model_name:
        _global_bert_classifier = BERTIntentClassifier(model_name)
    
    return _global_bert_classifier

def predict_intent_with_bert(text: str, model_name: str = "bert-base-uncased") -> Tuple[str, float]:
    """
    Convenient function for intent prediction with BERT
    
    Args:
        text: Input text
        model_name: BERT model to use
        
    Returns:
        Tuple of (predicted_intent, confidence)
    """
    classifier = get_bert_intent_classifier(model_name)
    return classifier.predict_intent(text) 