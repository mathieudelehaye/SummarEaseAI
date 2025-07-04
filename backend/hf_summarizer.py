"""
Hugging Face Transformers-based Summarization Module

This module provides local AI summarization using pre-trained transformer models
as an alternative to OpenAI API. It supports multiple models including BART, T5,
and Pegasus for different summarization tasks.
"""

import os
import logging
from typing import Optional, Dict, List
import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceSummarizer:
    """
    Local AI summarization using Hugging Face transformers
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summarizer with a pre-trained model
        
        Args:
            model_name: HuggingFace model name for summarization
                       Options: "facebook/bart-large-cnn", "t5-base", "google/pegasus-cnn_dailymail"
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.summarizer = None
        self.tokenizer = None
        self.model = None
        
        # Model configurations
        self.model_configs = {
            "facebook/bart-large-cnn": {
                "max_input_length": 1024,
                "max_output_length": 250,
                "min_output_length": 50,
                "description": "BART fine-tuned for CNN/DailyMail summarization"
            },
            "t5-base": {
                "max_input_length": 512,
                "max_output_length": 150,
                "min_output_length": 30,
                "description": "T5 base model for text summarization"
            },
            "google/pegasus-cnn_dailymail": {
                "max_input_length": 1024,
                "max_output_length": 200,
                "min_output_length": 40,
                "description": "Pegasus fine-tuned for news summarization"
            },
            "sshleifer/distilbart-cnn-12-6": {
                "max_input_length": 1024,
                "max_output_length": 200,
                "min_output_length": 40,
                "description": "Distilled BART for faster inference"
            }
        }
        
        logger.info(f"Initializing HuggingFace Summarizer with model: {model_name}")
        logger.info(f"Device: {self.device}")
        
    def load_model(self) -> bool:
        """Load the summarization model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Use pipeline for easier inference
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            return False
    
    def chunk_text(self, text: str, max_chunk_length: int = 1000) -> List[str]:
        """
        Split text into chunks for processing long articles
        
        Args:
            text: Input text to chunk
            max_chunk_length: Maximum length per chunk
            
        Returns:
            List of text chunks
        """
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize_text(
        self, 
        text: str, 
        max_length: Optional[int] = None, 
        min_length: Optional[int] = None,
        num_beams: int = 4,
        do_sample: bool = False
    ) -> str:
        """
        Summarize text using the loaded model
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            
        Returns:
            Generated summary
        """
        if not self.summarizer:
            if not self.load_model():
                return "Error: Could not load summarization model"
        
        try:
            config = self.model_configs.get(self.model_name, self.model_configs["facebook/bart-large-cnn"])
            
            # Set default lengths if not provided
            if max_length is None:
                max_length = config["max_output_length"]
            if min_length is None:
                min_length = config["min_output_length"]
            
            # Handle long texts by chunking
            if len(text) > config["max_input_length"]:
                logger.info("Text too long, chunking for processing...")
                chunks = self.chunk_text(text, config["max_input_length"])
                summaries = []
                
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    chunk_summary = self.summarizer(
                        chunk,
                        max_length=max_length // len(chunks) + 50,
                        min_length=min_length // len(chunks),
                        num_beams=num_beams,
                        do_sample=do_sample,
                        truncation=True
                    )[0]['summary_text']
                    summaries.append(chunk_summary)
                
                # Combine and re-summarize if needed
                combined_summary = " ".join(summaries)
                if len(combined_summary) > max_length * 2:
                    final_summary = self.summarizer(
                        combined_summary,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=num_beams,
                        do_sample=do_sample,
                        truncation=True
                    )[0]['summary_text']
                    return final_summary
                else:
                    return combined_summary
            else:
                # Process normally for shorter texts
                result = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    truncation=True
                )
                return result[0]['summary_text']
                
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def summarize_with_line_limit(self, text: str, max_lines: int = 30) -> str:
        """
        Summarize text with a specific line limit
        
        Args:
            text: Input text to summarize
            max_lines: Maximum number of lines in summary
            
        Returns:
            Generated summary with line limit
        """
        # Estimate tokens per line (approximately 15-20 tokens per line)
        estimated_tokens = max_lines * 18
        
        summary = self.summarize_text(
            text,
            max_length=min(estimated_tokens, 250),
            min_length=max(estimated_tokens // 3, 30)
        )
        
        # Post-process to ensure line limit
        lines = summary.split('. ')
        if len(lines) > max_lines:
            summary = '. '.join(lines[:max_lines]) + '.'
        
        return summary
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        config = self.model_configs.get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "device": self.device,
            "description": config.get("description", "Unknown model"),
            "max_input_length": config.get("max_input_length", "Unknown"),
            "max_output_length": config.get("max_output_length", "Unknown"),
            "cuda_available": torch.cuda.is_available(),
            "loaded": self.summarizer is not None
        }
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """Get available summarization models"""
        return {
            "facebook/bart-large-cnn": "BART CNN/DailyMail (Best quality)",
            "sshleifer/distilbart-cnn-12-6": "DistilBART (Faster)",
            "t5-base": "T5 Base (Balanced)",
            "google/pegasus-cnn_dailymail": "Pegasus (News focused)"
        }

# Global instance
_global_summarizer = None

def get_hf_summarizer(model_name: str = "facebook/bart-large-cnn") -> HuggingFaceSummarizer:
    """Get or create global HuggingFace summarizer instance"""
    global _global_summarizer
    
    if _global_summarizer is None or _global_summarizer.model_name != model_name:
        _global_summarizer = HuggingFaceSummarizer(model_name)
        
    return _global_summarizer

def summarize_with_huggingface(
    text: str, 
    max_lines: int = 30, 
    model_name: str = "facebook/bart-large-cnn"
) -> str:
    """
    Convenient function for summarizing text with HuggingFace models
    
    Args:
        text: Text to summarize
        max_lines: Maximum lines in summary
        model_name: HuggingFace model to use
        
    Returns:
        Generated summary
    """
    summarizer = get_hf_summarizer(model_name)
    return summarizer.summarize_with_line_limit(text, max_lines) 