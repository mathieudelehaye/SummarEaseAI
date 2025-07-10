"""
Train LSTM on Wikipedia data
Uses DirectML GPU acceleration
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wikipedia
import re
import random # Added for shuffling articles

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR

# Import TF first
import tensorflow as tf

# Configure DirectML for GPU support
try:
    import tensorflow_directml
    physical_devices = tf.config.list_physical_devices()
    dml_visible_devices = tf.config.list_physical_devices('DML')
    if dml_visible_devices:
        logger.info(f"DirectML devices found: {len(dml_visible_devices)}")
        for device in dml_visible_devices:
            logger.info(f"  {device.name}")
    else:
        logger.warning("No DirectML devices found. Running on CPU only.")
    
    logger.info(f"All available devices:")
    for device in physical_devices:
        logger.info(f"  {device.name} ({device.device_type})")
except ImportError:
    logger.warning("tensorflow-directml-plugin not found. Install with: pip install tensorflow-directml-plugin")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# Training parameters (matching train_lstm.py exactly)
MAX_WORDS = 10000  # Maximum vocabulary size
MAX_LEN = 100     # Maximum sequence length
EMBED_DIM = 100   # Word embedding dimension
LSTM_UNITS = 100  # LSTM layer units
EPOCHS = 10
BATCH_SIZE = 32

def clean_text(text):
    """Clean and normalize text for model training."""
    if pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace numbers with <NUM> token to preserve numeric patterns
    text = re.sub(r'\d+(\.\d+)?', ' <NUM> ', text)
    
    # Keep certain special characters that might be meaningful
    text = re.sub(r'[^\w\s\$\%\-\.]', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def check_gpu_support():
    """Check and log GPU/DirectML support status"""
    logger.info("Checking GPU support...")
    
    # Check TensorFlow version
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Check if CUDA is available (for systems with NVIDIA GPUs)
    cuda_available = len(tf.config.list_physical_devices('GPU')) > 0
    if cuda_available:
        logger.info("CUDA GPU devices found:")
        for device in tf.config.list_physical_devices('GPU'):
            logger.info(f"  {device.name}")
    
    # Check DirectML support
    dml_available = len(tf.config.list_physical_devices('DML')) > 0
    if dml_available:
        logger.info("DirectML devices found:")
        for device in tf.config.list_physical_devices('DML'):
            logger.info(f"  {device.name}")
    
    if not (cuda_available or dml_available):
        logger.warning("No GPU acceleration available. Running on CPU only.")
    
    return cuda_available or dml_available

def create_model(vocab_size, num_classes):
    """Create LSTM model with improved architecture."""
    model = Sequential([
        # Embedding layer with more dimensions
        Embedding(vocab_size, EMBED_DIM * 2, input_length=MAX_LEN),
        
        # First LSTM layer with batch normalization
        SpatialDropout1D(0.3),
        LSTM(LSTM_UNITS * 2, return_sequences=True, 
             dropout=0.3, recurrent_dropout=0.3),
        tf.keras.layers.BatchNormalization(),
        
        # Second LSTM layer
        LSTM(LSTM_UNITS, dropout=0.3, recurrent_dropout=0.3),
        tf.keras.layers.BatchNormalization(),
        
        # Dense layers with regularization
        Dense(LSTM_UNITS, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.4),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Use a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Model architecture:")
    model.summary(print_fn=logger.info)
    
    return model

def fetch_music_articles(num_articles=100):
    """Fetch music-related articles from Wikipedia."""
    logger.info("Fetching music-related articles from Wikipedia...")
    
    # Load existing data if available to check for duplicates
    data_file = Path(__file__).parent / "training_data" / "enhanced_wikipedia_training_data.csv"
    existing_titles = set()
    if data_file.exists():
        existing_df = pd.read_csv(data_file)
        existing_titles = set(existing_df['title'].str.lower())
    
    articles = []
    categories = [
        "Musical_instruments",  # Fixed category name
        "Music_genres",
        "Music_theory",
        "Musical_composition",
        "Music_industry",
        "Music_technology",
        "Music_software",
        "Musicians",
        "Musical_groups",
        "Record_labels",
        "Music_festivals",
        "Music_awards",
        "Music_venues",
        "Music_education",
        "Music_organizations",
        "Music_terminology",
        "Music_history",
        "Popular_music",
        "Classical_music",
        "Jazz",
        "Rock_music",
        "Electronic_music",
        "Folk_music",
        "World_music"
    ]
    
    for category in categories:
        if len(articles) >= num_articles:
            break
            
        try:
            # Get articles from category
            category_articles = wikipedia.page(category).links
            
            # Shuffle to get random articles from category
            random.shuffle(category_articles)
            
            for title in category_articles:
                if len(articles) >= num_articles:
                    break
                    
                # Skip if already downloaded
                if title.lower() in existing_titles:
                    logger.debug(f"Skipping existing article: {title}")
                    continue
                    
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    # Skip disambiguation pages
                    if "may refer to" in page.content[:100].lower():
                        continue
                        
                    article = {
                        "text": page.content,
                        "intent": "Music",
                        "title": page.title,
                        "text_clean": clean_text(page.content)
                    }
                    articles.append(article)
                    existing_titles.add(title.lower())
                    logger.info(f"Added music article: {title}")
                    
                except (wikipedia.exceptions.DisambiguationError,
                        wikipedia.exceptions.PageError,
                        wikipedia.exceptions.HTTPTimeoutError) as e:
                    logger.debug(f"Skipping article {title}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching music articles for category {category}: {str(e)}")
            continue
            
    logger.info(f"Fetched {len(articles)} music-related articles")
    return articles

def fetch_finance_articles(num_articles=100):
    """Fetch finance-related articles from Wikipedia."""
    logger.info("Fetching finance-related articles from Wikipedia...")
    
    # Load existing data if available to check for duplicates
    data_file = Path(__file__).parent / "training_data" / "enhanced_wikipedia_training_data.csv"
    existing_titles = set()
    if data_file.exists():
        existing_df = pd.read_csv(data_file)
        existing_titles = set(existing_df['title'].str.lower())
    
    articles = []
    # More specific categories that are less likely to have deep interconnections
    categories = [
        "Stock_exchanges",  # More specific than "Stock exchange"
        "Investment_banks",  # Specific companies
        "Financial_ratios",  # Specific concepts
        "Financial_regulation",  # Regulatory aspects
        "Corporate_finance",  # Business focus
        "Financial_economics",  # Academic focus
        "Derivatives_(finance)",  # Specific instruments
        "Hedge_funds",  # Specific institutions
        "Mutual_funds",  # Investment vehicles
        "Financial_risk",  # Risk management
        "Financial_analysts",  # Professionals
        "Financial_markets_by_country"  # Geographic focus
    ]
    
    for category in categories:
        if len(articles) >= num_articles:
            break
            
        try:
            # Get articles from category
            category_articles = wikipedia.page(category).links[:20]  # Limit to first 20 links per category
            
            # Shuffle to get random articles from category
            random.shuffle(category_articles)
            
            for title in category_articles[:5]:  # Take max 5 articles per category
                if len(articles) >= num_articles:
                    break
                    
                # Skip if already downloaded
                if title.lower() in existing_titles:
                    logger.debug(f"Skipping existing article: {title}")
                    continue
                
                # Skip if title contains certain keywords that indicate it might be too general
                skip_keywords = ['list of', 'history of', 'timeline', 'comparison of', 'overview of']
                if any(keyword in title.lower() for keyword in skip_keywords):
                    logger.debug(f"Skipping general article: {title}")
                    continue
                    
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    # Skip disambiguation and list pages
                    if "may refer to" in page.content[:100].lower() or "list of" in page.title.lower():
                        continue
                        
                    # Skip if content is too short (likely stub) or too long (likely too general)
                    if len(page.content) < 1000 or len(page.content) > 50000:
                        continue
                        
                    article = {
                        "text": page.content,
                        "intent": "Finance",
                        "title": page.title,
                        "text_clean": clean_text(page.content)
                    }
                    articles.append(article)
                    existing_titles.add(title.lower())
                    logger.info(f"Added finance article: {title}")
                    
                except (wikipedia.exceptions.DisambiguationError,
                        wikipedia.exceptions.PageError,
                        wikipedia.exceptions.HTTPTimeoutError) as e:
                    logger.debug(f"Skipping article {title}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching finance articles for category {category}: {str(e)}")
            continue
            
    logger.info(f"Fetched {len(articles)} finance-related articles")
    return articles

def fetch_wikipedia_data():
    """Fetch and enhance training data from Wikipedia."""
    logger.info("Loading and enhancing Wikipedia data...")
    
    TARGET_ARTICLES_PER_CATEGORY = 150  # Increased from 70 to get more balanced dataset
    
    # Get existing data if available
    data_file = Path(__file__).parent / "training_data" / "enhanced_wikipedia_training_data.csv"
    if data_file.exists():
        df = pd.read_csv(data_file)
        # Convert text columns to string and handle NaN
        df['text'] = df['text'].fillna('').astype(str)
        df['title'] = df['title'].fillna('').astype(str)
        df['intent'] = df['intent'].fillna('').astype(str)
        df['text_clean'] = df['text_clean'].fillna('')
        logger.info(f"Loaded {len(df)} existing articles")
        
        # Check category distribution
        category_counts = df["intent"].value_counts()
        logger.info("Current category distribution:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} articles")
        
        # Balance categories by reducing overrepresented ones
        balanced_dfs = []
        for category in df['intent'].unique():
            category_df = df[df['intent'] == category]
            if len(category_df) > TARGET_ARTICLES_PER_CATEGORY:
                # Keep most recent articles up to target_size
                category_df = category_df.tail(TARGET_ARTICLES_PER_CATEGORY)
            balanced_dfs.append(category_df)
        
        df = pd.concat(balanced_dfs, ignore_index=True)
        logger.info("After balancing existing categories:")
        for category, count in df["intent"].value_counts().items():
            logger.info(f"  {category}: {count} articles")
            
    else:
        df = pd.DataFrame(columns=["text", "intent", "title", "text_clean"])
    
    # Calculate how many articles we need for each category
    music_needed = max(0, TARGET_ARTICLES_PER_CATEGORY - len(df[df['intent'] == 'Music']))
    finance_needed = max(0, TARGET_ARTICLES_PER_CATEGORY - len(df[df['intent'] == 'Finance']))
    tech_needed = max(0, TARGET_ARTICLES_PER_CATEGORY - len(df[df['intent'] == 'Technology']))
    
    logger.info(f"Articles needed: Music={music_needed}, Finance={finance_needed}, Technology={tech_needed}")
    
    if music_needed > 0:
        music_articles = fetch_music_articles(music_needed)
        music_df = pd.DataFrame(music_articles)
        df = pd.concat([df, music_df], ignore_index=True)
        
    if finance_needed > 0:
        finance_articles = fetch_finance_articles(finance_needed)
        finance_df = pd.DataFrame(finance_articles)
        df = pd.concat([df, finance_df], ignore_index=True)
        
    if tech_needed > 0:
        tech_articles = fetch_tech_articles(tech_needed)
        tech_df = pd.DataFrame(tech_articles)
        df = pd.concat([df, tech_df], ignore_index=True)
    
    # Remove duplicates based on title
    df = df.drop_duplicates(subset=['title'], keep='first')
    
    # Clean text for new entries and ensure all text fields are strings
    mask = df['text_clean'].isna()
    df.loc[mask, 'text_clean'] = df.loc[mask, 'text'].apply(clean_text)
    
    # Ensure all text fields are strings
    df['text'] = df['text'].fillna('').astype(str)
    df['title'] = df['title'].fillna('').astype(str)
    df['intent'] = df['intent'].fillna('').astype(str)
    df['text_clean'] = df['text_clean'].fillna('').astype(str)
    
    # Save enhanced dataset
    df.to_csv(data_file, index=False)
    logger.info(f"Saved enhanced dataset with {len(df)} articles to {data_file}")
    
    # Log final category distribution
    category_counts = df["intent"].value_counts()
    logger.info("Final categories distribution:")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count} articles")
        
    return df

def fetch_tech_articles(num_articles=100):
    """Fetch technology-related articles from Wikipedia."""
    logger.info("Fetching technology-related articles from Wikipedia...")
    
    # Load existing data if available to check for duplicates
    data_file = Path(__file__).parent / "training_data" / "enhanced_wikipedia_training_data.csv"
    existing_titles = set()
    if data_file.exists():
        existing_df = pd.read_csv(data_file)
        existing_titles = set(existing_df['title'].str.lower())
    
    articles = []
    categories = [
        "Computer_programming",
        "Software_engineering",
        "Artificial_intelligence",
        "Machine_learning",
        "Cloud_computing",
        "Computer_hardware",
        "Mobile_technology",
        "Internet_technology",
        "Operating_systems",
        "Programming_languages",
        "Web_development",
        "Database_management_systems",
        "Computer_security",
        "Computer_networks",
        "Software_development",
        "Information_technology",
        "Digital_electronics",
        "Computer_architecture",
        "Software_design_patterns",
        "Computer_graphics",
        "Human-computer_interaction",
        "Computer_vision",
        "Natural_language_processing",
        "Robotics"
    ]
    
    for category in categories:
        if len(articles) >= num_articles:
            break
            
        try:
            # Get articles from category
            category_articles = wikipedia.page(category).links[:20]  # Limit to first 20 links per category
            
            # Shuffle to get random articles from category
            random.shuffle(category_articles)
            
            for title in category_articles[:10]:  # Take max 10 articles per category for tech
                if len(articles) >= num_articles:
                    break
                    
                # Skip if already downloaded
                if title.lower() in existing_titles:
                    logger.debug(f"Skipping existing article: {title}")
                    continue
                
                # Skip if title contains certain keywords that indicate it might be too general
                skip_keywords = ['list of', 'history of', 'timeline', 'comparison of', 'overview of']
                if any(keyword in title.lower() for keyword in skip_keywords):
                    logger.debug(f"Skipping general article: {title}")
                    continue
                    
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    # Skip disambiguation and list pages
                    if "may refer to" in page.content[:100].lower() or "list of" in page.title.lower():
                        continue
                        
                    # Skip if content is too short (likely stub) or too long (likely too general)
                    if len(page.content) < 1000 or len(page.content) > 50000:
                        continue
                        
                    article = {
                        "text": page.content,
                        "intent": "Technology",
                        "title": page.title,
                        "text_clean": clean_text(page.content)
                    }
                    articles.append(article)
                    existing_titles.add(title.lower())
                    logger.info(f"Added technology article: {title}")
                    
                except (wikipedia.exceptions.DisambiguationError,
                        wikipedia.exceptions.PageError,
                        wikipedia.exceptions.HTTPTimeoutError) as e:
                    logger.debug(f"Skipping article {title}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching technology articles for category {category}: {str(e)}")
            continue
            
    logger.info(f"Fetched {len(articles)} technology-related articles")
    return articles

def save_model(model, tokenizer, label_encoder, save_dir, training_history=None):
    """Save model and associated files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = save_dir / "lstm_model"
    model.save(model_path)
    logger.info(f"Model updated at {model_path}")
    
    # Save tokenizer
    tokenizer_path = save_dir / "tokenizer.json"
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    logger.info(f"Tokenizer updated at {tokenizer_path}")
    
    # Save label encoder
    label_encoder_path = save_dir / "label_encoder.pkl"
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Label encoder updated at {label_encoder_path}")
    
    # Save training history if provided
    if training_history is not None:
        history_path = save_dir / "training_history.json"
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {}
        for key, values in training_history.items():
            history_dict[key] = [float(val) for val in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        logger.info(f"Training history updated at {history_path}")

def check_existing_model():
    """Check if model files exist and return their paths."""
    model_dir = Path(__file__).parent / "lstm_model"
    model_files = {
        "Model": model_dir / "lstm_model.h5",
        "Tokenizer": model_dir / "tokenizer.pkl",
        "Label Encoder": model_dir / "label_encoder.pkl",
        "Training History": model_dir / "training_history.json"
    }
    
    existing_files = {name: path for name, path in model_files.items() if path.exists()}
    return model_dir, existing_files

def clean_old_model_files():
    """Clean up old model files before training, but ask user first."""
    model_dir, existing_files = check_existing_model()
    
    if not existing_files:
        logger.info("No existing model files found.")
        return True
    
    # Show existing files
    logger.info(f"\nFound existing model files in: {model_dir.absolute()}")
    for name, path in existing_files.items():
        logger.info(f"- {name}: {path.name}")
    
    # Prompt user
    while True:
        response = input("\nDo you want to delete the existing model files and train from scratch? (yes/no): ").lower().strip()
        if response in ['yes', 'no']:
            break
        print("Please answer 'yes' or 'no'")
    
    if response == 'no':
        logger.info("Keeping existing model files.")
        return False
    
    # Delete files if user confirmed
    logger.info("Deleting existing model files...")
    for name, file in existing_files.items():
        try:
            file.unlink()
            logger.info(f"Deleted {name}: {file.name}")
        except Exception as e:
            logger.error(f"Error deleting {file}: {str(e)}")
    
    return True

def calculate_class_weights(y):
    """Calculate class weights to handle imbalanced data."""
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    # Get unique classes and their counts
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    
    # Create dictionary mapping class indices to weights
    class_weights = {i: w for i, w in zip(classes, weights)}
    
    logger.info("Class weights:")
    for cls, weight in class_weights.items():
        logger.info(f"  Class {cls}: {weight:.2f}")
    
    return class_weights

def main():
    """Main training function."""
    # Check GPU support
    check_gpu_support()
    
    # Check for existing model and ask user what to do
    should_train_new = clean_old_model_files()
    if not should_train_new:
        logger.info("Exiting as user chose to keep existing model.")
        return
    
    # Load and preprocess data
    logger.info("Loading existing tokenizer...")
    tokenizer_path = Path(__file__).parent / "lstm_model" / "tokenizer.pkl"
    
    if tokenizer_path.exists():
        logger.warning("Found existing tokenizer but ignoring it for fresh training")
    
    # Get enhanced Wikipedia data
    df = fetch_wikipedia_data()
    
    # Prepare text data
    texts = df['text_clean'].values
    intents = df['intent'].values
    
    # Create and fit tokenizer
    logger.info("Creating new tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(intents)
    
    # Calculate class weights
    class_weights = calculate_class_weights(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Added stratify to maintain class distribution
    )
    
    # Create and train model
    model = create_model(
        vocab_size=min(MAX_WORDS, len(tokenizer.word_index) + 1),
        num_classes=len(label_encoder.classes_)
    )
    
    # Create callbacks
    callbacks = [
        TensorBoard(
            log_dir='logs',
            histogram_freq=1,
            write_graph=True
        ),
        ModelCheckpoint(
            filepath=str(Path(__file__).parent / "lstm_model" / "checkpoints" / "model_{epoch:02d}.h5"),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
    
    # Train model with class weights
    logger.info("Training model with class weights...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weights  # Added class weights
    )
    
    # Save model and associated files
    save_model(
        model=model,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        save_dir=str(Path(__file__).parent / "lstm_model"),
        training_history=history.history
    )
    
    # Evaluate model
    logger.info("\nEvaluating model on test set:")
    loss, accuracy = model.evaluate(X_test, y_test)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info(f"Test loss: {loss:.4f}")

if __name__ == '__main__':
    main() 