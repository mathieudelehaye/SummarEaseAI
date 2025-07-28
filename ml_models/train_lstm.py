"""
Train LSTM Intent Classifier (CPU version)
Uses PyTorch for training
"""

import logging
import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "training.log"), encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)

# Training parameters
MAX_WORDS = 10000  # Maximum vocabulary size
MAX_LEN = 100  # Maximum sequence length
EMBED_DIM = 100  # Word embedding dimension
LSTM_UNITS = 100  # LSTM layer units
EPOCHS = 10
BATCH_SIZE = 32


class IntentDataset(Dataset):
    """Custom dataset for intent classification"""

    def __init__(self, texts, labels, vocab_size=MAX_WORDS, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Create word index
        self.word_index = {}
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from texts"""
        words = {}
        for text in self.texts:
            for word in text.lower().split():
                words[word] = words.get(word, 0) + 1

        # Sort by frequency and take top MAX_WORDS
        sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
        for i, (word, _) in enumerate(sorted_words[: self.vocab_size - 1]):
            self.word_index[word] = i + 1  # Reserve 0 for padding

    def _text_to_sequence(self, text):
        """Convert text to sequence of word indices"""
        sequence = []
        for word in text.lower().split():
            if word in self.word_index:
                sequence.append(self.word_index[word])
            if len(sequence) >= self.max_len:
                break

        # Pad sequence
        if len(sequence) < self.max_len:
            sequence.extend([0] * (self.max_len - len(sequence)))
        return sequence

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Convert text to sequence
        sequence = self._text_to_sequence(text)

        return {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class LSTMClassifier(nn.Module):
    """LSTM-based intent classifier"""

    def __init__(self, vocab_size, embed_dim, lstm_units, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(lstm_units * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, sequence_length, embed_dim)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, sequence_length, lstm_units*2)

        # Take the last time step output
        lstm_out = lstm_out[:, -1, :]  # (batch_size, lstm_units*2)

        # Dropout and classification
        out = self.dropout(lstm_out)
        out = self.fc(out)

        return out


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device):
    """Train the model"""
    logger.info("\n🚀 Starting training...")

    best_accuracy = 0.0

    for epoch in range(epochs):
        logger.info(f"\n📊 Epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        logger.info(f"Accuracy: {accuracy:.4f}")

        # Save best model
        if accuracy > best_accuracy:
            logger.info("🎯 New best model! Saving...")
            best_accuracy = accuracy
            torch.save(
                model.state_dict(),
                os.path.join(os.path.dirname(__file__), "lstm_model.pt"),
            )

    logger.info("\n✅ Training completed!")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    return best_accuracy


def main():
    """Main training function"""
    try:
        # Set device to CPU
        device = torch.device("cpu")
        logger.info("🔧 Using CPU for training")

        # Load and preprocess data
        data_path = os.path.join(
            os.path.dirname(__file__), "training_data/intent_data.csv"
        )
        logger.info(f"Loading data from {data_path}")

        df = pd.read_csv(data_path)

        # Split data
        X = df["text"].values
        y = df["intent"].values

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create datasets
        train_dataset = IntentDataset(X_train, y_train)
        test_dataset = IntentDataset(X_test, y_test)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # Initialize model
        model = LSTMClassifier(
            vocab_size=MAX_WORDS,
            embed_dim=EMBED_DIM,
            lstm_units=LSTM_UNITS,
            num_classes=num_classes,
        ).to(device)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        # Train model
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=EPOCHS,
            device=device,
        )

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
