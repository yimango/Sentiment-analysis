import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset

# Load and preprocess the data
train_df = pd.read_csv('./train_df.csv')
train_df = train_df[['text', 'sentiment']]
# Convert sentiment to 0, 1, 2 from 'negative', 'neutral', 'positive'
train_df['sentiment'] = train_df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
train_df = train_df.sample(frac=1).reset_index(drop=True)

test_df = pd.read_csv('./test_df.csv')
test_df = test_df[['text', 'sentiment']]
test_df['sentiment'] = test_df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
test_df = test_df.sample(frac=1).reset_index(drop=True)

# Ensure all text entries are strings
train_df['text'] = train_df['text'].astype(str)
test_df['text'] = test_df['text'].astype(str)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to tokenize and encode the texts
def tokenize_and_encode(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

# Tokenize and encode the texts
train_encodings = tokenize_and_encode(train_df['text'].tolist())
test_encodings = tokenize_and_encode(test_df['text'].tolist())

num_embeddings = len(tokenizer.vocab)

# Dataset class
class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Custom sentiment classifier
class custom_sentiment_classifier(nn.Module):
    def __init__(self):
        super(custom_sentiment_classifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, 768)
        self.recurrent = nn.LSTM(768, 256, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(256, 3)
    
    def forward(self, input_ids, attention_mask):
        # Embedding
        embedded = self.embedding(input_ids)
        
        # LSTM layer
        lstm_out, _ = self.recurrent(embedded)
        
        # Pooling: Max pooling over the sequence dimension
        pooled, _ = torch.max(lstm_out, 1)
        
        # Apply dropout and ReLU
        output = self.relu(pooled)

        return output


# Prepare the datasets and dataloaders
train_dataset = TweetDataset(train_encodings, train_df['sentiment'].tolist())
test_dataset = TweetDataset(test_encodings, test_df['sentiment'].tolist())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# Set device and initialize model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = custom_sentiment_classifier()
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # Move inputs and labels to the device (GPU or CPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss:.4f}')

    # Evaluation loop
    model.eval()
    val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')
