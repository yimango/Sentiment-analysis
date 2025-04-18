import re
import pandas as pd
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, Dataset
from collections import Counter

# -----------------------
# Data Loading and Cleaning
# -----------------------
def load_and_preprocess_data(train_path, test_path):
    # Load JSONL files
    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    # Select relevant columns
    train_df = train_df[['text', 'label']]
    test_df = test_df[['text', 'label']]
    
    # Ensure all text entries are strings
    train_df['text'] = train_df['text'].astype(str)
    test_df['text'] = test_df['text'].astype(str)
    
    # Remove punctuation, special characters, links, and convert to lowercase
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
        text = text.lower()
        return text
    
    train_df['text'] = train_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].apply(clean_text)

    # After cleaning, filter out empty tweets
    train_df = train_df[train_df['text'].str.strip() != '']
    test_df = test_df[test_df['text'].str.strip() != '']
    
    return train_df, test_df

train_df, test_df = load_and_preprocess_data('./train.jsonl', './test.jsonl')

# -----------------------
# Tokenization and Numericalization
# -----------------------
def tokenize(text):
    return text.split()

# Build vocabulary from training data
all_tokens = [token for text in train_df['text'] for token in tokenize(text)]
vocab_counter = Counter(all_tokens)
# Start indexing from 2 to reserve 0 for <PAD> and 1 for <UNK>
vocab = {word: idx + 2 for idx, (word, _) in enumerate(vocab_counter.most_common())}
vocab['<PAD>'] = 0  # Padding token
vocab['<UNK>'] = 1  # Unknown token
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

def numericalize(text):
    return [vocab.get(word, vocab['<UNK>']) for word in tokenize(text)]

# Create sequences and record their original lengths
train_raw_sequences = [torch.tensor(numericalize(text), dtype=torch.long) for text in train_df['text']]
train_lengths = [len(seq) for seq in train_raw_sequences]
# Pad training sequences
train_sequences = torch.nn.utils.rnn.pad_sequence(train_raw_sequences, batch_first=True, padding_value=vocab['<PAD>'])

test_raw_sequences = [torch.tensor(numericalize(text), dtype=torch.long) for text in test_df['text']]
test_lengths = [len(seq) for seq in test_raw_sequences]
# Pad test sequences
test_sequences = torch.nn.utils.rnn.pad_sequence(test_raw_sequences, batch_first=True, padding_value=vocab['<PAD>'])

# Convert labels to tensors
y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
y_test = torch.tensor(test_df['label'].values, dtype=torch.long)

# -----------------------
# Dataset Class
# -----------------------
class TweetDataset(Dataset):
    def __init__(self, texts, lengths, labels):
        """
        texts: Padded tensor of token indices, shape: (num_samples, max_seq_len)
        lengths: List (or tensor) of original sequence lengths
        labels: Tensor of labels
        """
        self.texts = texts
        self.lengths = lengths
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],  # Already a tensor of type long
            'label': self.labels[idx],
            'length': self.lengths[idx]
        }

    def __len__(self):
        return len(self.labels)

# Create dataset instances
train_dataset = TweetDataset(train_sequences, train_lengths, y_train)
test_dataset = TweetDataset(test_sequences, test_lengths, y_test)


# -----------------------
# Model Definition: SentimentLSTM
# -----------------------
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        # If bidirectional, combine the outputs from both directions
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text: [batch_size, sent_len]
        embedded = self.embedding(text)  # [batch_size, sent_len, embedding_dim]
        embedded = self.dropout(embedded)  # Apply dropout to embeddings
        # Pack the embedded sequences for efficient processing
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,
                                                            batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # For bidirectional LSTM, concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        logits = self.fc(hidden)
        return logits

# -----------------------
# DataLoaders
# -----------------------
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------
# Model Initialization and Training Setup
# -----------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define hyperparameters for the LSTM model
vocab_size = len(vocab)
embedding_dim = 200      # You can adjust this
hidden_dim = 128         # Hidden state size of the LSTM
output_dim = 3           # Adjust depending on number of classes
n_layers = 2
bidirectional = True
dropout = 0.5
pad_idx = vocab['<PAD>']

model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim,
                      n_layers, bidirectional, dropout, pad_idx)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# -----------------------
# Training Loop
# -----------------------
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # Move inputs, lengths, and labels to device
        inputs = batch['text'].to(device)            # [batch_size, seq_len]
        lengths = batch['length']                     # List or tensor of lengths (kept on CPU for pack_padded_sequence)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: note that we pass both inputs and lengths
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
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
            inputs = batch['text'].to(device)
            lengths = batch['length']
            labels = batch['label'].to(device)
            
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

torch.save(model.state_dict(), 'sentiment_lstm_model.pth')