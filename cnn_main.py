import re
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer

# Load and preprocess the data
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
    
    return train_df, test_df

train_df, test_df = load_and_preprocess_data('./train.jsonl', './test.jsonl')

# Initialize the tokenizer (simple word-based)
vectorizer = CountVectorizer(max_features=5000)  # Limit vocab size to 5000 words

# Fit the vectorizer on training data and transform both train and test data
X_train = vectorizer.fit_transform(train_df['text']).toarray()
X_test = vectorizer.transform(test_df['text']).toarray()

# Convert labels to tensors with explicit dtype
y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
y_test = torch.tensor(test_df['label'].values, dtype=torch.long)

# Dataset class
class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.texts[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

# Fully connected (dense) neural network model
class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 128)         # Second fully connected layer
        self.fc3 = nn.Linear(128, 3)           # Output layer with 3 classes
        self.dropout = nn.Dropout(p=0.7)       # Dropout for regularization
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Prepare the datasets and dataloaders
train_dataset = TweetDataset(X_train, y_train)
test_dataset = TweetDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Set device and initialize model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
input_size = X_train.shape[1]  # Number of features in input
model = FeedForwardNN(input_size=input_size)
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # Move inputs and labels to the device (GPU or CPU)
        inputs = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
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
            inputs = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')
