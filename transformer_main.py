import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            texts.append(entry['text'])
            labels.append(entry['label'])
    return texts, labels

train_texts, train_labels = load_data('./train.jsonl')
test_texts, test_labels = load_data('./test.jsonl')

# Vectorize texts
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts).toarray()
X_test = vectorizer.transform(test_texts).toarray()

# Encode labels
label_encoder = LabelEncoder()
y_train = torch.tensor(label_encoder.fit_transform(train_labels), dtype=torch.long)
y_test = torch.tensor(label_encoder.transform(test_labels), dtype=torch.long)

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.texts[idx], dtype=torch.float32).unsqueeze(0),  # Add sequence length dimension
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, num_heads=8, num_layers=3, d_ff=256):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        output = self.transformer(x, x)
        output = output.mean(dim=0)  # Aggregate sequence outputs
        output = self.fc(output)
        return output

# Create datasets and dataloaders
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, optimizer, and loss function
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = TransformerModel(input_dim=X_train.shape[1], num_classes=len(label_encoder.classes_))
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
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
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')
