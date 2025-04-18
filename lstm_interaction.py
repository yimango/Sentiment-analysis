import torch
import torch.nn as nn
import pickle
import re

def clean_text(text):
    """
    Remove URLs, non-alphanumeric characters, and convert to lowercase.
    """
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    return text.lower().strip()

def tokenize(text):
    """
    Simple whitespace tokenizer.
    """
    return text.split()

def numericalize(text, vocab):
    """
    Convert a text string into a list of token indices using the provided vocabulary.
    """
    tokens = tokenize(text)
    # If the text is empty after tokenization, return <UNK> token.
    if not tokens:
        return [vocab['<UNK>']]
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths):
        # text: [batch_size, sent_len]
        embedded = self.embedding(text)  # [batch_size, sent_len, embedding_dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,
                                                            batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        logits = self.fc(hidden)
        return logits
    
# -----------------------
# Load the Vocabulary
# -----------------------
# Make sure that the vocabulary keys match what you expect.
vocab = pickle.load(open('vocab.pkl', 'rb'))
vocab_size = len(vocab)

# Check pad token key. Adjust if necessary.
# For example, if your vocab keys are uppercase:
pad_idx = vocab.get('<PAD>', None)
if pad_idx is None:
    # If not found as '<PAD>', try lowercase.
    pad_idx = vocab['<pad>']

# -----------------------
# Set Hyperparameters (must match your training settings)
# -----------------------
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
model_state_path = 'sentiment_lstm_model.pth'  # Path to your saved model state
model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
model.eval()

# -----------------------
# Sentiment Mapping
# -----------------------
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# -----------------------
# Interactive Inference Loop
# -----------------------
print("Interactive Sentiment Inference:")
print("Type a phrase and press Enter to see its predicted sentiment (or type 'exit' to quit).")

while True:
    phrase = input("> ")
    if phrase.lower().strip() == "exit":
        break

    # Preprocess the phrase.
    cleaned_phrase = clean_text(phrase)
    indices = numericalize(cleaned_phrase, vocab)
    
    # Convert list of indices into a tensor and add a batch dimension.
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # shape: [1, seq_len]
    # Define the sequence length as a list (required by pack_padded_sequence).
    input_length = [len(indices)]
    
    # Run the model.
    with torch.no_grad():
        logits = model(input_tensor, input_length)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    predicted_sentiment = sentiment_map.get(predicted_class, "Unknown")
    print(f"Predicted Sentiment: {predicted_sentiment}")
    print(f"Probabilities: {probabilities.cpu().numpy()}")
