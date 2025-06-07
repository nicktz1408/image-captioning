import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, output_dim=256):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, H/8, W/8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.features(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        x = self.fc(x)  # (B, output_dim)
        return x

class CaptioningModel(nn.Module):
    def __init__(self, cnn_out_dim, embed_dim, hidden_dim, vocab_size, pad_idx):
        super(CaptioningModel, self).__init__()
        self.cnn = SimpleCNN(cnn_out_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_hidden = nn.Linear(cnn_out_dim, hidden_dim)

    def forward(self, images, captions):
        batch_size = images.size(0)

        # Encode image
        image_features = self.cnn(images)  # (B, cnn_out_dim)
        h0 = torch.tanh(self.init_hidden(image_features)).unsqueeze(0)  # (1, B, hidden_dim)
        c0 = torch.zeros_like(h0)  # (1, B, hidden_dim)

        # Embed captions
        embeddings = self.embedding(captions)  # (B, seq_len, embed_dim)

        # Decode
        outputs, _ = self.lstm(embeddings, (h0, c0))  # (B, seq_len, hidden_dim)
        logits = self.fc(outputs)  # (B, seq_len, vocab_size)

        return logits