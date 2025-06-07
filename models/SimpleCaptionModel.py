import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCaptionModel(nn.Module):
    def __init__(self, vocab_size):#, embed_dim, hidden_dim, output_dim, model_name):
        super(SimpleCaptionModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # (B, 8, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  # (B, 8, 56, 56)
        )

        self.fc1 = nn.Linear(8 * 56 * 56, 256)  # (B, 256)

        self.embed = nn.Embedding(vocab_size, 256)

        self.rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)  # (B, seq_len, 256)

        self.fc2 = nn.Linear(256, vocab_size)  # (B, seq_len, vocab_size)
    
    def forward(self, images, captions):
        # images: (B, 3, 224, 224)
        # captions: (B, seq_len=38)

        F = self.cnn(images) # (B, 8, 56, 56)
        F = F.view(F.size(0), -1) # (B, 8*56*56)

        rnn_hidden = nn.Tanh()(self.fc1(F)) # (B, 256)
        rnn_input = self.embed(captions)  # (B, seq_len, 256)

        rnn_out, _ = self.rnn(rnn_input, (rnn_hidden.unsqueeze(0), torch.zeros_like(rnn_hidden).unsqueeze(0))) # (B, seq_len, 256)

        rnn_out = rnn_out.contiguous().reshape(-1, rnn_out.size(-1))  # (B * seq_len, 256)
        logits = self.fc2(rnn_out)  # (B * seq_len, vocab_size)

        return logits