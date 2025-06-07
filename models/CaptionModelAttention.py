import torch
import torch.nn as nn
import torch.nn.functional as F

from models.AdditiveAttention import AdditiveAttention
from models.RNNModels import LSTMWithAdditiveAttention

class CaptionModelAttention(nn.Module):
    def __init__(self, cnn_model, hidden_size, num_layers, vocab_size, embed_dim, attention_dim):
        super(CaptionModelAttention, self).__init__()

        self.cnn_output_dim = cnn_model.output_dim
        self.cnn_features_dim = cnn_model.features_dim

        self.embed_dim = embed_dim
        self.attention_dim = attention_dim

        self.input_size = self.cnn_features_dim + self.embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cnn_model = cnn_model
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention_model = AdditiveAttention(self.cnn_features_dim, hidden_size, self.attention_dim)
        self.rnn_model = LSTMWithAdditiveAttention(self.input_size, self.hidden_size, self.num_layers, self.attention_model)
        
        self.fc_final = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, images, captions):
        # captions: (B, seq_len)

        F = self.cnn_model(images) # (B, num_regions, cnn_out_dim)
        embed = self.embedding(captions) # (B, T, embed_dim)

        h_out = self.rnn_model(F, embed) # (B, T, hidden_dim)
        logits = self.fc_final(h_out.view(-1, h_out.size(-1))) # (B * T, vocab_size)
        
        return logits
