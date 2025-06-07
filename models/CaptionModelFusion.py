import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptionModelFusion(nn.Module):
    def __init__(self, cnn_model, rnn_model, vocab_size, embed_dim, attention_dim):
        super(CaptionModelFusion, self).__init__()

        self.cnn_model = cnn_model
        self.rnn_model = rnn_model

        self.cnn_output_dim = cnn_model.output_dim

        self.hidden_dim = rnn_model.hidden_size # output_dim same
        self.rnn_num_layers = rnn_model.num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.f_embed = nn.Linear(self.cnn_output_dim, attention_dim)

        self.fc_final = nn.Linear(self.hidden_dim, vocab_size) # (B * seq_length, rnn_input_dim) -> (B * seq_length, vocab_size)

    def forward(self, images, captions):
        F = self.cnn_model(images) # extracted features of dim (B, cnn_out_dim, W, H)
        F = F.view(F.size(0), -1) # (B, num_features)

        embed_input = self.embedding(captions) # (B, seq_len - 1, embed_dim)
        F_input = self.f_embed(F) # (B, attention_dim)

        # Goal: combine them to get (B, seq_len - 1, embed_dim + num_features), the features repeated seq_len-1 times
        seq_len = embed_input.size(1)
        
        F_expanded = F_input.unsqueeze(1).repeat(1, seq_len, 1) # (B, seq_len - 1, attention_dim)

        rnn_input = torch.cat([embed_input, F_expanded], dim=2)

        rnn_out, _ = self.rnn_model(rnn_input)  # (B, seq_len, rnn_input_dim)
        rnn_out = rnn_out.contiguous().view(-1, rnn_out.size(-1))  # Flatten the output for the fully connected layer

        logits = self.fc_final(rnn_out)  # (B * seq_len, vocab_size)
        return logits