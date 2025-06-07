import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptionModel(nn.Module):
    def __init__(self, cnn_model, rnn_model, vocab_size, embed_dim):
        super(CaptionModel, self).__init__()

        self.cnn_model = cnn_model
        self.rnn_model = rnn_model

        self.cnn_output_dim = cnn_model.output_dim

        self.hidden_dim = rnn_model.hidden_size # output_dim same
        self.rnn_num_layers = rnn_model.num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)


        self.fc_h0 = nn.Linear(self.cnn_output_dim, self.hidden_dim * self.rnn_num_layers)  # (B, channel * width * height) -> (B, rnn_input_dim)
        self.fc_c0 = nn.Linear(self.cnn_output_dim, self.hidden_dim * self.rnn_num_layers)  # (B, channel * width * height) -> (B, rnn_input_dim)

        self.tanh = nn.Tanh()
        
        self.fc_final = nn.Linear(self.hidden_dim, vocab_size) # (B * seq_length, rnn_input_dim) -> (B * seq_length, vocab_size)

    def forward(self, images, captions):
        F = self.cnn_model(images) # extracted features of dim (B, cnn_out_dim)
        F = F.view(F.size(0), -1)

        rnn_hidden_h0 = self.tanh(self.fc_h0(F))
        rnn_hidden_h0 = rnn_hidden_h0.view(self.rnn_num_layers, -1, self.hidden_dim)  # (B, num_layers, rnn_hidden_dim)

        rnn_hidden_c0 = self.tanh(self.fc_c0(F))
        rnn_hidden_c0 = rnn_hidden_c0.view(self.rnn_num_layers, -1, self.hidden_dim)  # (B, num_layers, rnn_hidden_dim)

        rnn_input = self.embedding(captions)

        rnn_out, _ = self.rnn_model(rnn_input, (rnn_hidden_h0, rnn_hidden_c0))  # (B, seq_len, rnn_input_dim)
        rnn_out = rnn_out.contiguous().view(-1, rnn_out.size(-1))  # Flatten the output for the fully connected layer

        logits = self.fc_final(rnn_out)  # (B * seq_len, vocab_size)
        return logits