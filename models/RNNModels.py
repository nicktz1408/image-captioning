import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import stack_rnn_cells

lstm512_cell = nn.LSTMCell(input_size=512, hidden_size=512)
lstm1024_cell = nn.LSTMCell(input_size=1024, hidden_size=1024)

lstm1024_2layer_cell = stack_rnn_cells(nn.LSTMCell, input_size=1024, hidden_size=1024, num_layers=2)


lstm512_3layer = nn.LSTM(input_size=512, hidden_size=512, num_layers=3, batch_first=True)
lstm1024_5layer = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=5, batch_first=True)

class LSTMWithAdditiveAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, attention_model):
        super(LSTMWithAdditiveAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.attention_model = attention_model

        self.rnn_cells = stack_rnn_cells(nn.LSTMCell, 
                                         input_size=self.input_size, 
                                         hidden_size=self.hidden_size, 
                                         num_layers=self.num_layers)


    def forward(self, F, embeddings):
        # F: (B, num_regions, cnn_out_dim)
        # embeddings: (B, T, embed_dim)

        B, seq_len, _ = embeddings.shape

        h = [ torch.zeros(B, self.hidden_size) for _ in range(self.num_layers) ] # list more performant than 3D tensor
        c = [ torch.zeros(B, self.hidden_size) for _ in range(self.num_layers) ]

        outputs = []

        for t in range(seq_len):
            embed = embeddings[:, t, :]  # (B, embed_dim)
            attention_vec = self.attention_model(F, h[0]) # (B, features_dim), only first layer for now

            rnn_input = torch.cat([embed, attention_vec], dim=1) # (B, rnn_input_dim)

            h[0], c[0] = self.rnn_cells[0](rnn_input, (h[0], c[0]))

            for l in range(1, self.num_layers): # use prev layer h as input
                h[l], c[l] = self.rnn_cells[l](h[l - 1], (h[l-1], c[l-1]))

            outputs.append(h[-1]) # final hidden layer

        # outputs: T times (B, hidden_dim)
        return torch.stack(outputs, dim=1) # (B, T, hidden_dim)
