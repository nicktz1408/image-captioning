import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, features_dim, hidden_dim, attention_dim):
        super(AdditiveAttention, self).__init__()

        self.features_dim = features_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(features_dim, attention_dim)
        self.fc2 = nn.Linear(hidden_dim, attention_dim)

        self.fc_out = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        # features: (B, num_regions, features_dim)
        # embeddings: (B, hidden_dim)

        B = features.size(0)

        features = features.contiguous().view(-1, self.features_dim) # (B * num_regions, features_dim)
        features_proj = self.fc1(features) # (B * num_regions, attention_dim)
        features_proj = features_proj.view(B, -1, features_proj.size(-1)) # (B, num_regions, attention_dim)


        hidden_proj = self.fc2(hidden_state) # (B, attention_dim)
        hidden_proj = hidden_proj.unsqueeze(1).expand(-1, features_proj.size(1), -1) # (B, num_regions, attention_dim)

        logits = self.fc_out(F.tanh(features_proj + hidden_proj)) # (B, num_regions, 1)
        logits = logits.squeeze(-1) # (B, num_regions)

        probs = F.softmax(logits, dim=-1).unsqueeze(-1) # (B, num_regions, 1)
        features = features.view(B, -1, features.size(-1)) # (B, num_regions, features_dim)

        return torch.sum(probs * features, dim=1) # (B, features_dim) broadcasts probs and then sums over num_regions (dim=1)
        