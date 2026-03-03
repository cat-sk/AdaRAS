import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        self.input_dim = input_dim

        self.attention_scorer_linear = nn.Linear(input_dim, 1)
        self.attention_activation = nn.Tanh()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, token_features):
        raw_scores = self.attention_scorer_linear(token_features)
        
        activated_scores = self.attention_activation(raw_scores)
        
        attention_weights = F.softmax(activated_scores, dim=0)
        
        weighted_avg_vector = (token_features * attention_weights).sum(dim=0)
        
        logit = self.classifier(weighted_avg_vector)

        return logit