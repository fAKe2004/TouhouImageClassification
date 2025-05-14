'''
MoE with ResNet50 as backbone and MLP as expert.
'''

import torch
import torch.nn as nn

class Expert(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class GatingNetwork(nn.Module):

    def __init__(self, num_experts, top_k):
        super(GatingNetwork, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(32, num_experts)

    def forward(self, x):
        features = self.cnn(x)
        logits = self.fc(features)
        top_k_weights, top_k_indeces = torch.topk(logits, k = self.top_k, dim = 1)
        smoothed_weights = torch.softmax(top_k_weights, dim = 1)
        return smoothed_weights, top_k_indeces

class MoEClassifier(nn.Module):

    def __init__(self, backbone, experts: nn.ModuleList, gate, top_k):
        super(MoEClassifier, self).__init__()
        self.experts = experts
        self.top_k = top_k
        self.shared_backbone = backbone
        self.gate = gate

    def forward(self, x):
        features = self.shared_backbone(x)  # (B, C)
        top_k_weights, top_k_indeces = self.gate(x) # (B, K), (B, K)
        gate_weights = torch.zeros(x.shape[0], len(self.experts))    # (B, N)
        gate_weights = torch.scatter(gate_weights, 1, top_k_indeces, top_k_weights)
        
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim = 1)
        combined_output = torch.bmm(gate_weights.unsqueeze(1), expert_outputs).squeeze(1)
        return combined_output, gate_weights, top_k_indeces
