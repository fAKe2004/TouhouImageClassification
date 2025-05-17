'''
MoE with ResNet50 as backbone and MLP as expert.
'''

import torch
import torch.nn as nn
import torchvision.models as models

import TIC.ViT.model as vit

class MLPExpert(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPExpert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class GatingNetwork(nn.Module):

    def __init__(self, num_experts, top_k, random_t, pretrained = True):
        super(GatingNetwork, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.vit = vit.ViT(num_classes = num_experts, pretrained=pretrained, model_name='google/vit-base-patch16-224')

    def forward(self, x):
        logits = self.vit(x).logits
        if self.training:
            logits += torch.randn_like(logits) * 0.01
        top_k_weights, top_k_indeces = torch.topk(logits, k = self.top_k, dim = 1)
        smoothed_weights = torch.softmax(top_k_weights, dim = 1)
        return smoothed_weights, top_k_indeces

class MoEClassifier(nn.Module):

    def __init__(self, backbone, experts: nn.ModuleList, gate, top_k, num_classes):
        super(MoEClassifier, self).__init__()
        self.experts = experts
        self.top_k = top_k
        self.num_classes = num_classes
        self.shared_backbone = backbone
        self.gate = gate

    def forward(self, x):
        features = self.shared_backbone(x)  # (B, C)
        top_k_weights, top_k_indeces = self.gate(x) # (B, K), (B, K)
        gate_weights = torch.zeros(x.shape[0], len(self.experts), device=x.device)    # (B, N)
        gate_weights = torch.scatter(gate_weights, 1, top_k_indeces, top_k_weights)
        
        expert_outputs = torch.stack([expert(features).logits for expert in self.experts], dim = 1)
        combined_output = torch.bmm(gate_weights.unsqueeze(1), expert_outputs).squeeze(1)
        return combined_output, gate_weights, top_k_indeces

def make_ViTMoE(num_classes : int, num_experts : int, top_k : int, gateway_t: float, pretrained : bool = True, model_name : str | None = None, gate_pretrained : bool = True):
    return MoEClassifier(
        backbone = nn.Identity(),
        experts = nn.ModuleList([
            vit.ViT(
                num_classes = num_classes, pretrained = pretrained, model_name = model_name,
            )
            for _ in range(num_experts)
        ]),
        gate = GatingNetwork(num_experts = num_experts, top_k = top_k, random_t = gateway_t, pretrained = gate_pretrained),
        top_k = top_k,
        num_classes = num_classes,
    )
