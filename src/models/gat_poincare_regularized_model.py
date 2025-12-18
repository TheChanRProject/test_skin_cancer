import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GATConv, global_mean_pool

### Poincaré embedding helper with regularization ###
class PoincareEmbedder(nn.Module):
    def __init__(self, in_dim, embed_dim, dropout=0.3, eps=1e-5):
        super().__init__()
        self.linear = nn.Linear(in_dim, embed_dim)
        self.batch_norm = nn.BatchNorm1d(embed_dim, momentum=0.1)
        self.dropout = nn.Dropout(p=dropout)
        self.eps = eps

        # Kaiming initialization for ReLU-based networks
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        z = self.linear(x)
        z = self.batch_norm(z)
        z = F.relu(z)
        z = self.dropout(z)

        norm = z.norm(p=2, dim=1, keepdim=True)
        max_norm = (1.0 - self.eps)
        # Scale down if norm > max_norm
        z = torch.where(norm >= max_norm, z / norm * max_norm, z)
        return z

def poincare_distance(u, v, eps=1e-5):
    diff = u - v
    diff_sq = (diff * diff).sum(dim=1)
    norm_u_sq = (u * u).sum(dim=1)
    norm_v_sq = (v * v).sum(dim=1)
    denom = (1.0 - norm_u_sq) * (1.0 - norm_v_sq)
    denom = torch.clamp(denom, min=eps)
    arg = 1.0 + 2.0 * diff_sq / denom
    arg = torch.clamp(arg, min=1.0 + eps)
    return torch.log(arg + torch.sqrt(arg * arg - 1.0))

### Main model with comprehensive regularization ###
class ImageGraphHyperbolicGATClassifier(nn.Module):
    def __init__(self, in_feature_size: int = 1024, num_classes=7, gat_hidden_dim=128, gat_heads=8,
                 poincare_dim=32, dropout=0.3, eps=1e-5, device: str = 'mps'):
        super().__init__()

        # Determine output feature dimension of backbone
        # For convnext_base, the final feature dim is 1024 (depends on version)
        in_feat_dim = in_feature_size

        self.device_type = device
        self.dropout = dropout

        # Input layer normalization for graph features
        self.input_norm = nn.LayerNorm(in_feat_dim)
        self.input_dropout = nn.Dropout(p=dropout)

        # Graph Attention Layers with regularization
        self.gat1 = GATConv(in_feat_dim, gat_hidden_dim, heads=gat_heads, concat=True, dropout=dropout)
        self.gat1_norm = nn.LayerNorm(gat_hidden_dim * gat_heads)
        self.gat1_dropout = nn.Dropout(p=dropout)

        self.gat2 = GATConv(gat_hidden_dim * gat_heads, gat_hidden_dim, heads=1, concat=False, dropout=dropout)
        self.gat2_norm = nn.LayerNorm(gat_hidden_dim)
        self.gat2_dropout = nn.Dropout(p=dropout)

        # Poincaré embedding with regularization
        self.poincare = PoincareEmbedder(gat_hidden_dim, poincare_dim, dropout=dropout, eps=eps)

        # Classifier with dropout and batch normalization
        self.classifier_dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(poincare_dim, num_classes)

        # Initialize GAT and classifier weights with Kaiming initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for ReLU activations"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                # Initialize normalization layers
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data):
        """
        data.x: raw images tensor (N, C, H, W) or feature tensor (N, F)
        data.edge_index: edge index tensor for graph (2, E)
        data.batch (optional): if batching multiple graphs
        """

        # X, Edges, and Batch
        x, edge_idx, batch = data.x, data.edge_index, data.batch

        x = x.to(torch.device(self.device_type))
        edge_idx = edge_idx.to(torch.device(self.device_type))
        batch = batch.to(torch.device(self.device_type))

        # Input normalization and dropout
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # First GAT layer with ReLU activation
        h = self.gat1(x, edge_idx)
        h = self.gat1_norm(h)
        h = F.relu(h)
        h = self.gat1_dropout(h)

        # Second GAT layer with ReLU activation
        h = self.gat2(h, edge_idx)
        h = self.gat2_norm(h)
        h = F.relu(h)
        h = self.gat2_dropout(h)

        # Global Mean Pooling
        h = global_mean_pool(x=h, batch=batch)

        # Hyperbolic embedding (includes its own normalization and dropout)
        z = self.poincare(h)  # (N, poincare_dim)

        # Classification with dropout
        z = self.classifier_dropout(z)
        logits = self.classifier(z)  # (N, num_classes)

        print("Logit shape: ", logits.shape)

        return logits, z
