import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GATConv, global_mean_pool

### Poincaré embedding helper ###
class PoincareEmbedder(nn.Module):
    def __init__(self, in_dim, embed_dim, eps=1e-5):
        super().__init__()
        self.linear = nn.Linear(in_dim, embed_dim)
        self.eps = eps

    def forward(self, x):
        z = self.linear(x)
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

### Main model ###
class ImageGraphHyperbolicGATClassifier(nn.Module):
    def __init__(self, in_feature_size: int = 1024, num_classes=7, gat_hidden_dim=128, gat_heads=8, 
                 poincare_dim=32, eps=1e-5, device: str = 'mps'):
        super().__init__()
        
        # Determine output feature dimension of backbone
        # For convnext_base, the final feature dim is 1024 (depends on version)
        in_feat_dim = in_feature_size
        
        self.device_type = device

        # 2. Graph Attention Layers
        self.gat1 = GATConv(in_feat_dim, gat_hidden_dim, heads=gat_heads, concat=True)
        self.gat2 = GATConv(gat_hidden_dim * gat_heads, gat_hidden_dim, heads=1, concat=False)
        
        # 3. Poincaré embedding
        self.poincare = PoincareEmbedder(gat_hidden_dim, poincare_dim, eps=eps)
        
        # 4. Classifier
        self.classifier = nn.Linear(poincare_dim, num_classes)
        
    def forward(self, data):
        """
        data.x: raw images tensor (N, C, H, W)
        data.edge_index: edge index tensor for graph (2, E)
        data.batch (optional): if batching multiple graphs
        """
        
        # X, Edges, and Batch
        x, edge_idx, batch = data.x, data.edge_index, data.batch

        x.to(torch.device(self.device_type))
        edge_idx.to(torch.device(self.device_type))
        batch.to(torch.device(self.device_type))

        # # Edge index for the graph attention
        # edge_index = torch.empty((2,0), dtype=torch.long)  # no edges

        # 2) Graph attention
        h = F.elu(self.gat1(x, edge_idx))
        h = F.elu(self.gat2(h, edge_idx))

        # Global Mean Pooling
        h = global_mean_pool(x=h, batch=batch)
        
        # 3) Hyperbolic embedding
        z = self.poincare(h)  # (N, poincare_dim)
        
        # 4) Classification
        logits = self.classifier(z)  # (N, num_classes)

        print("Logit shape: ", logits.shape)

        return logits, z
