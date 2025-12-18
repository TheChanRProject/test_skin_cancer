import torch
import torch.nn as nn
import cv2
import numpy as np
import sklearn.neighbors as nbr
from torch_geometric.data import Data
from skimage.segmentation import slic
from skimage.measure import regionprops

def build_graph_from_image(img: torch.Tensor,
                           label: int,
                           backbone: nn.Module,
                           device: torch.device,
                           n_segments=200,
                           k_nn=5
                           ) -> Data:
    """
    img : numpy array HxWx3 (RGB) or PIL image convertible
    label : int (0..6)
    backbone : timm (or any) model returning feature map
    device : torch.device
    Returns: torch_geometric.data.Data object
    """

    img_t = img.unsqueeze(0).to(device)  # shape [1,3,H,W]

    # 2. Run backbone to get feature map
    backbone = backbone.to(device)
    backbone.eval()
    with torch.no_grad():
        feat_map = backbone(img_t)  # assume returns [1, C, Hf, Wf]
    feat_map = feat_map.squeeze(0).cpu().numpy()  # shape (C, Hf, Wf)
    C, Hf, Wf = feat_map.shape

    # 3. Super-pixel segmentation
    img_np = img.permute(1,2,0).cpu().numpy() # H x W x 3
    segments = slic(img_np, n_segments=n_segments, compactness=10, start_label=0)
    num_nodes = segments.max() + 1

    # 4. Map segmentation to feature map resolution (downscale segments)
    segments_small = cv2.resize(segments.astype('int32'), (Wf, Hf), interpolation=cv2.INTER_NEAREST)

    # 5. Compute node features: average pooling over feature map per segment
    node_feats = np.zeros((num_nodes, C), dtype=np.float32)
    for s in range(num_nodes):
        mask = (segments_small == s)
        if mask.sum() == 0:
            node_feats[s] = 0
        else:
            node_feats[s] = feat_map[:, mask].mean(axis=1)

    # Optionally: append normalized centroid coordinates
    props = regionprops(segments)
    coords = np.zeros((num_nodes, 2), dtype=np.float32)
    H, W = segments.shape
    for s, prop in enumerate(props):
        y0, x0 = prop.centroid
        coords[s, 0] = x0 / W
        coords[s, 1] = y0 / H

    node_feats = np.concatenate([node_feats, coords], axis=1)  # shape [num_nodes, C+2]
    x = torch.tensor(node_feats, dtype=torch.float)

    # 6. Build edges: spatial adjacency
    adj_set = set()
    Hs, Ws = segments.shape
    for i in range(Hs - 1):
        for j in range(Ws - 1):
            s1 = segments[i, j]
            s2 = segments[i + 1, j]
            if s1 != s2:
                adj_set.add((s1, s2))
                adj_set.add((s2, s1))
            s3 = segments[i, j + 1]
            if s1 != s3:
                adj_set.add((s1, s3))
                adj_set.add((s3, s1))
    edge_list = list(adj_set)

    # 6b. Feature similarity edges: add top-k nearest neighbors
    neigh = nbr.NearestNeighbors(n_neighbors=k_nn + 1, metric='cosine')
    neigh.fit(node_feats)
    distances, indices = neigh.kneighbors(node_feats)
    for i in range(num_nodes):
        for idx in indices[i, 1:]:  # skip self
            edge_list.append((i, idx))
    edge_list = list(set(edge_list))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    y = torch.tensor(label, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    return data