from typing import Callable
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Custom Imports
from util.custom_image_loader import load_image
from util.image_graph_creation import build_graph_from_image

class ImageGraphFolderDataset(Dataset):
    def __init__(self, backbone, 
                 device, 
                 graph_kwargs: dict = {'n_segments': 200, 'k_nn': 5}, 
                 transform_kwargs: dict = {'input_size': (224,224), 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
                 root_path: str = '.'):
        self.transform: Callable = transforms.Compose([
        transforms.Resize(transform_kwargs['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_kwargs['mean'], std=transform_kwargs['std']),
    ])
        self.imagefolder = ImageFolder(root=root_path, transform=self.transform)

        self.backbone = backbone
        self.device = device
        self.graph_kwargs = graph_kwargs

    def __len__(self):
        return len(self.imagefolder)

    def __getitem__(self, idx):
        img, label = self.imagefolder[idx]  # PIL image via ImageFolder
        # If needed convert to numpy: img_np = np.array(img) etc.
        data = build_graph_from_image(
            img=img,
            label=label,
            backbone=self.backbone,
            device=self.device,
            **self.graph_kwargs
        )
        return data