import torch
from os import getcwd
from torch_geometric.loader import DataLoader

import path_resolve
from util.image_dataset import ImageGraphFolderDataset
from models.timm_nn import load_timm_model

def main():

    # Path to the images
    img_path = getcwd() + "/data/medium_publications/agentic_cv_skin_cancer/skin_cancer/skin_cancer_images_augmented"

    # Model
    model = load_timm_model()

    # Train Dataset object
    train_dataset = ImageGraphFolderDataset(
        root_path=f"{img_path}/train",
        backbone=model,
        device=torch.device('mps')
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=200,
        shuffle=True,
        num_workers=4
    )

    # Validation Dataset object
    val_dataset = ImageGraphFolderDataset(
        root_path=f"{img_path}/val",
        backbone=model,
        device=torch.device('mps')
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=200,
        shuffle=True,
        num_workers=4
    )


    return train_data_loader, val_data_loader

    


train_loader, val_loader = main()

print("Data loaders retrieved successfully.")
        
