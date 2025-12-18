import torch
from os import getcwd, listdir

# Custom Utility Imports
import path_resolve
from models.timm_nn import load_timm_model
from util.format_util import line_separator
from util.custom_image_loader import load_image
from util.image_graph_creation import build_graph_from_image

# Custom Decorator Imports
from decorators.calculate_time import calculate_time

@calculate_time
def workflow():

    # Get your skin cancer images
    img_path = getcwd() + "/data/medium_publications/agentic_cv_skin_cancer/skin_cancer/skin_cancer_images_dst/train/akiec"

    # All the akiec images
    akiec_imgs = listdir(img_path)

    # Get your image array
    img_array = load_image(img_path=f'{img_path}/{akiec_imgs[0]}')

    print("Image array created.")

    line_separator()

    # Load ConvNext backbone from timm
    model = load_timm_model()

    print(type(model))

    print("CNN backbone from Timm loaded successfully.")

    line_separator()
    
    # Graph data
    graph_data = build_graph_from_image(
        img=img_array,
        label=0,
        backbone=model,
        device=torch.device('mps')
    )

    print("Graph data created successfully.")

    line_separator()

    print(graph_data)

    return graph_data

workflow()