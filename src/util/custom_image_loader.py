import numpy as np
from torch import from_numpy, Tensor
from PIL import Image

def load_image(img_path: str) -> Tensor:
    """
    ### Inputs
    - img_path (string): Path to where your image is located.

    ### Output
    - img_array (PyTorch Tensor): Array representation of your loaded image from Pillow.
    """

    # Get the img
    img = Image.open(fp=img_path)

    # Convert it to a RGB image
    img = img.convert(mode='RGB')

    # Convert the RGB image into a numpy array
    return from_numpy(np.array(img))

