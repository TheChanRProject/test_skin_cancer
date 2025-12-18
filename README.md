# Skin Cancer Image Classification using Graph Neural Networks

## 1. Project Overview

![](https://p131.p1.n0.cdn.zight.com/items/jkuNmE6d/5436fedc-0d5a-4803-ae7d-1e1e018ec5ad.png?v=9d4b1cec915c66dedbfe732956fbdab3)

This project implements a skin cancer image classification system using a Graph Attention Network (GAT) with Poincaré embeddings. The core idea is to transform images into graphs and then use a GNN to perform classification. This approach allows the model to capture not just textural and color information, but also the spatial relationships between different regions of the image.

The workflow is as follows:
1.  Images are loaded and processed.
2.  A pre-trained backbone model (from `timm`) is used to extract features from different segments of the image.
3.  These features are used as node features in a graph, where nodes represent segments of the image. Edges are created based on spatial proximity (k-Nearest Neighbors).
4.  The resulting graph is fed into a `ImageGraphHyperbolicGATClassifier`, which is a GAT model that uses hyperbolic geometry to better model the hierarchical relationships in the data.
5.  The model is trained to classify the images into one of the skin cancer categories.

## 2. File Structure

-   `src/models/`: Contains the model definitions.
    -   `gat_poincare_regularized_model.py`: The main GAT model with hyperbolic embeddings.
    -   `timm_nn.py`: Loads a pre-trained model from the `timm` library to be used as a feature extractor.
-   `src/util/`: Contains utility functions for data processing and training.
    -   `image_dataset.py`: Defines the `ImageGraphFolderDataset` class, which loads images and converts them to graphs.
    -   `image_graph_creation.py`: Contains the logic for building a graph from an image.
    -   `focal_loss.py`: A custom loss function used for training.
-   `src/tests/`: Contains scripts for training and testing.
    -   `graph_cnn_train_loop.py`: The main training loop for the model.
    -   `graph_batch_test.py`: Loads the data and creates data loaders for training and validation.
-   `src/requirements.txt`: A list of the Python dependencies for this project.
-   `data/`: Directory for data. You will need to create subdirectories for your images.

## 3. Setup

### 3.1. Dependencies

To install the required dependencies, run the following command:

```bash
pip install -r src/requirements.txt
```

It is recommended to use a virtual environment to avoid conflicts with other projects.

### 3.2. Data

The model expects the data to be organized in a specific way. You need to create a directory (e.g., `data/skin_cancer_images`) and inside it, create `train` and `val` subdirectories. Inside `train` and `val`, create a subdirectory for each class of skin cancer, and place the corresponding images in these subdirectories.

The expected directory structure is:

```
data/skin_cancer_images/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
└── val/
    ├── class1/
    │   ├── image5.jpg
    │   └── image6.jpg
    └── class2/
        ├── image7.jpg
        └── image8.jpg
```

Once you have your data organized, you need to **update the path in `src/tests/graph_batch_test.py`**. Open the file and modify the `img_path` variable to point to your data directory:

```python
# In src/tests/graph_batch_test.py
# ...
def main():

    # Path to the images
    # Update this path to your data directory
    img_path = getcwd() + "/data/skin_cancer_images"
# ...
```

## 4. Training the Model

The main training script is `src/tests/graph_cnn_train_loop.py`. Before running it, you may need to make a few adjustments:

### 4.1. Hardware Configuration

The script is set up to use an Apple Silicon GPU (`mps`). If you are using a machine with an NVIDIA GPU, you need to change the `gpu_device` and `device_type` variables in both `src/tests/graph_cnn_train_loop.py` and `src/tests/graph_batch_test.py`.

In `src/tests/graph_cnn_train_loop.py`:
```python
# ...
# Replace mps with 'cuda' if using a Windows or Linux machine with a NVIDIA GPU
gpu_device = device('cuda')
device_type = 'cuda'
# ...
```

In `src/tests/graph_batch_test.py`:
```python
# ...
    # Train Dataset object
    train_dataset = ImageGraphFolderDataset(
        root_path=f"{img_path}/train",
        backbone=model,
        device=torch.device('cuda') # Change to 'cuda'
    )
# ...
    # Validation Dataset object
    val_dataset = ImageGraphFolderDataset(
        root_path=f"{img_path}/val",
        backbone=model,
        device=torch.device('cuda') # Change to 'cuda'
    )
# ...
```

### 4.2. Log and Model Save Paths

The training script saves logs and the trained model to specific directories. These paths are hardcoded in `src/tests/graph_cnn_train_loop.py`. It is recommended to change these paths to something more convenient.

For example, you can create `logs` and `saved_models` directories in the root of the project and update the paths as follows:

In `src/tests/graph_cnn_train_loop.py`:
```python
# ...
# Path to save the model
model_save_path = getcwd() + "/saved_models"

# Logging Setup
log_path = getcwd() + "/logs/gat_poincare_training_{time:YYYY-MM-DD_HH-mm-ss}.log"
# ...
```
Make sure these directories exist before running the training.

### 4.3. Running the Training

Once you have configured the paths and the device, you can start the training by running the following command from the root of the project:

```bash
python src/tests/graph_cnn_train_loop.py
```

The script will print the training and validation progress to the console and save the logs and the best model to the directories you specified.

### 4.4. Hyperparameters

The hyperparameters for the training (e.g., number of epochs, learning rate) are hardcoded in `src/tests/graph_cnn_train_loop.py`. You can modify them directly in the script to experiment with different settings.
