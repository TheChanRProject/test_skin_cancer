from torch.nn import Module, Identity
from timm import create_model

def load_timm_model(model_name: str = 'convnextv2_base.fcmae_ft_in22k_in1k_384') -> Module:

    model = create_model(model_name=model_name, pretrained=True)

    # Remove the classification head
    if hasattr(model, 'head'):
        # Replace with identity so you get features (before classifier)
        model.head = Identity()
    elif hasattr(model, 'fc'):
        model.fc = Identity()
    
    # Set the model to evaluation mode
    model.eval()

    return model
