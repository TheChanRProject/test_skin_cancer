from os import getcwd

# Custom
import path_resolve
from util.image_augmentation import generate_augmented_dataset

# Source root
train_source_root = getcwd() + "/data/medium_publications/agentic_cv_skin_cancer/skin_cancer/skin_cancer_images_dst/train"
val_source_root = getcwd() + "/data/medium_publications/agentic_cv_skin_cancer/skin_cancer/skin_cancer_images_dst/val"

# Output root
output_train_root = getcwd() + "/data/medium_publications/agentic_cv_skin_cancer/skin_cancer/skin_cancer_images_augmented/train"
output_val_root = getcwd() + "/data/medium_publications/agentic_cv_skin_cancer/skin_cancer/skin_cancer_images_augmented/val"


# # Training Set
# train_augmented_result = generate_augmented_dataset(source_root=train_source_root, output_root=output_train_root)

# print(train_augmented_result)

# Validation Set
val_augmented_result = generate_augmented_dataset(source_root=val_source_root, output_root=output_val_root)

print(val_augmented_result)