import torch.nn as nn
from os import getcwd
from loguru import logger
from torch import device, mps, argmax, no_grad, save
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR  # or another scheduler
from torchmetrics.classification import Accuracy, F1Score, MatthewsCorrCoef

# Custom
import path_resolve
from models.gat_poincare_regularized_model import ImageGraphHyperbolicGATClassifier
from graph_batch_test import train_loader, val_loader
from util.format_util import line_separator
from util.focal_loss import FocalLoss

# Compatible for training on a Mac
## Replace mps with 'cuda' if using a Windows or Linux machine with a NVIDIA GPU
gpu_device = device('mps')
device_type = 'mps'

# Number of classes to distinguish
num_classes = 7

# Set classes to 3 if testing on Iris data
model = ImageGraphHyperbolicGATClassifier(in_feature_size=1026, num_classes=num_classes, device=device_type).to(gpu_device)
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # example: decay LR every 10 epochs
num_epochs = 50


# Learning rate warm-up scheduler: ramp up for warmup_epochs, then apply StepLR decay
warmup_epochs = 5
total_epochs = num_epochs
base_lr = 1e-3

def lr_lambda(current_epoch):
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / float(warmup_epochs)
    else:
        return 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
decay_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


# criterion = nn.CrossEntropyLoss().to(gpu_device)

criterion = FocalLoss(alpha=0.25).to(gpu_device)

# Metrics for training
train_acc  = Accuracy(task="multiclass", num_classes=num_classes).to(gpu_device)
train_f1   = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(gpu_device)
train_mcc  = MatthewsCorrCoef(task="multiclass", num_classes=num_classes).to(gpu_device)

# Metrics for validation
val_acc    = Accuracy(task="multiclass", num_classes=num_classes).to(gpu_device)
val_f1     = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(gpu_device)
val_mcc    = MatthewsCorrCoef(task="multiclass", num_classes=num_classes).to(gpu_device)

best_val_loss = float('inf')

# Path to save the model
model_save_path = getcwd() + "/data/medium_publications/agentic_cv_skin_cancer/skin_cancer/saved_models/convnext_gat_regularized_poincare"

patience = 3
epochs_no_improve = 0


"""
Logging Setup
"""

log_path = getcwd() + "/data/medium_publications/agentic_cv_skin_cancer/skin_cancer/logs/convnext_gat_regularized_poincare/gat_poincare_training_{time:YYYY-MM-DD_HH-mm-ss}.log"

logger.remove()  # remove default stderr handler if you don’t want duplicate

# Configuration for the logger
logger.add(
    log_path,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    rotation="10 MB",        # rotate when file grows beyond size
    retention="7 days",      # keep past logs for 7 days
    compression="zip"        # compress older logs to save space
)

# Initiating the training process
logger.info("Training started: device={}", device)

line_separator()

print("Training process starting.")

line_separator()


# Main Process

if __name__ == '__main__':
    for epoch in range(num_epochs):
        ### --- Training Phase ---
        model.train()
        train_acc.reset()
        train_f1.reset()
        train_mcc.reset()
        running_loss = 0.0
        n_batches = 0

        for data in train_loader:   # using PyG DataLoader
            
            # Get Data on mps
            data = data.to(gpu_device)

            optimizer.zero_grad()

            print(type(data))
        
            logits, z = model(data)
            
            print(f"Predictions received for training epoch: {epoch}")

            print(data.y.shape)

            loss = criterion(logits, data.y)

            print(f"Training loss: {loss}")

            loss.backward()
            optimizer.step()

            # Update running loss & metrics
            running_loss += loss.item()
            n_batches += 1

            preds = argmax(logits, dim=1)
            train_acc.update(preds, data.y)
            train_f1.update(preds, data.y)
            train_mcc.update(preds, data.y)

            print(f"Metrics updated at epoch: {epoch}")

        epoch_train_loss = running_loss / n_batches
        epoch_train_acc  = train_acc.compute().item()
        epoch_train_f1   = train_f1.compute().item()
        epoch_train_mcc  = train_mcc.compute().item()

        # Logging
        print("Epoch {}/{} => Train loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, mcc: {:.4f}".format(epoch+1, total_epochs,
            epoch_train_loss, epoch_train_acc, epoch_train_f1, epoch_train_mcc))


        logger.info(
            "Epoch {}/{} => Train loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, mcc: {:.4f}"
            ,
            epoch+1, total_epochs,
            epoch_train_loss, epoch_train_acc, epoch_train_f1, epoch_train_mcc
        )

        # Scheduler steps
        warmup_scheduler.step()
        if epoch >= warmup_epochs:
            decay_scheduler.step()
        
        print("Starting validation phase: ")
        line_separator()

        ### --- Validation Phase ---
        model.eval()
        val_acc.reset()
        val_f1.reset()
        val_mcc.reset()
        val_running_loss = 0.0
        val_n_batches = 0

        with no_grad():
            for data in val_loader:
                
                data = data.to(gpu_device)

                logits, z = model(data)
                print(f"Predictions received for validation epoch: {epoch}")
                
                loss = criterion(logits, data.y)
                print(f"Validation loss: {loss}")

                val_running_loss += loss.item()
                val_n_batches += 1

                preds = argmax(logits, dim=1)
                val_acc.update(preds, data.y)
                val_f1.update(preds, data.y)
                val_mcc.update(preds, data.y)
                print(f"Metrics updated for validation at epoch: {epoch}")

        epoch_val_loss = val_running_loss / val_n_batches
        epoch_val_acc  = val_acc.compute().item()
        epoch_val_f1   = val_f1.compute().item()
        epoch_val_mcc  = val_mcc.compute().item()

        # Step learning rate scheduler
        scheduler.step()

        # Print summary of this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train   → Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, F1: {epoch_train_f1:.4f}, MCC: {epoch_train_mcc:.4f}")
        print(f"  Val     → Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, F1: {epoch_val_f1:.4f}, MCC: {epoch_val_mcc:.4f}")


        # Logging
        logger.info(
            "Epoch {}/{} => Train loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, mcc: {:.4f} | "
            "Val loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, mcc: {:.4f}",
            epoch+1, total_epochs,
            epoch_train_loss, epoch_train_acc, epoch_train_f1, epoch_train_mcc,
            epoch_val_loss, epoch_val_acc, epoch_val_f1, epoch_val_mcc
        )

        print(f"Training and validation finished at epoch: {epoch}")

        line_separator()

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            save(model.state_dict(), f"{model_save_path}/best_model_early_stopping.pth")
            logger.info("Saved new best model at epoch {}", epoch+1)
        else:
            epochs_no_improve += 1
            logger.warning("No improvement in val loss for {} epochs", epochs_no_improve)
            if epochs_no_improve >= patience:
                logger.error("Early stopping triggered at epoch {}", epoch+1)
                break

        # Save best model based on validation loss (or another metric)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save(model.state_dict(), f"{model_save_path}/best_model.pth")
            print("  → Saved new best model")
        
        logger.info("Training finished")
