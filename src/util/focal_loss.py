"""
Focal Loss Implementation for PyTorch

Focal Loss was introduced in "Focal Loss for Dense Object Detection" (Lin et al., 2017)
to address class imbalance by down-weighting easy examples and focusing on hard negatives.

The focal loss adds a modulating factor (1 - p_t)^gamma to the cross-entropy loss,
where p_t is the model's estimated probability for the true class.

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Key improvements over cross-entropy:
- Reduces loss contribution from easy examples (high p_t)
- Focuses training on hard, misclassified examples
- alpha parameter balances positive/negative examples
- gamma parameter controls the rate at which easy examples are down-weighted
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Args:
        alpha (float or torch.Tensor, optional): Weighting factor in [0, 1] to balance
            positive vs negative examples, or a tensor of weights for each class.
            Default: None (no class balancing)
        gamma (float): Focusing parameter for modulating loss. Higher gamma increases
            the focus on hard examples. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
        ignore_index (int): Specifies a target class to ignore. Default: -100

    Shape:
        - Input: (N, C) where N is batch size and C is number of classes, or
                 (N, C, d1, d2, ..., dk) for k-dimensional loss
        - Target: (N) where each value is 0 <= targets[i] <= C-1, or
                  (N, d1, d2, ..., dk) for k-dimensional loss
        - Output: scalar if reduction is 'mean' or 'sum', otherwise same shape as target

    Examples:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> inputs = torch.randn(8, 5, requires_grad=True)  # 8 samples, 5 classes
        >>> targets = torch.randint(0, 5, (8,))
        >>> loss = criterion(inputs, targets)
        >>> loss.backward()
    """

    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super(FocalLoss, self).__init__()

        if alpha is not None and not isinstance(alpha, (float, torch.Tensor)):
            raise TypeError(f"alpha must be float or torch.Tensor, got {type(alpha)}")

        if not isinstance(gamma, (int, float)):
            raise TypeError(f"gamma must be a number, got {type(gamma)}")

        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing focal loss.

        Args:
            inputs: Raw logits from the model (before softmax)
            targets: Ground truth class indices

        Returns:
            Computed focal loss
        """
        # Compute cross entropy loss (log softmax + nll loss)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            ignore_index=self.ignore_index
        )

        # Get probabilities
        p = F.softmax(inputs, dim=1)

        # Get the probability of the true class for each sample
        # Gather the probabilities at the target indices
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal loss modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight to cross entropy loss
        focal_loss = focal_weight * ce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Single alpha value for all classes
                alpha_t = self.alpha
            else:
                # Per-class alpha values
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                alpha_t = self.alpha.gather(0, targets)

            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            # Only average over non-ignored elements
            if self.ignore_index >= 0:
                valid_mask = targets != self.ignore_index
                return focal_loss[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=inputs.device)
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # reduction == 'none'
            return focal_loss


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Args:
        alpha (float, optional): Weighting factor for the positive class in [0, 1].
            Default: 0.25
        gamma (float): Focusing parameter. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'

    Shape:
        - Input: (N, *) where * means any number of additional dimensions
        - Target: (N, *) same shape as input
        - Output: scalar if reduction is 'mean' or 'sum', otherwise same shape as input

    Examples:
        >>> criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        >>> inputs = torch.randn(8, 1, requires_grad=True)
        >>> targets = torch.randint(0, 2, (8, 1)).float()
        >>> loss = criterion(inputs, targets)
        >>> loss.backward()
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(BinaryFocalLoss, self).__init__()

        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be a number, got {type(alpha)}")

        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        if not isinstance(gamma, (int, float)):
            raise TypeError(f"gamma must be a number, got {type(gamma)}")

        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing binary focal loss.

        Args:
            inputs: Raw logits from the model
            targets: Ground truth binary labels (0 or 1)

        Returns:
            Computed focal loss
        """
        # Compute probabilities
        p = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction='none'
        )

        # Compute p_t: probability of true class
        p_t = p * targets + (1 - p) * (1 - targets)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Compute focal loss
        focal_loss = focal_weight * bce_loss

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # reduction == 'none'
            return focal_loss


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[Union[float, torch.Tensor]] = None,
    gamma: float = 2.0,
    reduction: str = 'mean',
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Functional interface for Focal Loss.

    Args:
        inputs: Raw logits from the model (N, C) where C is number of classes
        targets: Ground truth class indices (N,)
        alpha: Weighting factor for class balance
        gamma: Focusing parameter
        reduction: Reduction method: 'none' | 'mean' | 'sum'
        ignore_index: Target class to ignore

    Returns:
        Computed focal loss

    Examples:
        >>> inputs = torch.randn(8, 5, requires_grad=True)
        >>> targets = torch.randint(0, 5, (8,))
        >>> loss = focal_loss(inputs, targets, alpha=0.25, gamma=2.0)
    """
    criterion = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index
    )
    return criterion(inputs, targets)


# if __name__ == "__main__":
#     # Example usage and testing
#     print("Testing Focal Loss Implementation\n")

#     # Multi-class example
#     print("Multi-class Focal Loss:")
#     num_samples = 16
#     num_classes = 5

#     inputs = torch.randn(num_samples, num_classes, requires_grad=True)
#     targets = torch.randint(0, num_classes, (num_samples,))

#     # Create focal loss with default parameters
#     criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
#     loss_focal = criterion_focal(inputs, targets)

#     # Compare with standard cross entropy
#     criterion_ce = nn.CrossEntropyLoss()
#     loss_ce = criterion_ce(inputs, targets)

#     print(f"  Focal Loss: {loss_focal.item():.4f}")
#     print(f"  Cross Entropy Loss: {loss_ce.item():.4f}")
#     print(f"  Ratio (FL/CE): {(loss_focal / loss_ce).item():.4f}\n")

#     # Binary classification example
#     print("Binary Focal Loss:")
#     inputs_binary = torch.randn(num_samples, 1, requires_grad=True)
#     targets_binary = torch.randint(0, 2, (num_samples, 1)).float()

#     criterion_binary = BinaryFocalLoss(alpha=0.25, gamma=2.0)
#     loss_binary = criterion_binary(inputs_binary, targets_binary)

#     bce_loss = F.binary_cross_entropy_with_logits(inputs_binary, targets_binary)

#     print(f"  Binary Focal Loss: {loss_binary.item():.4f}")
#     print(f"  Binary Cross Entropy Loss: {bce_loss.item():.4f}")
#     print(f"  Ratio (FL/BCE): {(loss_binary / bce_loss).item():.4f}\n")

#     # Test with class weights
#     print("Focal Loss with per-class weights:")
#     class_weights = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])
#     criterion_weighted = FocalLoss(alpha=class_weights, gamma=2.0)
#     loss_weighted = criterion_weighted(inputs, targets)
#     print(f"  Weighted Focal Loss: {loss_weighted.item():.4f}\n")

#     # Backward pass test
#     print("Testing backward pass:")
#     loss_focal.backward()
#     print(f"  Gradient shape: {inputs.grad.shape}")
#     print(f"  Gradient norm: {inputs.grad.norm().item():.4f}")
#     print("\nAll tests passed!")
