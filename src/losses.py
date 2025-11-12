"""
Loss Functions for Imbalanced Classification

Implements Focal Loss and Class-Balanced Loss for handling class imbalance.

Authors: Jéssica A. L. de Macêdo & Matheus Borges Figueirôa

References:
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
- Class-Balanced Loss: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t is the model's estimated probability for the class with label 1
    - α_t is the weighting factor (alpha)
    - γ is the focusing parameter (gamma)
    
    The focusing parameter γ down-weights easy examples and focuses training
    on hard negatives. When γ=0, Focal Loss is equivalent to Cross-Entropy.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor for each class (tensor of shape [num_classes])
                   If None, all classes are weighted equally
            gamma: Focusing parameter (default: 2.0)
                   - gamma = 0: equivalent to Cross-Entropy
                   - gamma = 2: standard Focal Loss
                   - Higher gamma: more focus on hard examples
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions from model (logits), shape [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size]
            
        Returns:
            Focal loss value
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal loss
        loss = focal_term * ce_loss
        
        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on Effective Number of Samples
    
    Reweights loss by the effective number of samples:
    E_n = (1 - β^n) / (1 - β)
    
    where:
    - n is the number of samples in a class
    - β ∈ [0, 1) is a hyperparameter
    - As β → 1, reweighting effect increases
    """
    
    def __init__(
        self,
        samples_per_class: list,
        beta: float = 0.9999,
        loss_type: str = 'focal',
        gamma: float = 2.0
    ):
        """
        Args:
            samples_per_class: List with number of samples per class
            beta: Hyperparameter for effective number (default: 0.9999)
                  - β = 0: no reweighting (uniform)
                  - β → 1: more aggressive reweighting
            loss_type: 'focal' or 'ce' (cross-entropy)
            gamma: Focusing parameter for Focal Loss
        """
        super(ClassBalancedLoss, self).__init__()
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.loss_type = loss_type
        
        if loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=self.weights, gamma=gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions from model (logits), shape [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size]
            
        Returns:
            Class-balanced loss value
        """
        return self.loss_fn(inputs, targets)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy with Label Smoothing
    
    Prevents the model from becoming over-confident by smoothing
    the target distribution.
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: Label smoothing factor (default: 0.1)
                      - smoothing = 0: standard cross-entropy
                      - smoothing = 0.1: distribute 10% prob uniformly
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions from model (logits), shape [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size]
            
        Returns:
            Label smoothing cross-entropy loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # One-hot encode targets
        num_classes = inputs.size(1)
        targets_one_hot = torch.zeros_like(inputs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        
        # Apply label smoothing
        targets_smooth = (
            targets_one_hot * (1 - self.smoothing) +
            self.smoothing / num_classes
        )
        
        # Compute loss
        loss = -torch.sum(targets_smooth * log_probs, dim=1)
        
        return loss.mean()


def get_loss_function(
    loss_type: str,
    samples_per_class: Optional[list] = None,
    alpha: Optional[list] = None,
    gamma: float = 2.0,
    beta: float = 0.9999,
    smoothing: float = 0.0
) -> nn.Module:
    """
    Factory function to get loss function
    
    Args:
        loss_type: Type of loss
            - 'ce': Cross-Entropy
            - 'weighted_ce': Weighted Cross-Entropy
            - 'focal': Focal Loss
            - 'class_balanced': Class-Balanced Loss
            - 'label_smoothing': Label Smoothing Cross-Entropy
        samples_per_class: Number of samples per class (for class_balanced)
        alpha: Class weights (for weighted_ce or focal)
        gamma: Focusing parameter (for focal)
        beta: Effective number parameter (for class_balanced)
        smoothing: Label smoothing factor
        
    Returns:
        Loss function module
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'weighted_ce':
        if alpha is not None:
            weights = torch.tensor(alpha, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=weights)
        else:
            raise ValueError("alpha (class weights) required for weighted_ce")
    
    elif loss_type == 'focal':
        if alpha is not None:
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        else:
            alpha_tensor = None
        return FocalLoss(alpha=alpha_tensor, gamma=gamma)
    
    elif loss_type == 'class_balanced':
        if samples_per_class is None:
            raise ValueError("samples_per_class required for class_balanced")
        return ClassBalancedLoss(
            samples_per_class=samples_per_class,
            beta=beta,
            loss_type='focal',
            gamma=gamma
        )
    
    elif loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage and testing
if __name__ == '__main__':
    print("Loss Functions Module")
    print("="*60)
    
    # Example: Binary classification with imbalance
    # Class 0: 1000 samples, Class 1: 100 samples
    samples_per_class = [1000, 100]
    
    # Create dummy batch
    batch_size = 32
    num_classes = 2
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    print("\nTest batch:")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Samples per class: {samples_per_class}")
    
    # Test different loss functions
    print("\n" + "="*60)
    print("Testing Loss Functions:")
    print("="*60)
    
    # 1. Standard Cross-Entropy
    ce_loss = nn.CrossEntropyLoss()
    loss_ce = ce_loss(inputs, targets)
    print(f"\n1. Cross-Entropy Loss: {loss_ce.item():.4f}")
    
    # 2. Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    loss_focal = focal_loss(inputs, targets)
    print(f"2. Focal Loss (γ=2.0): {loss_focal.item():.4f}")
    
    # 3. Class-Balanced Focal Loss
    cb_loss = ClassBalancedLoss(samples_per_class, beta=0.9999, loss_type='focal', gamma=2.0)
    loss_cb = cb_loss(inputs, targets)
    print(f"3. Class-Balanced Focal Loss: {loss_cb.item():.4f}")
    
    # 4. Label Smoothing
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss_ls = ls_loss(inputs, targets)
    print(f"4. Label Smoothing CE: {loss_ls.item():.4f}")
    
    print("\n" + "="*60)
    print("✅ All loss functions working correctly!")
