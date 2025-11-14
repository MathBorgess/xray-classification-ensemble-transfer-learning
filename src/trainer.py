"""
Training and evaluation functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


class EarlyStopping:
    """Early stopping handler"""

    def __init__(self, patience: int = 5, mode: str = 'min', delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            mode: 'min' or 'max' for metric comparison
            delta: Minimum change to consider as improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop

        Args:
            score: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.delta
        else:
            improved = score > self.best_score + self.delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model

    Args:
        model: PyTorch model
        data_loader: Data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Dictionary of metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Probability of positive class
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Calculate metrics
    loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(
        all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds,
                          average='binary', zero_division=0)

    # Calculate sensitivity and specificity
    tn = sum((p == 0 and l == 0) for p, l in zip(all_preds, all_labels))
    tp = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
    fn = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))
    fp = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    training_stage: str = 'baseline'
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train model with configuration

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to train on
        training_stage: Training stage ('baseline', 'stage_1', 'stage_2')

    Returns:
        Tuple of (trained_model, history)
    """
    training_config = config.get('training', {})

    # Get stage-specific configuration
    if training_stage == 'baseline':
        stage_config = training_config.get('baseline', {})
    elif training_stage == 'stage_1':
        stage_config = training_config.get(
            'progressive_unfreezing', {}).get('stage_1', {})
    elif training_stage == 'stage_2':
        stage_config = training_config.get(
            'progressive_unfreezing', {}).get('stage_2', {})
    else:
        stage_config = training_config.get('baseline', {})

    epochs = stage_config.get('epochs', 15)
    learning_rate = stage_config.get('learning_rate', 0.001)

    # Setup optimizer and loss
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    # Setup loss function based on configuration
    loss_config = training_config.get('loss', {})
    loss_type = loss_config.get('type', 'weighted_ce')

    if loss_type == 'focal' or loss_type == 'class_balanced':
        # Import Focal Loss
        from src.losses import get_loss_function

        # Calculate class weights or samples per class
        from src.data_loader import calculate_class_weights
        train_labels = [label for _, label in train_loader.dataset]

        if loss_type == 'focal':
            # Get class weights
            class_weights = calculate_class_weights(
                train_labels).cpu().numpy().tolist()
            criterion = get_loss_function(
                loss_type='focal',
                alpha=class_weights,
                gamma=loss_config.get('focal_gamma', 2.0)
            ).to(device)
        else:  # class_balanced
            # Count samples per class
            samples_per_class = [train_labels.count(i) for i in range(2)]
            criterion = get_loss_function(
                loss_type='class_balanced',
                samples_per_class=samples_per_class,
                beta=loss_config.get('class_balanced_beta', 0.9999),
                gamma=loss_config.get('focal_gamma', 2.0)
            ).to(device)
    elif training_config.get('use_class_weights', True):
        # Use weighted cross-entropy (original)
        from src.data_loader import calculate_class_weights
        train_labels = [label for _, label in train_loader.dataset]
        class_weights = calculate_class_weights(train_labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        # Standard cross-entropy
        criterion = nn.CrossEntropyLoss()

    # Early stopping
    early_stopping_config = training_config.get('early_stopping', {})
    early_stopping = EarlyStopping(
        patience=early_stopping_config.get('patience', 5),
        mode=early_stopping_config.get('mode', 'min')
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }

    best_val_auc = 0.0
    best_model_state = None

    print(f"\nTraining stage: {training_stage}")
    print(f"Epochs: {epochs}, Learning rate: {learning_rate}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = model.state_dict().copy()

        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest Validation AUC: {best_val_auc:.4f}")

    return model, history
