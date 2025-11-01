"""
Model architectures for Transfer Learning
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Any


class TransferLearningModel(nn.Module):
    """
    Base Transfer Learning Model with customizable backbone
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super(TransferLearningModel, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove original classifier
        )

        # Get number of features from backbone
        self.num_features = self.backbone.num_features

        # Create custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int = None):
        """
        Unfreeze backbone parameters

        Args:
            num_layers: Number of layers to unfreeze from the end
        """
        if num_layers is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last n layers
            all_params = list(self.backbone.parameters())
            for param in all_params[-num_layers:]:
                param.requires_grad = True


def create_model(
    architecture: str,
    config: Dict[str, Any],
    pretrained: bool = True
) -> TransferLearningModel:
    """
    Factory function to create models

    Args:
        architecture: Model architecture name
        config: Configuration dictionary
        pretrained: Whether to use pretrained weights

    Returns:
        TransferLearningModel instance
    """
    model_config = config.get('models', {})
    num_classes = model_config.get('num_classes', 2)
    dropout = model_config.get('dropout', 0.5)

    # Map architecture names to timm model names
    architecture_mapping = {
        'efficientnet_b0': 'efficientnet_b0',
        'resnet50': 'resnet50',
        'densenet121': 'densenet121'
    }

    model_name = architecture_mapping.get(architecture, architecture)

    model = TransferLearningModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )

    return model


class EnsembleModel(nn.Module):
    """
    Ensemble model combining multiple base models
    """

    def __init__(
        self,
        models: list,
        weights: list = None,
        method: str = 'simple_voting'
    ):
        """
        Args:
            models: List of trained models
            weights: List of weights for weighted voting
            method: Ensemble method ('simple_voting' or 'weighted_voting')
        """
        super(EnsembleModel, self).__init__()

        self.models = nn.ModuleList(models)
        self.method = method

        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            self.weights = weights_tensor / weights_tensor.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble

        Args:
            x: Input tensor

        Returns:
            Ensemble predictions
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        # Stack predictions
        predictions = torch.stack(predictions)

        # Apply ensemble method
        if self.method == 'simple_voting':
            # Simple average
            ensemble_pred = predictions.mean(dim=0)
        elif self.method == 'weighted_voting':
            # Weighted average
            weights = self.weights.view(-1, 1, 1).to(predictions.device)
            ensemble_pred = (predictions * weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

        return ensemble_pred

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions

        Args:
            x: Input tensor

        Returns:
            Probability predictions
        """
        logits = self.forward(x)
        probas = torch.softmax(logits, dim=1)
        return probas


def create_ensemble(
    models: list,
    val_metrics: list = None,
    config: Dict[str, Any] = None
) -> EnsembleModel:
    """
    Create ensemble model from trained models

    Args:
        models: List of trained models
        val_metrics: List of validation metrics for weighted voting
        config: Configuration dictionary

    Returns:
        EnsembleModel instance
    """
    if config is None:
        method = 'simple_voting'
        weights = None
    else:
        ensemble_config = config.get('ensemble', {})
        methods = ensemble_config.get('methods', ['simple_voting'])
        method = methods[0] if methods else 'simple_voting'

        # Calculate weights based on validation AUC
        if method == 'weighted_voting' and val_metrics is not None:
            weight_metric = ensemble_config.get('weight_metric', 'auc')
            weights = [m.get(weight_metric, 1.0) for m in val_metrics]
        else:
            weights = None

    ensemble = EnsembleModel(
        models=models,
        weights=weights,
        method=method
    )

    return ensemble
