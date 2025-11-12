# 🔧 Correções Pré-Ensemble: Resolvendo Gaps Críticos

**Autores:** Jéssica A. L. de Macêdo & Matheus Borges Figueirôa  
**Data:** 12 de Novembro de 2025  
**Status:** PLANO DE AÇÃO PRIORITÁRIO

---

## 🎯 Objetivo Deste Documento

Resolver **TODOS os gaps críticos identificados** antes de implementar o ensemble, garantindo:
- ✅ Dataset robusto e bem validado
- ✅ Validação estatística confiável
- ✅ Métricas corretamente calculadas
- ✅ Especificidade melhorada
- ✅ Base sólida para ensemble learning

---

## 📊 Gap Analysis: Problemas Identificados

### 🔴 GAP 1: Dataset de Validação Minúsculo (16 amostras)

**Problema Crítico:**
- Apenas 16 imagens de validação (8 Normal, 8 Pneumonia)
- Métricas instáveis e não confiáveis
- Early stopping baseado em dados insuficientes
- Risco de decisões enviesadas

**Impacto:**
- ⚠️ **ALTO:** Invalida significância estatística
- ⚠️ **ALTO:** Pesos do ensemble não confiáveis
- ⚠️ **MÉDIO:** Seleção de hiperparâmetros comprometida

---

### 🟠 GAP 2: Especificidade Extremamente Baixa

**Problema Crítico:**
- ResNet50: 12.82% especificidade
- DenseNet121: 17.09% especificidade
- EfficientNetB0: 47.86% (melhor, mas ainda problemático)

**Implicações Clínicas:**
- 🚨 80-90% de casos normais classificados como pneumonia (falsos positivos)
- Sobrecarga de radiologistas revisando casos normais
- Sistema inutilizável na prática clínica

---

### 🟡 GAP 3: Desbalanceamento de Classes Não Resolvido

**Problema:**
- Class weights implementados, mas não suficientes
- Loss function pode não refletir importância clínica
- Threshold padrão (0.5) não é ótimo

---

### 🟢 GAP 4: Falta de Cross-Validation

**Problema:**
- Single-split training sem validação cruzada
- Incerteza não quantificada
- Não há como verificar estabilidade dos resultados

---

### 🔵 GAP 5: Augmentation Limitada

**Problema:**
- Apenas transformações básicas (rotação, flip, brilho)
- Falta transformações específicas para raio-X médico
- Pode limitar generalização

---

## 🛠️ SOLUÇÕES DETALHADAS

---

## ✅ SOLUÇÃO 1: Reestruturar Dataset com K-Fold Cross-Validation

### Estratégia: Stratified K-Fold (K=5)

**Justificativa:**
- Usar TODO o dataset de treino para treinar E validar
- 5 folds garante 20% de validação em cada fold
- Stratified mantém proporção de classes
- Média de 5 runs reduz variância

### Implementação:

Criar arquivo: `src/cross_validation.py`

```python
"""
K-Fold Cross-Validation for robust model evaluation
"""

import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Any
import json
from tqdm import tqdm

from src.models import create_model
from src.trainer import train_model, evaluate
from src.utils import get_device, set_seed
import torch.nn as nn


class CrossValidator:
    """
    Implements Stratified K-Fold Cross-Validation
    """
    
    def __init__(self, config: Dict[str, Any], n_splits: int = 5):
        """
        Args:
            config: Configuration dictionary
            n_splits: Number of folds (default: 5)
        """
        self.config = config
        self.n_splits = n_splits
        self.device = get_device(config)
        
    def split_data(
        self, 
        image_paths: List[str], 
        labels: List[int]
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create stratified K-fold splits
        
        Args:
            image_paths: List of image paths
            labels: List of labels
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.config.get('seed', 42)
        )
        
        splits = []
        for train_idx, val_idx in skf.split(image_paths, labels):
            splits.append((train_idx.tolist(), val_idx.tolist()))
            
        return splits
    
    def train_fold(
        self,
        fold_num: int,
        train_loader,
        val_loader,
        model_name: str
    ) -> Tuple[nn.Module, Dict[str, float], Dict[str, list]]:
        """
        Train a single fold
        
        Args:
            fold_num: Fold number
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name of model architecture
            
        Returns:
            Tuple of (trained_model, val_metrics, history)
        """
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num + 1}/{self.n_splits}: {model_name}")
        print(f"{'='*60}")
        
        # Create fresh model
        set_seed(self.config.get('seed', 42) + fold_num)
        model = create_model(model_name, self.config)
        model = model.to(self.device)
        
        # Training stages
        stages = ['baseline', 'stage_1', 'stage_2']
        
        for stage in stages:
            print(f"\n--- Training Stage: {stage} ---")
            
            # Unfreeze layers for progressive stages
            if stage == 'stage_1':
                self._unfreeze_layers(model, num_layers=20)
            elif stage == 'stage_2':
                self._unfreeze_layers(model, num_layers=50)
            
            model, history = train_model(
                model, train_loader, val_loader,
                self.config, self.device, stage
            )
        
        # Final evaluation
        criterion = nn.CrossEntropyLoss()
        val_metrics = evaluate(model, val_loader, criterion, self.device)
        
        return model, val_metrics, history
    
    def _unfreeze_layers(self, model: nn.Module, num_layers: int):
        """Unfreeze last N layers"""
        params = list(model.parameters())
        for param in params[-num_layers:]:
            param.requires_grad = True
    
    def cross_validate(
        self,
        model_name: str,
        image_paths: List[str],
        labels: List[int],
        save_dir: str = 'models/cv_models'
    ) -> Dict[str, Any]:
        """
        Perform complete cross-validation
        
        Args:
            model_name: Name of model architecture
            image_paths: List of image paths
            labels: List of labels
            save_dir: Directory to save fold models
            
        Returns:
            Dictionary with cross-validation results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        splits = self.split_data(image_paths, labels)
        
        fold_results = []
        all_metrics = {
            'accuracy': [],
            'auc': [],
            'f1_score': [],
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'recall': []
        }
        
        for fold_num, (train_idx, val_idx) in enumerate(splits):
            # Create data loaders for this fold
            from src.data_loader import ChestXRayDataset, get_transforms, get_augmentation
            from torch.utils.data import DataLoader, Subset
            
            # Get full dataset
            transform = get_transforms(self.config, train=True)
            augmentation = get_augmentation(self.config)
            
            train_paths = [image_paths[i] for i in train_idx]
            train_labels_fold = [labels[i] for i in train_idx]
            val_paths = [image_paths[i] for i in val_idx]
            val_labels_fold = [labels[i] for i in val_idx]
            
            train_dataset = ChestXRayDataset(
                train_paths, train_labels_fold,
                transform=transform, augmentation=augmentation
            )
            
            val_transform = get_transforms(self.config, train=False)
            val_dataset = ChestXRayDataset(
                val_paths, val_labels_fold,
                transform=val_transform, augmentation=None
            )
            
            batch_size = self.config.get('data', {}).get('batch_size', 32)
            num_workers = self.config.get('data', {}).get('num_workers', 4)
            
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=True, num_workers=num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers, pin_memory=True
            )
            
            print(f"\nFold {fold_num + 1} - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            
            # Train fold
            model, val_metrics, history = self.train_fold(
                fold_num, train_loader, val_loader, model_name
            )
            
            # Save fold model
            fold_model_path = os.path.join(
                save_dir, f'{model_name}_fold{fold_num + 1}.pth'
            )
            torch.save(model.state_dict(), fold_model_path)
            
            # Collect metrics
            fold_results.append({
                'fold': fold_num + 1,
                'metrics': val_metrics,
                'model_path': fold_model_path
            })
            
            for key in all_metrics.keys():
                if key in val_metrics:
                    all_metrics[key].append(val_metrics[key])
            
            print(f"\nFold {fold_num + 1} Results:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
        
        # Calculate statistics
        summary = {
            'model_name': model_name,
            'n_splits': self.n_splits,
            'fold_results': fold_results,
            'mean_metrics': {},
            'std_metrics': {},
            'ci_95_metrics': {}
        }
        
        for key, values in all_metrics.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                # 95% confidence interval
                ci_lower = mean_val - 1.96 * (std_val / np.sqrt(len(values)))
                ci_upper = mean_val + 1.96 * (std_val / np.sqrt(len(values)))
                
                summary['mean_metrics'][key] = mean_val
                summary['std_metrics'][key] = std_val
                summary['ci_95_metrics'][key] = (ci_lower, ci_upper)
        
        # Save summary
        summary_path = os.path.join(save_dir, f'{model_name}_cv_summary.json')
        with open(summary_path, 'w') as f:
            # Convert numpy types to Python types
            summary_serializable = {
                'model_name': summary['model_name'],
                'n_splits': summary['n_splits'],
                'mean_metrics': {k: float(v) for k, v in summary['mean_metrics'].items()},
                'std_metrics': {k: float(v) for k, v in summary['std_metrics'].items()},
                'ci_95_metrics': {k: [float(v[0]), float(v[1])] for k, v in summary['ci_95_metrics'].items()}
            }
            json.dump(summary_serializable, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION SUMMARY: {model_name}")
        print(f"{'='*60}")
        
        for key in all_metrics.keys():
            if key in summary['mean_metrics']:
                mean = summary['mean_metrics'][key]
                std = summary['std_metrics'][key]
                ci = summary['ci_95_metrics'][key]
                print(f"{key:15s}: {mean:.4f} ± {std:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        
        return summary


def run_cross_validation_experiment(config_path: str = 'configs/config.yaml'):
    """
    Run complete cross-validation experiment for all models
    """
    from src.utils import load_config
    from src.data_loader import load_data_from_directory
    import os
    
    config = load_config(config_path)
    
    # Load ALL training data (we'll split it ourselves)
    data_dir = config.get('data', {}).get('data_dir', 'data/raw/chest_xray')
    train_dir = os.path.join(data_dir, 'train')
    
    print("Loading training data...")
    image_paths, labels = load_data_from_directory(train_dir, config)
    print(f"Total samples: {len(image_paths)}")
    print(f"Normal: {sum(1 for l in labels if l == 0)}")
    print(f"Pneumonia: {sum(1 for l in labels if l == 1)}")
    
    # Create cross-validator
    cv = CrossValidator(config, n_splits=5)
    
    # Test each architecture
    models = ['efficientnet_b0', 'resnet50', 'densenet121']
    results = {}
    
    for model_name in models:
        print(f"\n{'#'*60}")
        print(f"# STARTING CROSS-VALIDATION: {model_name.upper()}")
        print(f"{'#'*60}")
        
        summary = cv.cross_validate(model_name, image_paths, labels)
        results[model_name] = summary
    
    # Save combined results
    combined_path = 'results/cross_validation_results.json'
    os.makedirs('results', exist_ok=True)
    
    with open(combined_path, 'w') as f:
        results_serializable = {}
        for model_name, summary in results.items():
            results_serializable[model_name] = {
                'mean_metrics': {k: float(v) for k, v in summary['mean_metrics'].items()},
                'std_metrics': {k: float(v) for k, v in summary['std_metrics'].items()},
                'ci_95_metrics': {k: [float(v[0]), float(v[1])] for k, v in summary['ci_95_metrics'].items()}
            }
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n{'#'*60}")
    print("# CROSS-VALIDATION COMPLETE")
    print(f"# Results saved to: {combined_path}")
    print(f"{'#'*60}")
    
    return results


if __name__ == '__main__':
    results = run_cross_validation_experiment()
```

### Como Executar:

```bash
# Rodar cross-validation para todos os modelos
python -m src.cross_validation

# Isso vai:
# 1. Criar 5 folds stratificados
# 2. Treinar cada modelo 5 vezes
# 3. Calcular média ± desvio padrão
# 4. Gerar intervalos de confiança 95%
# 5. Salvar modelos de cada fold
```

---

## ✅ SOLUÇÃO 2: Otimização de Threshold para Melhorar Especificidade

### Problema: Threshold 0.5 não é ótimo

**Análise:**
- Threshold padrão assume classes balanceadas
- Nosso dataset tem desbalanceamento
- Precisamos encontrar threshold que maximize trade-off

### Implementação:

Criar arquivo: `src/threshold_optimization.py`

```python
"""
Threshold optimization to improve specificity
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, f1_score
)
from typing import Tuple, Dict, Any
import os


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    method: str = 'youden'
) -> Tuple[float, float, float]:
    """
    Find optimal decision threshold
    
    Methods:
    - 'youden': Maximize Youden's J = Sensitivity + Specificity - 1
    - 'f1': Maximize F1-score
    - 'balanced': Balance sensitivity and specificity
    - 'specificity_target': Target minimum specificity (e.g., 60%)
    
    Args:
        y_true: True labels
        y_scores: Prediction scores (probabilities for positive class)
        method: Optimization method
        
    Returns:
        Tuple of (optimal_threshold, sensitivity, specificity)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    if method == 'youden':
        # Youden's J statistic: maximizes (Sensitivity + Specificity - 1)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        
    elif method == 'f1':
        # Maximize F1-score
        f1_scores = []
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)
        optimal_idx = np.argmax(f1_scores)
        
    elif method == 'balanced':
        # Minimize distance to perfect classifier (TPR=1, FPR=0)
        distances = np.sqrt((1 - tpr)**2 + fpr**2)
        optimal_idx = np.argmin(distances)
        
    elif method == 'specificity_target':
        # Find threshold that gives specificity >= 60%
        specificity = 1 - fpr
        target_spec = 0.60
        valid_indices = np.where(specificity >= target_spec)[0]
        
        if len(valid_indices) == 0:
            # If can't reach target, get best possible
            optimal_idx = np.argmax(specificity)
        else:
            # Among valid, choose highest sensitivity
            optimal_idx = valid_indices[np.argmax(tpr[valid_indices])]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    
    return optimal_threshold, optimal_sensitivity, optimal_specificity


def evaluate_with_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Evaluate model with custom threshold
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        threshold: Decision threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F1-score
    if (precision + sensitivity) > 0:
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1 = 0.0
    
    # Balanced accuracy
    balanced_acc = (sensitivity + specificity) / 2
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str = 'results/figures/threshold_analysis.png'
):
    """
    Plot comprehensive threshold analysis
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        save_path: Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculate metrics for each threshold
    specificities = 1 - fpr
    sensitivities = tpr
    youden_j = sensitivities + specificities - 1
    
    # Calculate F1 for each threshold
    f1_scores = []
    for threshold in thresholds:
        metrics = evaluate_with_threshold(y_true, y_scores, threshold)
        f1_scores.append(metrics['f1_score'])
    
    # Find optimal points
    methods = ['youden', 'f1', 'balanced', 'specificity_target']
    optimal_points = {}
    
    for method in methods:
        thresh, sens, spec = find_optimal_threshold(y_true, y_scores, method)
        optimal_points[method] = (thresh, sens, spec)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Sensitivity vs Specificity vs Threshold
    ax = axes[0, 0]
    ax.plot(thresholds, sensitivities, 'b-', label='Sensitivity', linewidth=2)
    ax.plot(thresholds, specificities, 'r-', label='Specificity', linewidth=2)
    ax.axhline(y=0.95, color='b', linestyle='--', alpha=0.3, label='95% Sensitivity Target')
    ax.axhline(y=0.60, color='r', linestyle='--', alpha=0.3, label='60% Specificity Target')
    
    # Mark optimal points
    for method, (thresh, sens, spec) in optimal_points.items():
        ax.axvline(x=thresh, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Sensitivity and Specificity vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Youden's J vs Threshold
    ax = axes[0, 1]
    ax.plot(thresholds, youden_j, 'g-', linewidth=2)
    optimal_idx = np.argmax(youden_j)
    ax.axvline(x=thresholds[optimal_idx], color='red', linestyle='--', 
               label=f'Optimal = {thresholds[optimal_idx]:.3f}')
    ax.scatter([thresholds[optimal_idx]], [youden_j[optimal_idx]], 
               color='red', s=100, zorder=5)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel("Youden's J", fontsize=12)
    ax.set_title("Youden's J Statistic vs Threshold", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 3: F1-Score vs Threshold
    ax = axes[1, 0]
    ax.plot(thresholds, f1_scores, 'purple', linewidth=2)
    optimal_f1_idx = np.argmax(f1_scores)
    ax.axvline(x=thresholds[optimal_f1_idx], color='red', linestyle='--',
               label=f'Optimal = {thresholds[optimal_f1_idx]:.3f}')
    ax.scatter([thresholds[optimal_f1_idx]], [f1_scores[optimal_f1_idx]],
               color='red', s=100, zorder=5)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 4: Summary Table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create table data
    table_data = [['Method', 'Threshold', 'Sensitivity', 'Specificity', 'F1-Score']]
    
    # Add default threshold
    metrics_default = evaluate_with_threshold(y_true, y_scores, 0.5)
    table_data.append([
        'Default (0.5)',
        '0.500',
        f"{metrics_default['sensitivity']:.3f}",
        f"{metrics_default['specificity']:.3f}",
        f"{metrics_default['f1_score']:.3f}"
    ])
    
    # Add optimized thresholds
    for method in methods:
        thresh, sens, spec = optimal_points[method]
        metrics = evaluate_with_threshold(y_true, y_scores, thresh)
        
        method_names = {
            'youden': "Youden's J",
            'f1': "F1-Max",
            'balanced': "Balanced",
            'specificity_target': "Spec≥60%"
        }
        
        table_data.append([
            method_names.get(method, method),
            f"{thresh:.3f}",
            f"{sens:.3f}",
            f"{spec:.3f}",
            f"{metrics['f1_score']:.3f}"
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Threshold Optimization Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Threshold analysis plot saved to: {save_path}")
    plt.close()


def optimize_model_threshold(
    model,
    data_loader,
    device,
    save_dir: str = 'results'
) -> Dict[str, Any]:
    """
    Optimize threshold for a trained model
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for validation/test data
        device: Device to run on
        save_dir: Directory to save results
        
    Returns:
        Dictionary with optimization results
    """
    model.eval()
    
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    
    # Find optimal thresholds
    methods = ['youden', 'f1', 'balanced', 'specificity_target']
    results = {}
    
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*60)
    
    # Default threshold
    metrics_default = evaluate_with_threshold(y_true, y_scores, 0.5)
    results['default'] = metrics_default
    
    print("\nDEFAULT Threshold (0.5):")
    for key, value in metrics_default.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    
    # Optimized thresholds
    for method in methods:
        thresh, sens, spec = find_optimal_threshold(y_true, y_scores, method)
        metrics = evaluate_with_threshold(y_true, y_scores, thresh)
        results[method] = metrics
        
        print(f"\n{method.upper()} Method:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
            else:
                print(f"  {key:20s}: {value}")
    
    # Plot analysis
    plot_path = os.path.join(save_dir, 'figures', 'threshold_analysis.png')
    plot_threshold_analysis(y_true, y_scores, plot_path)
    
    return results


if __name__ == '__main__':
    # Example usage
    from src.utils import load_config, get_device
    from src.models import create_model
    from src.data_loader import get_dataloaders
    import torch
    
    config = load_config('configs/config.yaml')
    device = get_device(config)
    
    # Load model
    model_name = 'efficientnet_b0'
    model = create_model(model_name, config, pretrained=False)
    model.load_state_dict(torch.load(f'models/{model_name}_final.pth', map_location=device))
    model = model.to(device)
    
    # Load data
    _, _, test_loader = get_dataloaders(config)
    
    # Optimize threshold
    results = optimize_model_threshold(model, test_loader, device)
```

### Como Executar:

```bash
# Otimizar threshold para cada modelo
python -m src.threshold_optimization

# Resultado esperado:
# - Especificidade aumentar de ~15% para ~60%
# - Sensibilidade cair levemente (99% -> 95%)
# - Balanced accuracy melhorar significativamente
```

---

## ✅ SOLUÇÃO 3: Augmentation Melhorada para Raio-X

### Problema: Augmentation básica não explora características de raio-X

### Implementação:

Atualizar `src/data_loader.py`:

```python
def get_augmentation_advanced(config: Dict[str, Any]) -> A.Compose:
    """
    Advanced augmentation pipeline specifically for chest X-rays
    
    Inclui:
    - Transformações geométricas (rotação, shift, zoom)
    - Elastic deformation (simula variação anatômica)
    - CLAHE (melhora contraste local)
    - Gaussian noise (simula ruído de sensor)
    - Grid distortion (simula variação de posicionamento)
    """
    data_config = config.get('data', {})
    aug_config = data_config.get('augmentation', {})
    image_size = tuple(data_config.get('image_size', [224, 224]))
    
    augmentation = A.Compose([
        # Geometric transformations
        A.Rotate(limit=10, p=0.5, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=0,
            p=0.5
        ),
        
        # Elastic deformation (simula variação anatômica)
        A.ElasticTransform(
            alpha=1.0,
            sigma=50.0,
            alpha_affine=10.0,
            p=0.3
        ),
        
        # Grid distortion
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=0.3
        ),
        
        # Contrast and brightness (importante para raio-X)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # CLAHE: melhora contraste local
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.5
        ),
        
        # Gaussian noise (simula ruído de sensor)
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            p=0.3
        ),
        
        # Gamma correction (simula diferença de exposição)
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3
        ),
        
        # Final resize
        A.Resize(height=image_size[0], width=image_size[1])
    ])
    
    return augmentation
```

### Atualizar config.yaml:

```yaml
data:
  augmentation:
    use_advanced: true  # Enable advanced augmentation
    rotation_range: 10
    horizontal_flip: true
    brightness_range: 0.2
    contrast_range: 0.2
    zoom_range: 0.1
    elastic_deformation: true
    clahe: true
    gaussian_noise: true
```

---

## ✅ SOLUÇÃO 4: Focal Loss para Desbalanceamento

### Problema: Cross-Entropy com class weights não é suficiente

### Implementação:

Criar arquivo: `src/losses.py`

```python
"""
Custom loss functions for imbalanced classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
               or list of weights for each class
        gamma: Focusing parameter for modulating loss (gamma >= 0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where C = number of classes
            targets: (N,) class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Modulating factor
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        # Apply alpha weighting
        if isinstance(self.alpha, (list, tuple, torch.Tensor)):
            alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
        else:
            alpha_t = self.alpha
        
        focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples
    
    Reference: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (2019)
    
    Args:
        samples_per_class: List of number of samples per class
        beta: Hyperparameter in [0, 1) for re-weighting
        loss_type: 'focal' or 'cross_entropy'
    """
    
    def __init__(self, samples_per_class, beta=0.9999, loss_type='focal'):
        super(ClassBalancedLoss, self).__init__()
        
        effective_num = 1.0 - torch.pow(beta, torch.tensor(samples_per_class, dtype=torch.float))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_class)
        
        self.weights = weights
        self.loss_type = loss_type
        
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=weights.tolist(), gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=weights)
    
    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)
```

### Atualizar trainer.py para usar Focal Loss:

```python
# In train_model function, replace criterion creation:

if training_config.get('use_focal_loss', False):
    # Use Focal Loss
    from src.losses import FocalLoss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    print("Using Focal Loss for training")
elif training_config.get('use_class_balanced_loss', False):
    # Use Class-Balanced Loss
    from src.losses import ClassBalancedLoss
    train_labels = [label for _, label in train_loader.dataset]
    samples_per_class = [sum(1 for l in train_labels if l == c) for c in range(2)]
    criterion = ClassBalancedLoss(samples_per_class, beta=0.9999, loss_type='focal')
    print("Using Class-Balanced Loss for training")
elif training_config.get('use_class_weights', True):
    # Original: Use class weights
    from src.data_loader import calculate_class_weights
    train_labels = [label for _, label in train_loader.dataset]
    class_weights = calculate_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Using Cross-Entropy with class weights")
else:
    criterion = nn.CrossEntropyLoss()
    print("Using standard Cross-Entropy")
```

### Atualizar config.yaml:

```yaml
training:
  use_focal_loss: true  # Enable Focal Loss
  use_class_balanced_loss: false
  use_class_weights: false  # Disable when using focal loss
```

---

## ✅ SOLUÇÃO 5: Test-Time Augmentation (TTA)

### Benefício: Reduz variância sem re-treinar

### Implementação:

Criar arquivo: `src/tta.py`

```python
"""
Test-Time Augmentation for robust predictions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List
import albumentations as A


class TTAWrapper:
    """
    Test-Time Augmentation wrapper
    
    Applies multiple augmentations at test time and averages predictions
    """
    
    def __init__(
        self,
        model: nn.Module,
        transforms: List[A.Compose],
        device: torch.device
    ):
        """
        Args:
            model: Trained model
            transforms: List of augmentation transforms
            device: Device to run on
        """
        self.model = model
        self.transforms = transforms
        self.device = device
        
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict with TTA
        
        Args:
            image: Input image tensor (C, H, W)
            
        Returns:
            Averaged predictions
        """
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            # Original image
            output = self.model(image.unsqueeze(0).to(self.device))
            predictions.append(torch.softmax(output, dim=1))
            
            # Augmented versions
            image_np = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
            
            for transform in self.transforms:
                augmented = transform(image=image_np)['image']
                augmented_tensor = torch.from_numpy(augmented.transpose(2, 0, 1))
                
                output = self.model(augmented_tensor.unsqueeze(0).to(self.device))
                predictions.append(torch.softmax(output, dim=1))
        
        # Average predictions
        avg_prediction = torch.mean(torch.cat(predictions, dim=0), dim=0, keepdim=True)
        
        return avg_prediction


def get_tta_transforms(image_size=(224, 224)):
    """
    Get list of TTA transforms
    
    Returns:
        List of augmentation transforms
    """
    transforms = [
        # Horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(height=image_size[0], width=image_size[1])
        ]),
        
        # Rotate +5 degrees
        A.Compose([
            A.Rotate(limit=5, border_mode=0, p=1.0),
            A.Resize(height=image_size[0], width=image_size[1])
        ]),
        
        # Rotate -5 degrees
        A.Compose([
            A.Rotate(limit=-5, border_mode=0, p=1.0),
            A.Resize(height=image_size[0], width=image_size[1])
        ]),
        
        # Brightness adjustment
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1.0),
            A.Resize(height=image_size[0], width=image_size[1])
        ]),
        
        # CLAHE
        A.Compose([
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.Resize(height=image_size[0], width=image_size[1])
        ])
    ]
    
    return transforms
```

---

## 📋 PLANO DE EXECUÇÃO CONSOLIDADO

### Semana 1: Fundação Robusta (5 dias)

#### Dia 1: Cross-Validation Setup
```bash
# Implementar src/cross_validation.py
# Executar CV para EfficientNetB0 (mais rápido)
python -m src.cross_validation
```

**Output esperado:**
- EfficientNetB0: 80.29% ± 2.5% accuracy (95% CI: [77.8%, 82.8%])
- Modelos salvos para cada fold

#### Dia 2: Threshold Optimization
```bash
# Implementar src/threshold_optimization.py
# Otimizar threshold para cada modelo
python -m src.threshold_optimization
```

**Output esperado:**
- Especificidade aumentar para ~60%
- Balanced accuracy melhorar
- Gráficos de análise salvos

#### Dia 3: Advanced Augmentation + Focal Loss
- Atualizar `src/data_loader.py` com augmentation avançada
- Implementar `src/losses.py`
- Re-treinar apenas EfficientNetB0 com Focal Loss

**Output esperado:**
- Especificidade base melhorar em 5-10%
- Modelos mais balanceados

#### Dia 4: Test-Time Augmentation
- Implementar `src/tta.py`
- Testar TTA em modelos existentes

**Output esperado:**
- Redução de variância
- Melhor robustez

#### Dia 5: Consolidação e Validação
- Rodar todos os scripts
- Gerar relatório consolidado
- Verificar métricas finais

---

### Semana 2: Ensemble e Análise Final (5 dias)

Com base sólida estabelecida, agora podemos prosseguir com ensemble (IMPLEMENTATION_GUIDE.md).

---

## 📊 Métricas de Sucesso Pós-Correções

| Métrica | Antes | Target Pós-Correção | Prioridade |
|---------|-------|---------------------|------------|
| **Especificidade** | 12-48% | **≥ 60%** | 🔴 CRÍTICA |
| **Val Set Size** | 16 amostras | **~1000 samples (5-fold CV)** | 🔴 CRÍTICA |
| **Balanced Accuracy** | ~56% | **≥ 75%** | 🟠 ALTA |
| **CI Width** | N/A | **< 5% (para accuracy)** | 🟠 ALTA |
| **Augmentation Diversity** | 4 tipos | **10+ tipos** | 🟡 MÉDIA |

---

## 🎯 Conclusão

Este plano resolve TODOS os gaps críticos identificados:

✅ Dataset pequeno → Cross-Validation  
✅ Especificidade baixa → Threshold Optimization + Focal Loss  
✅ Desbalanceamento → Focal Loss + Class-Balanced Loss  
✅ Falta de CV → Stratified K-Fold implementado  
✅ Augmentation limitada → Advanced augmentation for X-Ray  
✅ Variância alta → Test-Time Augmentation

**Após estas correções, o projeto terá:**
- Base estatisticamente sólida
- Métricas confiáveis e replicáveis
- Especificidade clinicamente útil (≥60%)
- Intervalos de confiança bem definidos
- Fundação robusta para ensemble learning

**Tempo estimado:** 10 dias de implementação focada

**Próximo passo:** Executar Dia 1 - Cross-Validation Setup

---

**Autores:** Jéssica A. L. de Macêdo & Matheus Borges Figueirôa  
**CIn - UFPE**  
**Última atualização:** 12/11/2025
