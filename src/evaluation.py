"""
Evaluation metrics and robustness testing
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# imports adicionais
import argparse
import os
import json
from pathlib import Path
from src.utils import load_config, get_device
from src.models import create_model
from src.data_loader import load_data_from_directory, ChestXRayDataset, get_transforms
from torch.utils.data import DataLoader
from src.trainer import evaluate # importa a função evaluate do trainer
# Nota: O TTAWrapper será importado dentro da função se use_tta for True.

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> np.ndarray:
    if class_names is None:
        class_names = ['Normal', 'Pneumonia']

    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None
):
    """
    Plot confusion matrix
    """
    if class_names is None:
        class_names = ['Normal', 'Pneumonia']

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str = None
):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str = None
):
    """
    Plot Precision-Recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric: str = 'auc',
    save_path: str = None
):
    """
    Compare multiple models on a metric
    """
    models = list(results.keys())
    scores = [results[model][metric] for model in models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color='steelblue', alpha=0.7)

    # add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.xlabel('Model')
    plt.ylabel(metric.upper())
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Perform paired t-test
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    is_significant = p_value < alpha

    return t_stat, p_value, is_significant


def apply_gaussian_noise(
    images: torch.Tensor,
    sigma: float = 10.0
) -> torch.Tensor:
    """
    Apply Gaussian noise to images
    """
    noise = torch.randn_like(images) * (sigma / 255.0)
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)


def apply_contrast_reduction(
    images: torch.Tensor,
    level: float = 0.5
) -> torch.Tensor:
    """
    Reduce contrast of images
    """
    mean = images.mean(dim=(2, 3), keepdim=True)
    contrast_reduced = mean + level * (images - mean)
    return torch.clamp(contrast_reduced, 0.0, 1.0)


def apply_rotation(
    images: torch.Tensor,
    angle: float
) -> torch.Tensor:
    """
    Apply rotation to images
    """
    from torchvision.transforms.functional import rotate
    return rotate(images, angle)


def test_robustness(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Test model robustness under perturbations
    """
    from src.trainer import evaluate

    eval_config = config.get('evaluation', {})
    perturbations_config = eval_config.get('perturbations', {})

    results = {}
    criterion = torch.nn.CrossEntropyLoss()

    # baseline (no perturbation)
    print("\nTesting baseline (no perturbation)...")
    baseline_metrics = evaluate(model, test_loader, criterion, device)
    results['baseline'] = baseline_metrics

    # gaussian noise
    if 'gaussian_noise' in perturbations_config:
        sigmas = perturbations_config['gaussian_noise'].get('sigma', [10, 20])
        for sigma in sigmas:
            print(f"\nTesting Gaussian noise (sigma={sigma})...")
            # create perturbed dataset
            perturbed_metrics = test_with_perturbation(
                model, test_loader, device, criterion,
                perturbation_fn=lambda x: apply_gaussian_noise(x, sigma)
            )
            results[f'gaussian_noise_sigma_{sigma}'] = perturbed_metrics

    # contrast reduction
    if 'contrast_reduction' in perturbations_config:
        levels = perturbations_config['contrast_reduction'].get('levels', [
                                                             0.5, 0.7])
        for level in levels:
            print(f"\nTesting contrast reduction (level={level})...")
            perturbed_metrics = test_with_perturbation(
                model, test_loader, device, criterion,
                perturbation_fn=lambda x: apply_contrast_reduction(x, level)
            )
            results[f'contrast_reduction_{int(level*100)}'] = perturbed_metrics

    # rotation
    if 'rotation' in perturbations_config:
        angles = perturbations_config['rotation'].get('angles', [5, 10])
        for angle in angles:
            print(f"\nTesting rotation (angle=±{angle}°)...")
            perturbed_metrics = test_with_perturbation(
                model, test_loader, device, criterion,
                perturbation_fn=lambda x: apply_rotation(x, angle)
            )
            results[f'rotation_{angle}deg'] = perturbed_metrics

    return results


def test_with_perturbation(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    perturbation_fn
) -> Dict[str, float]:
    """
    Test model with specific perturbation
    """
    from src.trainer import evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # apply perturbation
            inputs = perturbation_fn(inputs)

            # forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_score = 0.0
    f1 = f1_score(all_labels, all_preds, average='binary')

    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'f1_score': f1
    }


# main code
def run_final_evaluation(config: Dict, use_tta: bool = False):
    """
    carrega modelos e executa a avaliação final (Ensemble, TTA, Robustez).
    """
    device = get_device(config)
    save_dir = Path(config.get('paths', {}).get('results', 'results/'))
    
    # carregar o test set
    data_dir = config.get('data', {}).get('data_dir', 'data/raw/chest_xray')
    test_dir = os.path.join(data_dir, 'test')

    print("\nLoading test data for final evaluation...")
    test_paths, test_labels = load_data_from_directory(test_dir, config)
    test_transform = get_transforms(config, train=False)
    
    test_dataset = ChestXRayDataset(test_paths, test_labels, transform=test_transform, augmentation=None)
    
    batch_size = config.get('data', {}).get('batch_size', 32)
    
    # IMPORTANTE: CORREÇÃO DO MULTIPROCESSING NO WINDOWS
    num_workers = config.get('data', {}).get('num_workers', 4)
    if os.name == 'nt' and num_workers > 0: 
        print("AVISO: Multiprocessing desativado (num_workers=0) para evitar travamento no Windows.")
        num_workers = 0 

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    print(f"Total de amostras de teste carregadas: {len(test_dataset)}")

    # carregar thresholds otimos (necessário para o ensemble)
    thresholds_path = save_dir / 'optimal_thresholds.json'
    if not thresholds_path.exists():
        print(f"❌ ERRO: Arquivo de thresholds não encontrado em {thresholds_path}. Execute o threshold_optimization.py primeiro.")
        return

    with open(thresholds_path, 'r') as f:
        optimized_thresholds = json.load(f)
        
    final_results = {}
    models_to_evaluate = ['efficientnet_b0', 'resnet50', 'densenet121']

    # avaliação de desempenho (individual + robustez)
    for model_name in models_to_evaluate:
        for fold_num in range(config.get('evaluation', {}).get('n_splits', 5)):
            fold_index = fold_num + 1
            print(f"\n--- Avaliando {model_name} Fold {fold_index} ---")
            
            checkpoint_path = Path('models/cv_models') / f'{model_name}_fold{fold_index}.pth'
            if not checkpoint_path.exists():
                continue

            model = create_model(model_name, config)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model = model.to(device)

            # aval. base e robustez
            robustness_metrics = test_robustness(
                model, test_loader, config, device
            )
            
            # avaliação com TTA (se solicitado)
            tta_metrics = None
            if use_tta:
                from src.tta import TTAWrapper # Importa localmente para evitar problemas de dependência
                tta_wrapper = TTAWrapper(model, config, n_augmentations=5, device=device)
                tta_metrics = tta_wrapper.evaluate_with_tta(test_loader)
            
            final_results[f'{model_name}_fold{fold_index}'] = {
                'base_metrics': robustness_metrics.get('baseline', {}),
                'robustness': robustness_metrics,
                'tta_metrics': tta_metrics
            }


    # salvar
    final_output_path = save_dir / 'final_evaluation_summary.json'
    
    def convert_numpy(obj):
        if isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
            return None
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(final_output_path, 'w') as f:
        json.dump(final_results, f, indent=4, default=convert_numpy)
    
    print(f"\nAvaliação Final Concluída e salva em: {final_output_path}")

    # TODO: FASE FINAL: Cálculo do Ensemble (voto ponderado)

    return final_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Final Model Evaluation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--use-tta', action='store_true',
                        help='Use Test-Time Augmentation for final prediction')
    
    args = parser.parse_args()

    # define variável OMP para ignorar o erro de MKL/openmp
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    config = load_config(args.config)

    # IMPORTANTE: Desativar temporariamente o multiprocessing no Windows
    # para evitar RuntimeErrors no DataLoader
    if os.name == 'nt' and config.get('data', {}).get('num_workers', 4) > 0:
        config['data']['num_workers'] = 0 
    
    try:
        run_final_evaluation(config, use_tta=args.use_tta)
    except Exception as e:
        print(f"\n❌ ERRO FATAL DURANTE A AVALIAÇÃO FINAL: {e}")
        # print(traceback.format_exc()) # Descomente para ver o erro completo