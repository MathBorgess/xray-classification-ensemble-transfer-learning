# --- validate_tta.py ---

import os
import torch
import matplotlib
import numpy as np
import argparse 

# --- CORREÇÃO DE ERROS AMBIENTAIS ---
# 1. Corrige o erro OMP (libiomp5md.dll)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 2. Corrige o erro de Plotting (Matplotlib backend)
matplotlib.use('Agg')
# --- FIM CORREÇÃO DE ERROS AMBIENTAIS ---

#imports do projeto
from src.tta import compare_with_without_tta
from src.models import create_model
from src.data_loader import load_data_from_directory, ChestXRayDataset, get_transforms
from src.utils import load_config, get_device
from torch.utils.data import DataLoader


def run_tta_validation(config: dict):
    device = get_device(config)
    print(f"Usando device: {device}")

    # carrega testset
    data_dir = config.get('data', {}).get('data_dir', 'data/raw/chest_xray')
    test_dir = os.path.join(data_dir, 'test')
    
    print("Loading test data for TTA validation...")
    test_paths, test_labels = load_data_from_directory(test_dir, config)
    test_transform = get_transforms(config, train=False)
    
    test_dataset = ChestXRayDataset(test_paths, test_labels, transform=test_transform, augmentation=None)
    
    batch_size = config.get('data', {}).get('batch_size', 32)
    num_workers = config.get('data', {}).get('num_workers', 4)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    print(f"Total de amostras de teste carregadas: {len(test_dataset)}")

    # carrega modelo treinado (Ex: EfficientNetB0, Fold 1)
    model_name = 'efficientnet_b0'
    model_path = os.path.join('models', 'cv_models', f'{model_name}_fold1.pth')

    if not os.path.exists(model_path):
        print(f"\nErro: Arquivo do modelo não encontrado em {model_path}. Por favor, verifique.")
        return

    print(f"\nCarregando modelo {model_name} para validação TTA...")
    model = create_model(model_name, config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # executa comparação TTA
    print("\nIniciando Comparação TTA vs. Padrão...")
    results = compare_with_without_tta(
        model, test_loader, config, device, n_augmentations=5
    )

    print("\n--- Validação TTA Concluída ---")
    print(f"Métricas (sem TTA): AUC={results['without_tta']['auc']:.4f}, Spec={results['without_tta']['specificity']:.4f}")
    print(f"Métricas (com TTA): AUC={results['with_tta']['auc']:.4f}, Spec={results['with_tta']['specificity']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TTA Validation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    run_tta_validation(config)