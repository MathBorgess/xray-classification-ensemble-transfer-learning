# 🚀 Guia de Execução - Correções e Re-treinamento

## ⚡ Quick Start

### Passo 1: Verificar Sistema

```bash
python3 scripts/quickstart_fixes.py
```

### Passo 2: Executar Re-treinamento Completo

```bash
# Modo completo (todos os 3 modelos)
python3 retrain_with_improvements.py

# Modo rápido (apenas EfficientNetB0)
python3 retrain_with_improvements.py --quick
```

Este comando executa:
- ✅ Cross-Validation (K=5)
- ✅ Threshold Optimization
- ✅ Test-Time Augmentation

**Tempo estimado:**
- Modo rápido: ~3-4 horas
- Modo completo: ~8-10 horas

---

## 📋 Execução Detalhada (Por Etapa)

### Etapa 1: Cross-Validation

```bash
python3 -m src.cross_validation --config configs/config.yaml
```

Ou apenas EfficientNetB0:
```bash
python3 -m src.cross_validation --quick
```

**Resultado esperado:**
- 5 modelos por arquitetura (folds 1-5)
- Métricas com intervalos de confiança (95% CI)
- Especificidade esperada: 40-60% (melhoria de ~30%)

**Saída:**
```
models/cv_models/
├── efficientnet_b0_fold1.pth
├── efficientnet_b0_fold2.pth
├── efficientnet_b0_fold3.pth
├── efficientnet_b0_fold4.pth
├── efficientnet_b0_fold5.pth
├── efficientnet_b0_cv_summary.json
└── ... (outros modelos)

results/
└── cross_validation_results.json
```

---

### Etapa 2: Threshold Optimization

Requer modelos treinados (Etapa 1).

```python
from src.threshold_optimization import optimize_threshold_for_model
from src.models import create_model
from src.data_loader import get_dataloaders
from src.utils import load_config, get_device
import torch

# Load config and data
config = load_config('configs/config.yaml')
device = get_device(config)
_, val_loader, _ = get_dataloaders(config, use_existing_splits=True)

# Load trained model
model = create_model('efficientnet_b0', config)
model.load_state_dict(torch.load('models/cv_models/efficientnet_b0_fold1.pth'))
model = model.to(device)

# Optimize thresholds
results = optimize_threshold_for_model(
    model, val_loader, device,
    methods=['youden', 'f1', 'balanced', 'target_specificity'],
    save_dir='results/threshold_efficientnet_b0'
)

# Check target_specificity result
target_result = results['target_specificity']
print(f"Optimal threshold: {target_result['threshold']:.4f}")
print(f"Sensitivity: {target_result['metrics']['sensitivity']:.4f}")
print(f"Specificity: {target_result['metrics']['specificity']:.4f}")
```

**Resultado esperado:**
- Threshold otimizado para Especificidade ≥ 60%
- Gráficos ROC com pontos ótimos
- Comparação de métodos

---

### Etapa 3: Test-Time Augmentation

```python
from src.tta import compare_with_without_tta
from src.models import create_model
from src.data_loader import get_dataloaders
from src.utils import load_config, get_device
import torch

# Load config and data
config = load_config('configs/config.yaml')
device = get_device(config)
_, _, test_loader = get_dataloaders(config, use_existing_splits=True)

# Load model
model = create_model('efficientnet_b0', config)
model.load_state_dict(torch.load('models/cv_models/efficientnet_b0_fold1.pth'))
model = model.to(device)

# Compare with/without TTA
results = compare_with_without_tta(
    model, test_loader, config, device, n_augmentations=5
)

print("\nImprovement with TTA:")
for key in ['accuracy', 'auc', 'specificity']:
    std_val = results['without_tta'][key]
    tta_val = results['with_tta'][key]
    improvement = tta_val - std_val
    print(f"{key}: {std_val:.4f} → {tta_val:.4f} (+{improvement:.4f})")
```

**Resultado esperado:**
- Melhoria de 1-3% em todas as métricas
- Redução de variância nas predições

---

## 🔧 Opções Avançadas

### Executar apenas fases específicas

```bash
# Apenas Cross-Validation
python3 retrain_with_improvements.py --skip-threshold --skip-tta

# Apenas Threshold Optimization
python3 retrain_with_improvements.py --skip-cv --skip-tta

# Apenas TTA
python3 retrain_with_improvements.py --skip-cv --skip-threshold
```

### Especificar modelos

```bash
# Apenas EfficientNetB0 e ResNet50
python3 retrain_with_improvements.py --models efficientnet_b0 resnet50
```

---

## 📊 Validação dos Resultados

### Checklist de Validação

Após executar todas as etapas, verifique:

- [ ] **Cross-Validation**
  - [ ] 5 modelos por arquitetura gerados
  - [ ] Intervalos de confiança (CI) calculados
  - [ ] CI width < 5% para métricas principais
  - [ ] Especificidade média ≥ 40%

- [ ] **Threshold Optimization**
  - [ ] Threshold otimizado salvo para cada modelo
  - [ ] Especificidade ≥ 60% alcançada
  - [ ] Sensibilidade mantida ≥ 90%
  - [ ] Gráficos gerados

- [ ] **Test-Time Augmentation**
  - [ ] Melhoria observada em pelo menos 2 métricas
  - [ ] Especificidade aumentada
  - [ ] AUC aumentada

### Script de Validação

```python
import json
from pathlib import Path

# Check CV results
cv_file = Path('results/cross_validation_results.json')
if cv_file.exists():
    with open(cv_file) as f:
        cv_results = json.load(f)
    
    print("✅ Cross-Validation Results:")
    for model, metrics in cv_results.items():
        spec = metrics['mean_metrics']['specificity']
        spec_ci = metrics['ci_95_metrics']['specificity']
        ci_width = spec_ci[1] - spec_ci[0]
        
        print(f"  {model}:")
        print(f"    Specificity: {spec:.4f} ± {ci_width/2:.4f}")
        print(f"    CI Width: {ci_width:.4f}")
        
        # Validation
        if spec >= 0.40:
            print("    ✅ Specificity target met (≥40%)")
        else:
            print("    ⚠️  Specificity below target")
        
        if ci_width < 0.05:
            print("    ✅ CI width acceptable (<5%)")
        else:
            print("    ⚠️  CI width too large")

# Check threshold optimization
thresh_files = list(Path('results/improved_training').glob('*_optimal_thresholds.json'))
print(f"\n✅ Threshold Optimization: {len(thresh_files)} models")

for thresh_file in thresh_files:
    with open(thresh_file) as f:
        thresh_results = json.load(f)
    
    if 'target_specificity' in thresh_results:
        result = thresh_results['target_specificity']
        spec = result['metrics']['specificity']
        sens = result['metrics']['sensitivity']
        
        print(f"  {thresh_file.stem}:")
        print(f"    Specificity: {spec:.4f}")
        print(f"    Sensitivity: {sens:.4f}")
        
        if spec >= 0.60:
            print("    ✅ Target specificity met (≥60%)")
        else:
            print("    ⚠️  Target not met")
```

---

## 🎯 Metas de Sucesso

### Antes das Correções
```
Dataset Validação:    16 amostras
Especificidade:       12-48%
Cross-Validation:     ❌ Ausente
Intervalo Confiança:  ❌ Ausente
```

### Após as Correções (Esperado)
```
Dataset Validação:    ~1000 samples (5-fold CV)
Especificidade:       ≥ 60%
Cross-Validation:     ✅ 5-fold
Intervalo Confiança:  ✅ 95% CI
```

### Critérios de Aceitação

| Métrica | Meta | Status |
|---------|------|--------|
| Especificidade | ≥ 60% | 🎯 |
| Sensibilidade | ≥ 90% | 🎯 |
| Balanced Accuracy | ≥ 75% | 🎯 |
| AUC | ≥ 0.85 | 🎯 |
| CI Width | < 5% | 🎯 |

---

## ⏱️ Cronograma de Execução

| Etapa | Duração | Descrição |
|-------|---------|-----------|
| Cross-Validation | 6-8h | 3 modelos × 5 folds × ~30min/fold |
| Threshold Optimization | 30min | Análise de curvas ROC |
| TTA Evaluation | 1-2h | 5 augmentations por imagem |
| **Total** | **8-11h** | Pode rodar overnight |

---

## 🐛 Troubleshooting

### Erro: CUDA Out of Memory

```bash
# Reduzir batch size
# Edit configs/config.yaml:
data:
  batch_size: 16  # Era 32
```

### Erro: Model file not found

```bash
# Certifique-se de executar CV primeiro
python3 -m src.cross_validation --quick
```

### Validação demora muito

```bash
# Use modo quick para testar
python3 retrain_with_improvements.py --quick
```

---

## 📁 Estrutura de Saída

Após execução completa:

```
models/
└── cv_models/
    ├── efficientnet_b0_fold1.pth
    ├── efficientnet_b0_fold2.pth
    ├── ...
    ├── efficientnet_b0_cv_summary.json
    └── ... (outros modelos)

results/
├── cross_validation_results.json
└── improved_training/
    ├── training_summary.json
    ├── efficientnet_b0_optimal_thresholds.json
    ├── resnet50_optimal_thresholds.json
    ├── densenet121_optimal_thresholds.json
    ├── threshold_efficientnet_b0/
    │   └── threshold_optimization.png
    └── ... (outros modelos)
```

---

## ✅ Próximos Passos

Após completar todas as etapas com sucesso:

1. **Validar resultados** usando o script de validação acima
2. **Revisar métricas** em `results/improved_training/training_summary.json`
3. **Documentar melhorias** para o artigo
4. **Prosseguir para Ensemble** usando `IMPLEMENTATION_GUIDE.md`

---

**Autores:** Jéssica A. L. de Macêdo & Matheus Borges Figueirôa  
**Data:** 12 de Novembro de 2025  
**Status:** 🔴 PRONTO PARA EXECUÇÃO
