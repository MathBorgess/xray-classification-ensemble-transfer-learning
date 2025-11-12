# 🛠️ Guia de Implementação Técnica - Próximos Passos

**Documento Complementar ao Progress Report**  
**Autores:** Jéssica A. L. de Macêdo & Matheus Borges Figueirôa  
**Data:** Novembro 2025

---

## 🎯 Visão Geral: 5 Tarefas Críticas

Este documento fornece **scripts executáveis** e **código pronto** para as 5 tarefas mais urgentes do projeto.

### Checklist de Implementação

- [ ] **Tarefa 1:** Executar Ensemble (30 min)
- [ ] **Tarefa 2:** Análise de Robustez (1h)
- [ ] **Tarefa 3:** Visualizações Grad-CAM (45 min)
- [ ] **Tarefa 4:** Validação Estatística (1h)
- [ ] **Tarefa 5:** Threshold Tuning (30 min)

**Tempo Total Estimado:** ~4 horas de execução

---

## 📦 Tarefa 1: Executar Ensemble (PRIORIDADE MÁXIMA)

### Comandos Rápidos:

```bash
# 1. Verificar modelos treinados
ls -lh models/*.pth

# Você deve ter:
# - efficientnet_b0_final.pth
# - resnet50_final.pth
# - densenet121_final.pth

# 2. Executar ensemble
python ensemble.py --model_dir models --output_dir results

# 3. Ver resultados
cat results/ensemble_comparison.txt

# 4. Visualizar gráficos
open results/figures/comparison_accuracy.png
open results/figures/comparison_auc.png
open results/figures/comparison_f1_score.png
```

### O que Esperar:

**Arquivo gerado:** `results/ensemble_comparison.txt`

```
Ensemble Model Comparison
================================================================================

Model                          Accuracy     AUC          F1-Score     Sensitivity  Specificity
--------------------------------------------------------------------------------
efficientnet_b0                0.8029       0.9761       0.8635       0.9974       0.4786
resnet50                       0.6715       0.9230       0.7915       0.9974       0.1282
densenet121                    0.6891       0.9505       0.8008       1.0000       0.1709
simple_voting                  0.XXXX       0.XXXX       0.XXXX       0.XXXX       0.XXXX
weighted_voting                0.XXXX       0.XXXX       0.XXXX       0.XXXX       0.XXXX
```

### Se Houver Erro:

```bash
# Erro comum: modelo não encontrado
# Solução: Treinar modelos faltantes

python train.py --model efficientnet_b0
python train.py --model resnet50
python train.py --model densenet121
```

---

## 🧪 Tarefa 2: Análise de Robustez

### Script Completo:

```bash
# Testar cada modelo sob perturbações
for model in efficientnet_b0 resnet50 densenet121; do
    echo "Testing robustness of $model..."
    python test_robustness.py \
        --model $model \
        --model_path models/${model}_final.pth \
        --output_dir results
done

# Ver resultados
cat results/efficientnet_b0_robustness.txt
cat results/resnet50_robustness.txt
cat results/densenet121_robustness.txt
```

### Análise dos Resultados:

Após executar, criar tabela comparativa:

```python
# Script para consolidar resultados: consolidate_robustness.py

import pandas as pd
import glob

results = []
for file in glob.glob('results/*_robustness.txt'):
    model_name = file.split('/')[-1].replace('_robustness.txt', '')
    # Parse file and extract metrics
    # Add to results list

df = pd.DataFrame(results)
df.to_csv('results/robustness_comparison.csv', index=False)
print(df.to_markdown())
```

### Métricas a Analisar:

1. **Degradação Absoluta:** `baseline_acc - perturbed_acc`
2. **Degradação Relativa:** `(baseline_acc - perturbed_acc) / baseline_acc * 100%`
3. **Ranking de Robustez:** Qual modelo é mais estável?

---

## 👁️ Tarefa 3: Visualizações Grad-CAM

### Script para Gerar Grad-CAM:

Criar arquivo: `scripts/generate_gradcam.py`

```python
"""
Script to generate Grad-CAM visualizations
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from pathlib import Path
from src.utils import load_config, get_device
from src.data_loader import get_dataloaders
from src.models import create_model
from src.interpretability import visualize_multiple_samples, get_target_layer

def main():
    # Load config
    config = load_config('configs/config.yaml')
    device = get_device(config)

    # Load data
    _, _, test_loader = get_dataloaders(config)

    # Get sample batch
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Models to visualize
    models = ['efficientnet_b0', 'resnet50', 'densenet121']

    for model_name in models:
        print(f"\nGenerating Grad-CAM for {model_name}...")

        # Load model
        model = create_model(model_name, config, pretrained=False)
        model.load_state_dict(torch.load(f'models/{model_name}_final.pth', map_location=device))
        model = model.to(device)
        model.eval()

        # Get predictions
        with torch.no_grad():
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

        # Generate Grad-CAM
        save_path = f'results/figures/gradcam_{model_name}.png'
        visualize_multiple_samples(
            model, images, labels, predictions,
            model_name, num_samples=8, save_path=save_path
        )
        print(f"Saved to {save_path}")

if __name__ == '__main__':
    main()
```

### Executar:

```bash
python scripts/generate_gradcam.py

# Ver visualizações
open results/figures/gradcam_efficientnet_b0.png
open results/figures/gradcam_resnet50.png
open results/figures/gradcam_densenet121.png
```

### Análise Qualitativa:

Para cada visualização, responder:

1. **O modelo está olhando para as regiões corretas?**

   - Campos pulmonares (onde ocorre pneumonia)
   - Não para bordas, artefatos ou textos

2. **Há consistência entre modelos?**

   - Modelos concordam nas regiões importantes?
   - Diferenças podem explicar performance?

3. **Casos de erro:**
   - Em falsos positivos, o que o modelo vê?
   - Em falsos negativos, o modelo perdeu algo óbvio?

---

## 📊 Tarefa 4: Validação Estatística

### Script de Análise Estatística:

Criar arquivo: `scripts/statistical_analysis.py`

```python
"""
Statistical validation of results
"""

import numpy as np
from scipy import stats
from sklearn.utils import resample
import pandas as pd

def paired_t_test(scores_a, scores_b, labels=("Model A", "Model B")):
    """Perform paired t-test"""
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

    print(f"\nPaired t-test: {labels[0]} vs {labels[1]}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"  ✓ Significant difference (p < 0.05)")
        if t_stat > 0:
            print(f"  → {labels[0]} is significantly better")
        else:
            print(f"  → {labels[1]} is significantly better")
    else:
        print(f"  ✗ No significant difference (p ≥ 0.05)")

    return t_stat, p_value

def bootstrap_ci(y_true, y_pred, metric_fn, n_iterations=1000, confidence=0.95):
    """Calculate bootstrap confidence intervals"""
    scores = []
    n = len(y_true)

    for _ in range(n_iterations):
        indices = resample(range(n), n_samples=n)
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]
        scores.append(metric_fn(y_true_boot, y_pred_boot))

    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    mean = np.mean(scores)

    return mean, lower, upper

def mcnemar_test(y_true, pred_a, pred_b, labels=("Model A", "Model B")):
    """Perform McNemar's test"""
    from statsmodels.stats.contingency_tables import mcnemar

    # Create contingency table
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    both_correct = np.sum(correct_a & correct_b)
    a_correct_b_wrong = np.sum(correct_a & ~correct_b)
    a_wrong_b_correct = np.sum(~correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)

    table = [[both_correct, a_correct_b_wrong],
             [a_wrong_b_correct, both_wrong]]

    result = mcnemar(table, exact=True)

    print(f"\nMcNemar's test: {labels[0]} vs {labels[1]}")
    print(f"  Contingency table:")
    print(f"    Both correct: {both_correct}")
    print(f"    Only {labels[0]} correct: {a_correct_b_wrong}")
    print(f"    Only {labels[1]} correct: {a_wrong_b_correct}")
    print(f"    Both wrong: {both_wrong}")
    print(f"  p-value: {result.pvalue:.4f}")

    if result.pvalue < 0.05:
        print(f"  ✓ Significant difference (p < 0.05)")
    else:
        print(f"  ✗ No significant difference (p ≥ 0.05)")

    return result

# Example usage:
if __name__ == '__main__':
    # Load predictions from each model
    # TODO: Implement loading logic

    # Example:
    # y_true = load_true_labels()
    # pred_efficientnet = load_predictions('efficientnet_b0')
    # pred_ensemble = load_predictions('ensemble')

    # Perform tests
    # paired_t_test(scores_efficientnet, scores_ensemble,
    #               labels=("EfficientNet", "Ensemble"))

    # Bootstrap CI for accuracy
    # from sklearn.metrics import accuracy_score
    # mean, lower, upper = bootstrap_ci(y_true, pred_ensemble, accuracy_score)
    # print(f"\nEnsemble Accuracy: {mean:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")

    pass
```

### Executar Análise:

```bash
# Após coletar predições de todos os modelos
python scripts/statistical_analysis.py
```

### Reportar Resultados:

**Formato para o artigo:**

> A votação ponderada alcançou accuracy de 0.8234 (95% CI: [0.8012, 0.8456]),
> superando significativamente o EfficientNetB0 individual (0.8029, p = 0.032,
> teste t-pareado). O teste de McNemar confirmou diferença significativa
> (p = 0.041), com o ensemble corrigindo 23 erros do modelo individual.

---

## 🎚️ Tarefa 5: Threshold Tuning (Otimização de Especificidade)

### Script de Otimização:

Criar arquivo: `scripts/optimize_threshold.py`

```python
"""
Optimize decision threshold to improve specificity
"""

import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def find_optimal_threshold(y_true, y_scores, method='youden'):
    """
    Find optimal decision threshold

    Methods:
    - 'youden': Maximize Youden's J = Sensitivity + Specificity - 1
    - 'f1': Maximize F1-score
    - 'balanced': Balance sensitivity and specificity (closest to perfect classifier)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    if method == 'youden':
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
    elif method == 'f1':
        # Approximate F1 (need to calculate for each threshold)
        f1_scores = []
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        optimal_idx = np.argmax(f1_scores)
    elif method == 'balanced':
        # Minimize distance to perfect classifier (0, 1)
        distances = np.sqrt((1 - tpr)**2 + fpr**2)
        optimal_idx = np.argmin(distances)

    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]

    return optimal_threshold, optimal_tpr, optimal_fpr

def evaluate_with_threshold(y_true, y_scores, threshold):
    """Evaluate model with custom threshold"""
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'threshold': threshold
    }

# Example usage:
if __name__ == '__main__':
    # Load predictions
    # y_true = ...
    # y_scores = ...  # probability scores for positive class

    # Find optimal thresholds
    print("Threshold Optimization Results")
    print("=" * 60)

    for method in ['youden', 'f1', 'balanced']:
        threshold, tpr, fpr = find_optimal_threshold(y_true, y_scores, method=method)
        metrics = evaluate_with_threshold(y_true, y_scores, threshold)

        print(f"\n{method.upper()} Method:")
        print(f"  Optimal threshold: {threshold:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")

    # Compare with default threshold (0.5)
    print("\n" + "=" * 60)
    print("DEFAULT Threshold (0.5):")
    metrics_default = evaluate_with_threshold(y_true, y_scores, 0.5)
    print(f"  Accuracy: {metrics_default['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics_default['sensitivity']:.4f}")
    print(f"  Specificity: {metrics_default['specificity']:.4f}")
    print(f"  F1-score: {metrics_default['f1']:.4f}")
```

### Executar:

```bash
python scripts/optimize_threshold.py
```

### Impacto Esperado:

| Métrica     | Default (0.5) | Youden     | F1         | Balanced |
| ----------- | ------------- | ---------- | ---------- | -------- |
| Threshold   | 0.5000        | ~0.35      | ~0.45      | ~0.40    |
| Accuracy    | 80.29%        | **82-84%** | 81-83%     | 81-83%   |
| Sensitivity | 99.74%        | 95-98%     | 96-99%     | 96-98%   |
| Specificity | 47.86%        | **55-65%** | 50-60%     | 52-62%   |
| F1          | 86.35%        | 85-87%     | **86-88%** | 85-87%   |

**Insight:** Threshold tuning pode melhorar especificidade em ~10-15% com pequena perda de sensibilidade.

---

## 📈 Tarefa Bônus: Curvas de Aprendizado

### Script para Análise de Overfitting:

Criar arquivo: `scripts/plot_learning_curves.py`

```python
"""
Plot learning curves to analyze overfitting
"""

import matplotlib.pyplot as plt
import json

def plot_learning_curves(history, model_name, save_path):
    """
    Plot training and validation curves

    history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_auc'
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    # AUC
    axes[2].plot(epochs, history['val_auc'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('AUC', fontsize=12)
    axes[2].set_title(f'{model_name} - Validation AUC', fontsize=14, fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curves to {save_path}")

    # Análise de overfitting
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    gap = final_val_loss - final_train_loss

    print(f"\nOverfitting Analysis for {model_name}:")
    print(f"  Final train loss: {final_train_loss:.4f}")
    print(f"  Final val loss: {final_val_loss:.4f}")
    print(f"  Gap: {gap:.4f}")

    if gap > 0.2:
        print(f"  ⚠️ Significant overfitting detected")
    elif gap > 0.1:
        print(f"  ⚡ Moderate overfitting")
    else:
        print(f"  ✓ Good generalization")

# Load and plot for each model
if __name__ == '__main__':
    models = ['efficientnet_b0', 'resnet50', 'densenet121']

    for model_name in models:
        # TODO: Load history from training
        # history = load_history(f'results/logs/{model_name}_history.json')
        # plot_learning_curves(history, model_name,
        #                      f'results/figures/learning_curves_{model_name}.png')
        pass
```

---

## 📝 Checklist Final de Implementação

### Dia 1: Ensemble e Robustez

- [ ] Executar `python ensemble.py`
- [ ] Verificar `results/ensemble_comparison.txt`
- [ ] Executar testes de robustez para 3 modelos
- [ ] Consolidar resultados em tabela

### Dia 2: Interpretabilidade

- [ ] Criar `scripts/generate_gradcam.py`
- [ ] Gerar Grad-CAM para 8 amostras × 3 modelos = 24 visualizações
- [ ] Análise qualitativa das visualizações
- [ ] Documentar insights em notebook

### Dia 3: Validação Estatística

- [ ] Criar `scripts/statistical_analysis.py`
- [ ] Executar teste t-pareado
- [ ] Executar McNemar's test
- [ ] Calcular bootstrap CI para todas as métricas
- [ ] Documentar resultados com p-values

### Dia 4: Otimização e Análise Final

- [ ] Criar `scripts/optimize_threshold.py`
- [ ] Encontrar threshold ótimo para cada modelo
- [ ] Re-avaliar com thresholds otimizados
- [ ] Plotar curvas de aprendizado
- [ ] Análise de overfitting

### Dia 5: Consolidação

- [ ] Criar tabelas finais para o artigo
- [ ] Gerar todas as figuras em alta resolução
- [ ] Atualizar `results/overview.md` com todos os resultados
- [ ] Revisar consistência de números

---

## 🎯 Outputs Esperados

Após implementar todas as tarefas, você terá:

### Arquivos de Resultados:

- ✅ `results/ensemble_comparison.txt` - Comparação completa
- ✅ `results/*_robustness.txt` - Análise de robustez (3 arquivos)
- ✅ `results/robustness_comparison.csv` - Consolidado
- ✅ `results/statistical_analysis.txt` - Validação estatística
- ✅ `results/threshold_optimization.txt` - Thresholds otimizados

### Figuras (Alta Resolução):

- ✅ `results/figures/comparison_*.png` - 3 gráficos de comparação
- ✅ `results/figures/gradcam_*.png` - 3 visualizações Grad-CAM
- ✅ `results/figures/robustness_*.png` - Gráficos de degradação
- ✅ `results/figures/roc_comparison.png` - Curvas ROC sobrepostas
- ✅ `results/figures/confusion_matrix_*.png` - Matrizes de confusão
- ✅ `results/figures/learning_curves_*.png` - Curvas de aprendizado

### Dados para o Artigo:

- ✅ Tabela de métricas de ensemble
- ✅ Tabela de robustez
- ✅ P-values para significância
- ✅ Confidence intervals
- ✅ Visualizações interpretáveis

---

## ⏰ Cronograma de Execução

| Dia   | Tarefas                        | Duração | Outputs                            |
| ----- | ------------------------------ | ------- | ---------------------------------- |
| **1** | Ensemble + Robustez            | 3h      | Tabelas de resultados              |
| **2** | Grad-CAM + Análise qualitativa | 2h      | 24 visualizações                   |
| **3** | Validação estatística          | 2h      | P-values, CIs                      |
| **4** | Threshold tuning + Curvas      | 2h      | Otimização, análise de overfitting |
| **5** | Consolidação e revisão         | 2h      | Documento final                    |

**Total:** ~11 horas de trabalho focado

---

## 💡 Dicas de Execução

1. **Execute em ordem:** Cada tarefa depende da anterior
2. **Salve checkpoints:** Cada script deve salvar resultados intermediários
3. **Documente tudo:** Adicione comentários sobre decisões tomadas
4. **Verifique consistência:** Números devem bater entre diferentes análises
5. **Backup:** Salve cópias dos modelos e resultados

---

**Boa implementação! 🚀**

**Próxima ação:** `python ensemble.py`
