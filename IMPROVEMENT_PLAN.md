# Plano de Melhorias - Redução de Falsos Positivos e Aumento de Acurácia

## 📊 Análise do Problema Atual

### Situação Atual (EfficientNet-B0 - Melhor Modelo)

- ✅ **Accuracy**: 80.29%
- ✅ **AUC**: 0.9761 (excelente)
- ✅ **Sensitivity**: 99.74% (quase perfeito)
- ❌ **Specificity**: 47.86% (CRÍTICO - 52% de falsos positivos)
- ❌ **122 falsos positivos** de 234 casos normais

### Diagnóstico do Problema

**Por que temos tantos falsos positivos?**

1. **Desbalanceamento severo** (1:2.89 ratio Normal:Pneumonia)

   - Modelo aprende viés para classe majoritária (Pneumonia)
   - Class weights (1.945/0.673) não foram suficientes

2. **Loss function inadequada** (CrossEntropyLoss standard)

   - Não foca em exemplos difíceis
   - Trata todos os erros igualmente

3. **Threshold fixo** (0.5)

   - Não otimizado para balancear Sensitivity/Specificity
   - Favorece classe com maior probabilidade média

4. **Augmentation limitado**

   - Apenas horizontal flip
   - Não simula variações reais de imagem médica

5. **Validação pequena** (16 samples)

   - Early stopping instável
   - Pode ter parado longe do ótimo

6. **Ensemble fraco**
   - Modelos correlacionados (todos CNNs em ImageNet)
   - Weighted voting ineficaz (pesos quase iguais)

---

## 🎯 Objetivos de Melhoria

### Metas Quantitativas

- **Specificity**: 47.86% → **≥ 65%** (reduzir FP de 122 para ~82)
- **Accuracy**: 80.29% → **≥ 85%**
- **Sensitivity**: Manter **≥ 95%** (máximo 19 FN)
- **Balanced Accuracy**: 73.80% → **≥ 80%**

### Estratégia

1. **Reduzir falsos positivos** é prioridade #1
2. **Manter sensibilidade alta** (custo de FN é crítico)
3. **Melhorar generalização** (cross-validation)

---

## 📋 TASKS - Implementação Recomendada

### 🔴 PRIORIDADE ALTA (Impacto Imediato - 1-2 semanas)

#### TASK 1: Threshold Optimization ⭐⭐⭐⭐⭐

**Objetivo**: Encontrar threshold ótimo para Specificity ≥ 65% mantendo Sensitivity ≥ 95%

**Métodos a implementar**:

1. **Youden's Index**: $J = \text{Sensitivity} + \text{Specificity} - 1$
2. **F1-Score Maximization**: Otimizar balanceamento Precision-Recall
3. **Target Specificity**: Fixar Spec=65%, encontrar threshold
4. **Cost-Sensitive**: Custo(FN)=10, Custo(FP)=1 (ajustar por contexto clínico)

**Implementação**:

```python
# threshold_optimization.py
def optimize_threshold(y_true, y_probs, method='youden'):
    thresholds = np.linspace(0, 1, 1000)
    best_threshold = 0.5
    best_score = 0

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)

        if method == 'youden':
            score = sensitivity + specificity - 1
        elif method == 'target_spec':
            if specificity >= 0.65:
                score = sensitivity
        # ...

    return best_threshold
```

**Tempo estimado**: 2-3 dias
**Impacto esperado**: Specificity +15-20%, Sensitivity -2-4%
**Risco**: Baixo (não requer re-treinamento)

---

#### TASK 2: Focal Loss Implementation ⭐⭐⭐⭐⭐

**Objetivo**: Focar aprendizado em exemplos difíceis (hard negatives)

**Teoria**:
$$FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$$

onde:

- $p_t = p$ se $y=1$, senão $1-p$
- $\gamma = 2.0$ (focusing parameter) - reduz peso de exemplos fáceis
- $\alpha_t$ = class weight (1.945 para Normal, 0.673 para Pneumonia)

**Por que funciona?**

- Exemplos fáceis (bem classificados): $(1-p_t)^\gamma \approx 0$ → peso baixo
- Exemplos difíceis (mal classificados): $(1-p_t)^\gamma \approx 1$ → peso alto
- Força modelo a aprender casos limítrofes (onde ocorrem FP/FN)

**Implementação**:

```python
# src/losses.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # [1.945, 0.673] para nosso dataset
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()
```

**Modificação no treinamento**:

```python
# train.py
alpha = torch.tensor([1.945, 0.673]).to(device)
criterion = FocalLoss(alpha=alpha, gamma=2.0)
```

**Tempo estimado**: 1-2 dias implementação + 8-10 horas re-treinamento
**Impacto esperado**: Specificity +8-12%, Balanced Acc +5-7%
**Risco**: Médio (requer re-treinamento completo)

---

#### TASK 3: Cross-Validation (K=5 Stratified) ⭐⭐⭐⭐

**Objetivo**: Validação robusta + usar 100% dos dados de treino

**Problemas atuais**:

- Validação = 16 samples (0.3% do dataset!)
- Early stopping baseado em conjunto minúsculo
- Desperdício de ~1,325 samples (não usa validação oficial no treino)

**Estratégia**:

```
Original Training Set (5,216 samples)
    ↓ Split K=5 stratified
Fold 1: Train 4,173 | Val 1,043
Fold 2: Train 4,173 | Val 1,043
Fold 3: Train 4,173 | Val 1,043
Fold 4: Train 4,173 | Val 1,043
Fold 5: Train 4,173 | Val 1,043
    ↓ Aggregate
Final: Train em 100% | Métricas = média(5 folds) ± std
```

**Benefícios**:

- ✅ Validação em 1,043 samples (vs 16 atual) = 65× mais robusto
- ✅ Early stopping confiável
- ✅ Intervalos de confiança (95% CI) para todas as métricas
- ✅ Detecção de overfitting mais precisa

**Implementação**:

```python
# src/cross_validation.py
from sklearn.model_selection import StratifiedKFold

def cross_validate_model(model_class, config, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Treinar modelo no fold
        model = train_fold(model_class, train_idx, val_idx, config)

        # Avaliar no test set (mesmo para todos os folds)
        metrics = evaluate(model, test_loader)
        results.append(metrics)

    # Agregar resultados
    mean_metrics = np.mean(results, axis=0)
    std_metrics = np.std(results, axis=0)

    return mean_metrics, std_metrics
```

**Tempo estimado**: 3-4 dias implementação + 40-50 horas treino (5 folds × 8-10h)
**Impacto esperado**: Accuracy +2-3%, métricas mais confiáveis
**Risco**: Baixo (metodologia padrão)

**Recomendação**: Executar em paralelo ou usar cloud computing (AWS, GCP) para acelerar.

---

#### TASK 4: Advanced Medical Augmentation ⭐⭐⭐⭐

**Objetivo**: Simular variações realistas de raio-X para melhor generalização

**Augmentations médicos específicos**:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**

   - Melhora contraste local sem amplificar ruído
   - Simula variações de exposição de raio-X

   ```python
   transforms.Lambda(lambda img: clahe(img, clip_limit=2.0, tile_grid_size=(8,8)))
   ```

2. **Elastic Deformation**

   - Simula variações anatômicas (posicionamento do paciente)
   - Mantém estruturas anatômicas realistas

   ```python
   A.ElasticTransform(alpha=1, sigma=50, p=0.3)
   ```

3. **Grid Distortion**

   - Simula distorções de lente/detector

   ```python
   A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3)
   ```

4. **Gaussian Noise**

   - Simula ruído de detector

   ```python
   A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
   ```

5. **Brightness/Contrast**

   - Variações de exposição

   ```python
   A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
   ```

6. **Shift/Scale/Rotate**

   - Variações de posicionamento

   ```python
   A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5)
   ```

7. **Coarse Dropout**
   - Simula oclusões/artifacts
   ```python
   A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)
   ```

**Implementação completa**:

```python
# src/advanced_augmentation.py
import albumentations as A

def get_advanced_train_transform(image_size=224):
    return A.Compose([
        # Geometric
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),

        # Elastic & Grid
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),

        # Intensity
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # Noise & Quality
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),

        # Cutout/Dropout
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                        min_holes=1, min_height=8, min_width=8, p=0.3),

        # Final resize & normalize
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

**Tempo estimado**: 2 dias implementação + 8-10 horas re-treinamento
**Impacto esperado**: Accuracy +3-5%, melhor generalização
**Risco**: Baixo (augmentation é padrão)

---

#### TASK 5: Test-Time Augmentation (TTA) ⭐⭐⭐⭐

**Objetivo**: Reduzir variância de predições através de múltiplas versões da mesma imagem

**Conceito**:

```
Imagem original
    ↓ Gera N augmentations
Aug 1 → Prediction p1
Aug 2 → Prediction p2
...
Aug N → Prediction pN
    ↓ Agregar
Final prediction = mean([p1, p2, ..., pN])
```

**Benefícios**:

- ✅ Predições mais estáveis e confiantes
- ✅ Reduz impacto de augmentations específicos
- ✅ Melhora AUC e calibração
- ✅ Não requer re-treinamento!

**Implementação**:

```python
# src/tta.py
def predict_with_tta(model, image, n_augmentations=5):
    """
    Apply Test-Time Augmentation

    Args:
        model: Trained model
        image: Input image tensor [C, H, W]
        n_augmentations: Number of augmented versions (default 5)

    Returns:
        Average prediction across all augmentations
    """
    model.eval()
    predictions = []

    # Original image
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predictions.append(F.softmax(output, dim=1))

    # Augmented versions
    tta_transforms = [
        A.HorizontalFlip(p=1.0),
        A.Rotate(limit=5, p=1.0),
        A.Rotate(limit=-5, p=1.0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0, rotate_limit=0, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
    ]

    for transform in tta_transforms[:n_augmentations-1]:
        aug_image = transform(image=image.numpy())['image']
        aug_image = torch.from_numpy(aug_image)

        with torch.no_grad():
            output = model(aug_image.unsqueeze(0))
            predictions.append(F.softmax(output, dim=1))

    # Average predictions
    final_pred = torch.mean(torch.stack(predictions), dim=0)
    return final_pred
```

**Uso**:

```python
# Durante teste/inferência
for images, labels in test_loader:
    predictions = []
    for image in images:
        pred = predict_with_tta(model, image, n_augmentations=5)
        predictions.append(pred)
    # Avaliar
```

**Tempo estimado**: 1-2 dias implementação
**Impacto esperado**: AUC +0.01-0.02, Accuracy +1-2%
**Risco**: Muito baixo (apenas inferência)

---

### 🟡 PRIORIDADE MÉDIA (Melhorias Arquiteturais - 2-4 semanas)

#### TASK 6: Ensemble Inteligente - Stacking ⭐⭐⭐⭐

**Objetivo**: Superar melhor modelo individual através de meta-learning

**Por que Simple/Weighted Voting falhou?**

- Modelos fracos (ResNet/DenseNet) "puxam para baixo"
- Pesos fixos não se adaptam a características da imagem
- Não aprende quando confiar em cada modelo

**Solução: Stacked Generalization**

```
Level 0 (Base Models):
    EfficientNet-B0 → Predictions P1
    ResNet-50       → Predictions P2
    DenseNet-121    → Predictions P3
            ↓ Concatenate
    Features: [P1, P2, P3]
            ↓
Level 1 (Meta-Learner):
    Logistic Regression / XGBoost / LightGBM
            ↓
    Final Prediction
```

**Meta-learner aprende**:

- Quando EfficientNet é mais confiável
- Quando ResNet detecta algo que outros perdem
- Padrões de concordância/discordância

**Implementação**:

```python
# src/stacking_ensemble.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb

class StackingEnsemble:
    def __init__(self, base_models, meta_model='logistic'):
        self.base_models = base_models

        if meta_model == 'logistic':
            self.meta_model = LogisticRegression(max_iter=1000)
        elif meta_model == 'xgboost':
            self.meta_model = GradientBoostingClassifier(n_estimators=100)
        elif meta_model == 'lightgbm':
            self.meta_model = lgb.LGBMClassifier(n_estimators=100)

    def fit(self, X_val, y_val):
        """
        Treinar meta-model nas predições dos base models
        """
        # Gerar predições de todos os base models
        base_predictions = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                preds = model(X_val)
                probs = F.softmax(preds, dim=1).cpu().numpy()
                base_predictions.append(probs)

        # Concatenar: shape (N_samples, N_models * N_classes)
        X_meta = np.hstack(base_predictions)

        # Treinar meta-model
        self.meta_model.fit(X_meta, y_val.cpu().numpy())

    def predict(self, X_test):
        """
        Predição com stacking
        """
        base_predictions = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                preds = model(X_test)
                probs = F.softmax(preds, dim=1).cpu().numpy()
                base_predictions.append(probs)

        X_meta = np.hstack(base_predictions)
        return self.meta_model.predict(X_meta)
```

**Variantes a testar**:

1. **Logistic Regression** (simples, interpretável)
2. **XGBoost** (não-linear, captura interações)
3. **LightGBM** (mais rápido, similar ao XGBoost)
4. **Neural Network** (1-2 camadas, máxima capacidade)

**Tempo estimado**: 3-5 dias
**Impacto esperado**: Accuracy +3-5% vs EfficientNet individual
**Risco**: Médio (pode overfittar se validação pequena - usar com cross-validation!)

---

#### TASK 7: Arquiteturas Modernas - Vision Transformer ⭐⭐⭐

**Objetivo**: Capturar dependências long-range (contexto anatômico global)

**Limitação de CNNs**:

- Receptive field limitado
- Dificulta capturar relações entre regiões distantes (ex: coração + pulmões)

**Vantagem de Transformers**:

- Atenção global desde a primeira camada
- Captura contexto completo da imagem

**Modelos recomendados**:

1. **ViT-Base** (86M params)
   - Vision Transformer original
   - Pré-treinado em ImageNet-21k
2. **Swin Transformer-Tiny** (28M params)
   - Shifted windows (eficiência)
   - Hierárquico (multi-scale features)
3. **BEiT-Base** (86M params)

   - Self-supervised pré-training
   - Melhor para domínios específicos

4. **ConvNeXt-Tiny** (28M params)
   - Hybrid CNN-Transformer
   - Eficiência de CNN + capacidade de Transformer

**Implementação**:

```python
# src/models.py
import timm

def get_vision_transformer(model_name='vit_base_patch16_224', num_classes=2):
    """
    Load Vision Transformer
    """
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )
    return model

# Opções:
# - 'vit_base_patch16_224' (ViT-Base)
# - 'swin_tiny_patch4_window7_224' (Swin-Tiny)
# - 'beit_base_patch16_224' (BEiT-Base)
# - 'convnext_tiny' (ConvNeXt-Tiny)
```

**Estratégia de treinamento**:

- Mesmo progressive unfreezing (5 épocas head-only + 20 full)
- Learning rate menor: 5e-5 (Transformers são sensíveis)
- Gradient clipping: max_norm=1.0 (estabilidade)
- Warmup: 5 épocas com LR crescente (0 → 5e-5)

**Tempo estimado**: 5-7 dias
**Impacto esperado**: Accuracy +2-5% (se dataset suficiente)
**Risco**: Alto (pode overfittar; requer mais dados ou regularização forte)

---

#### TASK 8: Mixup / CutMix Augmentation ⭐⭐⭐

**Objetivo**: Regularização avançada através de interpolação de exemplos

**Mixup**:

```python
# Interpola duas imagens
x_mixed = λ * x_i + (1-λ) * x_j
y_mixed = λ * y_i + (1-λ) * y_j

# λ ~ Beta(α, α), α=0.2
```

**CutMix**:

```python
# Recorta região de x_j e cola em x_i
x_cutmix = M ⊙ x_i + (1-M) ⊙ x_j
y_cutmix = λ * y_i + (1-λ) * y_j

# λ = área da região recortada / área total
```

**Por que funciona?**

- Força modelo a não depender de regiões específicas
- Reduz overfitting
- Melhora calibração (predições mais confiantes)

**Implementação**:

```python
# src/mixup_cutmix.py
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Random box
    W, H = x.size(2), x.size(3)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam

# No training loop:
if use_mixup:
    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
    outputs = model(inputs)
    loss = lam * criterion(outputs, targets_a) + (1-lam) * criterion(outputs, targets_b)
```

**Tempo estimado**: 2-3 dias
**Impacto esperado**: Accuracy +1-3%, melhor generalização
**Risco**: Baixo

---

### 🟢 PRIORIDADE BAIXA (Otimizações Avançadas - 1-2 meses)

#### TASK 9: Class-Balanced Loss ⭐⭐⭐

**Objetivo**: Lidar com desbalanceamento através de re-ponderação baseada em frequência efetiva

**Teoria**:
$$w_c = \frac{1 - \beta}{1 - \beta^{n_c}}$$

onde:

- $n_c$ = número de samples da classe $c$
- $\beta \in [0, 1)$ (tipicamente 0.9999)

**Intuição**: Classes raras têm peso maior, mas não linear (evita overweight extremo)

**Implementação**:

```python
# src/losses.py
def get_class_balanced_weights(samples_per_class, beta=0.9999):
    effective_num = 1.0 - np.power(beta, samples_per_class)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)

# Uso:
samples_per_class = [1341, 3875]  # Normal, Pneumonia
cb_weights = get_class_balanced_weights(samples_per_class, beta=0.9999)
criterion = nn.CrossEntropyLoss(weight=cb_weights)
```

**Tempo estimado**: 1 dia + re-treinamento
**Impacto esperado**: Balanced Acc +2-4%
**Risco**: Baixo

---

#### TASK 10: Self-Supervised Pre-training ⭐⭐⭐

**Objetivo**: Pré-treinar em dados de raio-X (sem labels) antes de fine-tuning

**Abordagens**:

1. **SimCLR**: Contrastive learning (imagens similares = embeddings próximos)
2. **MoCo**: Momentum Contrast (queue de negative examples)
3. **BYOL**: Bootstrap Your Own Latent (sem negatives)
4. **MAE**: Masked Autoencoder (reconstruir patches mascarados)

**Por que funciona?**

- ImageNet tem fotos naturais; raio-X é domínio diferente
- Self-supervised aprende features específicas de raio-X
- Pode usar datasets grandes não-anotados (ChestX-ray14, MIMIC-CXR)

**Pipeline**:

```
1. Coletar raio-X não-anotados (100K-1M imagens)
2. Pré-treinar com SimCLR/MAE (1-2 semanas GPU)
3. Fine-tunar no nosso dataset (pneumonia)
4. Comparar com ImageNet pré-training
```

**Tempo estimado**: 3-4 semanas
**Impacto esperado**: Accuracy +3-7% (se dataset grande disponível)
**Risco**: Alto (requer expertise, computação)

---

#### TASK 11: Ensemble de Ensembles ⭐⭐

**Objetivo**: Combinar múltiplos ensembles treinados com seeds diferentes

**Estratégia**:

```
Seed 1: Train 3 models → Ensemble 1
Seed 2: Train 3 models → Ensemble 2
Seed 3: Train 3 models → Ensemble 3
    ↓ Aggregate
Final: Average(Ensemble 1, Ensemble 2, Ensemble 3)
```

**Benefícios**:

- Reduz variância de inicialização
- Mais robusto a outliers
- Melhora calibração

**Tempo estimado**: 1 semana (treino massivo)
**Impacto esperado**: Accuracy +1-2%
**Risco**: Alto (custo computacional)

---

## 🎯 Roadmap Recomendado

### Fase 1: Quick Wins (1-2 semanas) - SEM RE-TREINAMENTO

**Objetivo**: Melhorias imediatas sem custo computacional

1. ✅ **TASK 1: Threshold Optimization** (2-3 dias)

   - Implementar 4 métodos
   - Validar em test set
   - **Expectativa**: Spec 47% → 62-65%, Sens 99% → 95-97%

2. ✅ **TASK 5: Test-Time Augmentation** (1-2 dias)
   - Implementar TTA com 5 augmentations
   - **Expectativa**: AUC +0.01-0.02, Acc +1-2%

**Resultado esperado Fase 1**:

- Accuracy: 80.29% → **81-82%**
- Specificity: 47.86% → **62-65%**
- Sensitivity: 99.74% → **95-97%**
- Balanced Acc: 73.80% → **78-81%**

---

### Fase 2: Re-training com Melhorias (2-3 semanas)

**Objetivo**: Re-treinar modelos com técnicas avançadas

3. ✅ **TASK 2: Focal Loss** (1-2 dias + 10h treino)

   - Implementar Focal Loss (γ=2.0)
   - Re-treinar EfficientNet-B0
   - **Expectativa**: Spec +8-12%

4. ✅ **TASK 4: Advanced Augmentation** (2 dias + 10h treino)

   - Implementar 12+ augmentations
   - Re-treinar com novo pipeline
   - **Expectativa**: Acc +3-5%

5. ✅ **TASK 3: Cross-Validation** (3-4 dias + 50h treino)
   - Implementar K=5 stratified CV
   - Treinar 5 modelos (paralelo se possível)
   - **Expectativa**: Métricas mais confiáveis, Acc +2-3%

**Resultado esperado Fase 2**:

- Accuracy: 82% → **85-87%**
- Specificity: 65% → **68-72%**
- Sensitivity: **95-97%** (mantido)
- Balanced Acc: 81% → **83-86%**

---

### Fase 3: Ensemble Inteligente (1 semana)

**Objetivo**: Superar melhor modelo individual

6. ✅ **TASK 6: Stacking Ensemble** (3-5 dias)
   - Treinar meta-learner (LightGBM)
   - Comparar com Simple/Weighted Voting
   - **Expectativa**: Acc +2-4% vs melhor individual

**Resultado esperado Fase 3**:

- Accuracy: 87% → **88-90%**
- Specificity: **70-75%**
- Sensitivity: **95-97%**
- Balanced Acc: **86-88%**

---

### Fase 4: Arquiteturas Modernas (Opcional - 2-3 semanas)

**Objetivo**: Estado da arte absoluto

7. ✅ **TASK 7: Vision Transformer** (5-7 dias)
   - Treinar Swin Transformer
   - Comparar com EfficientNet
8. ✅ **TASK 8: Mixup/CutMix** (2-3 dias)
   - Adicionar ao pipeline de treino

**Resultado esperado Fase 4**:

- Accuracy: **90-92%**
- Specificity: **75-80%**
- Sensitivity: **95-97%**
- Balanced Acc: **88-90%**

---

## 📊 Comparação de Impacto vs Esforço

| Task                          | Impacto    | Esforço                     | Risco       | Prioridade |
| ----------------------------- | ---------- | --------------------------- | ----------- | ---------- |
| **1. Threshold Optimization** | ⭐⭐⭐⭐⭐ | Baixo (2-3 dias)            | Baixo       | 🔴 ALTA    |
| **2. Focal Loss**             | ⭐⭐⭐⭐⭐ | Médio (2 dias + 10h treino) | Médio       | 🔴 ALTA    |
| **3. Cross-Validation**       | ⭐⭐⭐⭐   | Alto (4 dias + 50h treino)  | Baixo       | 🔴 ALTA    |
| **4. Advanced Augmentation**  | ⭐⭐⭐⭐   | Médio (2 dias + 10h treino) | Baixo       | 🔴 ALTA    |
| **5. Test-Time Augmentation** | ⭐⭐⭐⭐   | Baixo (1-2 dias)            | Muito Baixo | 🔴 ALTA    |
| **6. Stacking Ensemble**      | ⭐⭐⭐⭐   | Médio (3-5 dias)            | Médio       | 🟡 MÉDIA   |
| **7. Vision Transformer**     | ⭐⭐⭐     | Alto (5-7 dias)             | Alto        | 🟡 MÉDIA   |
| **8. Mixup/CutMix**           | ⭐⭐⭐     | Baixo (2-3 dias)            | Baixo       | 🟡 MÉDIA   |
| **9. Class-Balanced Loss**    | ⭐⭐⭐     | Baixo (1 dia + treino)      | Baixo       | 🟢 BAIXA   |
| **10. Self-Supervised**       | ⭐⭐⭐     | Muito Alto (3-4 semanas)    | Alto        | 🟢 BAIXA   |
| **11. Ensemble de Ensembles** | ⭐⭐       | Muito Alto (1 semana)       | Alto        | 🟢 BAIXA   |

---

## ✅ Checklist de Implementação

### Antes de começar qualquer TASK:

- [ ] **Backup dos modelos atuais**

  ```bash
  cp -r models/ models_backup_$(date +%Y%m%d)/
  ```

- [ ] **Criar branch Git**

  ```bash
  git checkout -b improvements/task-name
  ```

- [ ] **Documentar baseline atual**

  - Salvar todas as métricas atuais
  - Registrar hiperparâmetros
  - Anotar tempo de treinamento

- [ ] **Configurar logging detalhado**
  ```python
  import wandb  # ou TensorBoard
  wandb.init(project="pneumonia-improvements")
  ```

### Durante implementação:

- [ ] **Commits frequentes**

  ```bash
  git commit -m "feat: implement threshold optimization"
  ```

- [ ] **Validação incremental**

  - Testar cada componente isoladamente
  - Comparar com baseline após cada mudança

- [ ] **Monitorar recursos**
  - GPU memory usage
  - Training time
  - Disk space

### Após cada TASK:

- [ ] **Análise comparativa**

  - Gerar tabela comparativa (antes vs depois)
  - Calcular significância estatística
  - Visualizar métricas

- [ ] **Documentar resultados**

  ```markdown
  ## TASK X: Nome

  - Implementação: [data]
  - Baseline: Acc=80.29%, Spec=47.86%
  - Resultado: Acc=85.12%, Spec=65.43%
  - Ganho: +4.83% Acc, +17.57% Spec
  - Tempo: 12h treino
  - Observações: ...
  ```

- [ ] **Merge se bem-sucedido**
  ```bash
  git checkout main
  git merge improvements/task-name
  git push
  ```

---

## 🚀 Próximos Passos Imediatos

### Esta Semana (Começar AGORA):

1. **Segunda-feira**: Implementar TASK 1 (Threshold Optimization)

   - Código: `src/threshold_optimization.py`
   - Testar 4 métodos
   - Validar em test set
   - **Meta**: Spec ≥ 65%

2. **Terça-feira**: Implementar TASK 5 (TTA)

   - Código: `src/tta.py`
   - Testar com 5 augmentations
   - Medir impacto em AUC

3. **Quarta-Quinta**: Implementar TASK 2 (Focal Loss)

   - Código: `src/losses.py` (já existe!)
   - Modificar `train.py`
   - Iniciar re-treinamento

4. **Sexta**: Análise de resultados Fase 1
   - Comparar threshold optimization + TTA
   - Documentar ganhos
   - Decidir próximos passos

### Próxima Semana:

5. **TASK 4**: Advanced Augmentation (2 dias)
6. **TASK 3**: Cross-Validation (iniciar - rodar em background/cloud)

---

## 📈 Expectativas Finais

### Após todas as melhorias (Fases 1-3):

| Métrica              | Atual   | Meta         | Melhoria                    |
| -------------------- | ------- | ------------ | --------------------------- |
| **Accuracy**         | 80.29%  | **≥ 88%**    | +7.71%                      |
| **AUC**              | 0.9761  | **≥ 0.98**   | +0.004                      |
| **F1-Score**         | 0.8635  | **≥ 0.90**   | +0.037                      |
| **Sensitivity**      | 99.74%  | **95-97%**   | -2-4% (trade-off aceitável) |
| **Specificity**      | 47.86%  | **≥ 70%**    | +22.14%                     |
| **Balanced Acc**     | 73.80%  | **≥ 86%**    | +12.20%                     |
| **Falsos Positivos** | 122/234 | **≤ 70/234** | -52 casos                   |
| **Falsos Negativos** | 1/390   | **≤ 19/390** | +18 casos (aceitável)       |

### Impacto clínico:

**Antes** (EfficientNet-B0 atual):

- ✅ Excelente detecção de pneumonia (99.74% sens)
- ❌ Muitos falsos alarmes (52% FPR)
- ⚠️ Sobrecarga de radiologistas

**Depois** (com todas as melhorias):

- ✅ Ótima detecção de pneumonia (95-97% sens)
- ✅ Falsos alarmes reduzidos (30% FPR)
- ✅ Carga de trabalho viável
- ✅ **Pronto para uso clínico com supervisão**

---

## 🎓 Referências e Recursos

### Papers importantes:

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
2. **Mixup**: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
3. **CutMix**: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers", ICCV 2019
4. **Class-Balanced Loss**: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
5. **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
6. **Swin Transformer**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021

### Código de referência:

- **timm**: https://github.com/huggingface/pytorch-image-models
- **Albumentations**: https://albumentations.ai/
- **Focal Loss**: https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py

---

## 💬 Perguntas Frequentes

**Q: Devo implementar todas as tasks?**
A: Não! Siga o roadmap por fases. Fase 1-2 já deve alcançar 85-87% accuracy.

**Q: E se Focal Loss não funcionar?**
A: Tente Class-Balanced Loss (TASK 9) ou ajuste γ (testar 1.0, 1.5, 2.0, 2.5).

**Q: Cross-validation demora muito. Alternativas?**
A: Use apenas 3 folds (K=3) ou treine em cloud (AWS/GCP) com múltiplas GPUs.

**Q: Ensemble continua falhando. O que fazer?**
A: Implemente TASK 6 (Stacking). Se ainda falhar, foque em melhorar o EfficientNet individual.

**Q: Vale a pena usar Vision Transformer?**
A: Apenas se você tiver GPU potente (≥ 16GB VRAM) ou cloud. EfficientNet + melhorias já alcança 85-88%.

**Q: Como sei se está funcionando?**
A: Monitore Balanced Accuracy e Specificity. Se ambos subirem, você está no caminho certo.

---

**Autor**: AI Assistant  
**Data**: 14 de Novembro de 2025  
**Versão**: 1.0  
**Status**: Pronto para implementação
