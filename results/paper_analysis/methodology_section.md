# SEÇÃO 3: METODOLOGIA

## 3.1 Dataset e Pré-processamento

### 3.1.1 Descrição do Dataset

Utilizamos o **Chest X-Ray Images (Pneumonia)** dataset [1], disponível publicamente no Kaggle, contendo 5,863 imagens de raio-X torácico de pacientes pediátricos (1-5 anos) do Guangzhou Women and Children's Medical Center.

**Distribuição do dataset:**

| Conjunto | Normal | Pneumonia | Total | Ratio |
|----------|--------|-----------|-------|-------|
| **Training** | 1,341 | 3,875 | 5,216 | 1:2.89 |
| **Validation** | 8 | 8 | 16 | 1:1.00 |
| **Test** | 234 | 390 | 624 | 1:1.67 |
| **Total** | 1,583 | 4,273 | 5,856 | 1:2.70 |

**Características:**
- Formato: JPEG (RGB convertido para Grayscale)
- Resolução: Variável (1000-3000 pixels)
- Anotações: Validadas por especialistas radiologistas
- Classes: Binary (NORMAL vs PNEUMONIA - bacterial/viral)

### 3.1.2 Pipeline de Pré-processamento

Implementamos um pipeline de pré-processamento padronizado para todas as arquiteturas:

```python
# Transformações de treino
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Resize para input padrão
    transforms.RandomHorizontalFlip(p=0.5),  # Augmentation simples
    transforms.ToTensor(),
    transforms.Normalize(                     # Normalização ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Transformações de validação/teste
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**Decisões de design:**
1. **Resize 224×224**: Compatível com todas as arquiteturas pré-treinadas
2. **ImageNet normalization**: Preserva distribuição dos pesos pré-treinados
3. **Horizontal flip (treino)**: Única augmentation, pois orientação anatômica importa
4. **Sem vertical flip**: Anatomia pulmonar não é verticalmente simétrica

### 3.1.3 Data Loading e Batching

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Otimização para GPU
)
```

**Configurações:**
- **Batch size**: 32 (balanceamento memória GPU × convergência)
- **Shuffle**: True (previne overfitting de ordem)
- **Workers**: 4 (paralelização de carregamento)
- **Pin memory**: True (transferência CPU→GPU mais rápida)

## 3.2 Arquiteturas de Transfer Learning

### 3.2.1 Seleção de Modelos

Selecionamos três arquiteturas state-of-the-art com características complementares:

#### EfficientNet-B0 [2]

**Compound Scaling Principle:**

$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$

onde $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ e $\alpha, \beta, \gamma \geq 1$

**Especificações:**
- Parâmetros: 5.3M
- FLOPs: 0.39B
- Input: 224×224
- Profundidade: 18 layers (7 MBConv blocks)
- Vantagens: Eficiência computacional, balanceamento de escalas

#### ResNet-50 [3]

**Residual Connection:**

$$\mathcal{F}(x) = \mathcal{H}(x) - x$$

onde $\mathcal{H}(x) = \text{conv}_2(\text{ReLU}(\text{conv}_1(x)))$

**Especificações:**
- Parâmetros: 25.6M
- FLOPs: 4.1B
- Input: 224×224
- Profundidade: 50 layers (16 residual blocks)
- Vantagens: Skip connections, gradiente estável

#### DenseNet-121 [4]

**Dense Connectivity:**

$$x_\ell = \mathcal{H}_\ell([x_0, x_1, \ldots, x_{\ell-1}])$$

onde $[x_0, x_1, \ldots, x_{\ell-1}]$ denota concatenação de feature maps

**Especificações:**
- Parâmetros: 8.0M
- FLOPs: 2.9B
- Input: 224×224
- Profundidade: 121 layers (4 dense blocks)
- Vantagens: Feature reuse, gradiente direto

### 3.2.2 Justificativa da Seleção

| Arquitetura | Parâmetros | Vantagem Principal | Trade-off |
|-------------|------------|-------------------|----------|
| **EfficientNet-B0** | 5.3M | Eficiência | Menos capacidade |
| **ResNet-50** | 25.6M | Robustez | Alto custo computacional |
| **DenseNet-121** | 8.0M | Feature reuse | Memória intensiva |

**Diversidade arquitetural:**
1. EfficientNet: Compound scaling (profundidade × largura × resolução)
2. ResNet: Residual learning (skip connections)
3. DenseNet: Dense connectivity (feature concatenation)

## 3.3 Estratégia de Fine-tuning

### 3.3.1 Progressive Unfreezing

Implementamos fine-tuning em duas fases para evitar catastrophic forgetting:

**Fase 1: Classifier-only Training (5 épocas)**

```python
# Congelar feature extractor
for param in model.parameters():
    param.requires_grad = False

# Descongelar classificador
for param in model.classifier.parameters():  # ou model.fc
    param.requires_grad = True
```

**Objetivo**: Adaptar classificador ao novo domínio (raio-X torácico) sem perturbar features pré-treinadas (ImageNet).

**Fase 2: Full Fine-tuning (20 épocas)**

```python
# Descongelar todas as camadas
for param in model.parameters():
    param.requires_grad = True
```

**Objetivo**: Refinar features de baixo nível para características específicas de raio-X.

### 3.3.2 Modificação da Camada de Classificação

Substituímos a última camada (1000 classes ImageNet → 2 classes Pneumonia):

```python
# EfficientNet-B0
model.classifier[1] = nn.Linear(1280, 2)

# ResNet-50
model.fc = nn.Linear(2048, 2)

# DenseNet-121
model.classifier = nn.Linear(1024, 2)
```

**Inicialização**: Kaiming He initialization para camadas convolucionais, Xavier para camadas lineares.

### 3.3.3 Class Weighting

Devido ao desbalanceamento do dataset (1:2.89), aplicamos class weighting na loss function:

$$w_{\text{normal}} = \frac{n_{\text{total}}}{2 \cdot n_{\text{normal}}} = \frac{5216}{2 \cdot 1341} = 1.945$$

$$w_{\text{pneumonia}} = \frac{n_{\text{total}}}{2 \cdot n_{\text{pneumonia}}} = \frac{5216}{2 \cdot 3875} = 0.673$$

```python
class_weights = torch.tensor([1.945, 0.673]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Intuição**: Penalizar mais erros na classe minoritária (NORMAL) para balancear aprendizado.

## 3.4 Configuração de Treinamento

### 3.4.1 Otimização

**Optimizer**: AdamW [5]

$$\theta_{t+1} = \theta_t - \eta \left( \frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_t \right)$$

**Hiperparâmetros:**
- Learning rate inicial: $\eta_0 = 10^{-4}$
- Weight decay: $\lambda = 10^{-4}$
- Betas: $\beta_1 = 0.9, \beta_2 = 0.999$
- Epsilon: $\epsilon = 10^{-8}$

**Justificativa**: AdamW combina momentum adaptativo (Adam) com weight decay correto, prevenindo overfitting.

### 3.4.2 Learning Rate Scheduling

**ReduceLROnPlateau** [6]

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # LR ← LR × 0.5
    patience=3,      # Espera 3 épocas
    min_lr=1e-6
)
```

**Estratégia**: Reduzir learning rate quando validation loss estagna por 3 épocas.

**Sequência esperada:**
- Épocas 1-5: $\eta = 10^{-4}$ (classifier-only)
- Épocas 6-X: $\eta = 10^{-4}$ (full fine-tuning)
- Se plateau: $\eta \rightarrow 5 \times 10^{-5} \rightarrow 2.5 \times 10^{-5} \rightarrow \ldots$

### 3.4.3 Early Stopping

Implementamos early stopping para prevenir overfitting:

```python
early_stopping = EarlyStopping(
    patience=7,           # 7 épocas sem melhora
    min_delta=0.001,      # Melhora mínima de 0.1%
    mode='min',           # Monitorar validation loss
    restore_best=True     # Restaurar melhores pesos
)
```

**Critério**: Parar treinamento se validation loss não melhorar por 7 épocas consecutivas.

### 3.4.4 Regularização

**Técnicas aplicadas:**

1. **Weight decay**: $\lambda = 10^{-4}$ (L2 regularization)
   $$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{CE}} + \lambda \sum_i \theta_i^2$$

2. **Dropout**: Presente nas arquiteturas originais
   - EfficientNet: Dropout(0.2) antes do classificador
   - DenseNet: Dropout(0.0) (não usa)
   - ResNet: Dropout(0.0) (não usa)

3. **Data augmentation**: Horizontal flip (p=0.5)

## 3.5 Métodos de Ensemble

### 3.5.1 Simple Voting Ensemble

**Estratégia**: Votação majoritária simples.

$$\hat{y} = \text{mode}(f_1(x), f_2(x), f_3(x))$$

onde $f_i$ é a predição do modelo $i$.

**Implementação:**
```python
predictions = []
for model in models:
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)
        predictions.append(pred)

# Votação majoritária
final_pred = torch.mode(torch.stack(predictions), dim=0)[0]
```

**Vantagens:**
- Simples e interpretável
- Não requer treinamento adicional
- Robusto a outliers

**Desvantagens:**
- Ignora confiança das predições
- Modelos fracos têm mesmo peso

### 3.5.2 Weighted Voting Ensemble

**Estratégia**: Votação ponderada por AUC individual.

$$\hat{y} = \arg\max_c \sum_{i=1}^{N} w_i \cdot P_i(y=c|x)$$

onde $w_i = \frac{\text{AUC}_i}{\sum_j \text{AUC}_j}$ (pesos normalizados)

**Cálculo dos pesos:**

$$w_{\text{EfficientNet}} = \frac{0.9761}{2.8496} = 0.3426$$
$$w_{\text{DenseNet}} = \frac{0.9505}{2.8496} = 0.3336$$
$$w_{\text{ResNet}} = \frac{0.9230}{2.8496} = 0.3238$$

**Implementação:**
```python
weighted_probs = torch.zeros(num_classes)
for model, weight in zip(models, weights):
    model.eval()
    with torch.no_grad():
        output = F.softmax(model(image), dim=1)
        weighted_probs += weight * output

final_pred = torch.argmax(weighted_probs, dim=1)
```

**Vantagens:**
- Considera qualidade dos modelos
- Usa probabilidades (mais informação)
- Teoricamente superior a simple voting

**Desvantagens:**
- Requer métrica de avaliação confiável
- Sensível a overfitting na métrica escolhida

## 3.6 Métricas de Avaliação

### 3.6.1 Métricas Primárias

**1. Accuracy**

$$\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$$

**2. Area Under ROC Curve (AUC)**

$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d[\text{FPR}(t)]$$

onde $\text{TPR} = \frac{TP}{TP + FN}$ e $\text{FPR} = \frac{FP}{FP + TN}$

**3. F1-Score**

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

**4. Sensitivity (Recall)**

$$\text{Sensitivity} = \frac{TP}{TP + FN}$$

**Interpretação clínica**: Proporção de pacientes com pneumonia corretamente identificados.

**5. Specificity**

$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Interpretação clínica**: Proporção de pacientes saudáveis corretamente identificados.

### 3.6.2 Métricas Secundárias

**6. Positive Predictive Value (PPV)**

$$\text{PPV} = \frac{TP}{TP + FP}$$

**7. Negative Predictive Value (NPV)**

$$\text{NPV} = \frac{TN}{TN + FN}$$

**8. Balanced Accuracy**

$$\text{Balanced Acc} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$

### 3.6.3 Critérios de Avaliação Clínica

Para aplicação clínica, estabelecemos thresholds mínimos:

| Métrica | Threshold Mínimo | Justificativa |
|---------|------------------|---------------|
| **Sensitivity** | ≥ 95% | Evitar falsos negativos (casos não tratados) |
| **Specificity** | ≥ 60% | Reduzir sobrecarga de falsos alarmes |
| **AUC** | ≥ 0.90 | Capacidade discriminativa excelente |
| **Balanced Acc** | ≥ 75% | Evitar viés para classe majoritária |

## 3.7 Infraestrutura Computacional

### 3.7.1 Hardware

**Configuração:**
- **CPU**: Apple Silicon M-series (MPS backend)
- **Memória**: 16 GB RAM
- **Storage**: SSD 512 GB
- **Framework**: PyTorch 2.0+ com MPS support

**Device detection automático:**
```python
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
```

### 3.7.2 Software

**Dependências principais:**
- Python 3.8+
- PyTorch 2.0.1
- torchvision 0.15.2
- timm 0.9.2 (PyTorch Image Models)
- scikit-learn 1.3.0
- numpy 1.24.3
- pandas 1.5.3

### 3.7.3 Tempo de Treinamento

| Modelo | Fase 1 (5 épocas) | Fase 2 (20 épocas) | Total |
|--------|-------------------|-------------------|-------|
| EfficientNet-B0 | ~25 min | ~2h 0min | ~2h 25min |
| ResNet-50 | ~35 min | ~2h 50min | ~3h 25min |
| DenseNet-121 | ~30 min | ~2h 20min | ~2h 50min |
| **Total** | - | - | **~8h 40min** |

**Nota**: Tempos medidos em Apple M1 Pro (MPS backend).

## 3.8 Reprodutibilidade

### 3.8.1 Seeds Aleatórias

```python
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 3.8.2 Versionamento

**Dataset**: Chest X-Ray Images (Pneumonia) v2.0 (Kaggle)
**Código**: GitHub repository com commits identificados
**Modelos**: Pesos salvos em `models/` com nomenclatura consistente

### 3.8.3 Disponibilidade

Todo o código-fonte, configurações e resultados estão disponíveis publicamente em:
- **Repositório**: [GitHub link]
- **Licença**: MIT License
- **Dataset**: [Kaggle link]

## 3.9 Pipeline Experimental

### 3.9.1 Workflow Completo

```
1. Download dataset (Kaggle API)
   ↓
2. Pré-processamento (resize, normalize)
   ↓
3. Data loading (DataLoader com batch=32)
   ↓
4. Transfer Learning Setup
   ├── EfficientNet-B0 (ImageNet weights)
   ├── ResNet-50 (ImageNet weights)
   └── DenseNet-121 (ImageNet weights)
   ↓
5. Fine-tuning (Progressive Unfreezing)
   ├── Fase 1: Classifier-only (5 épocas)
   └── Fase 2: Full fine-tuning (20 épocas)
   ↓
6. Avaliação Individual (Test set)
   ├── Accuracy, AUC, F1
   └── Sensitivity, Specificity
   ↓
7. Ensemble Learning
   ├── Simple Voting
   └── Weighted Voting (AUC-based)
   ↓
8. Análise Comparativa
   ├── Performance tables
   ├── ROC curves
   └── Statistical tests
   ↓
9. Validação Clínica
   └── Interpretação de métricas
```

### 3.9.2 Validação Cross-Model

Para garantir fairness na comparação:
- **Mesmo dataset**: Split idêntico para todos os modelos
- **Mesmas transformações**: Pré-processamento padronizado
- **Mesmos hiperparâmetros**: LR, batch size, épocas
- **Mesmo hardware**: Todos treinados no mesmo dispositivo
- **Mesma seed**: Reprodutibilidade garantida

## 3.10 Considerações Éticas

### 3.10.1 Dataset

O dataset utilizado é público e foi coletado com aprovação ética do Guangzhou Women and Children's Medical Center. Imagens foram anonimizadas e não contêm informações de identificação pessoal.

### 3.10.2 Aplicação Clínica

**IMPORTANTE**: Este trabalho é experimental e não deve ser utilizado para diagnóstico clínico real sem:
1. Validação prospectiva em ambiente clínico
2. Aprovação regulatória (FDA, ANVISA, etc.)
3. Integração com workflow médico existente
4. Treinamento de profissionais de saúde
5. Monitoramento contínuo de performance

### 3.10.3 Vieses e Limitações

**Vieses identificados:**
1. **Demográfico**: Dataset de pacientes pediátricos (1-5 anos) pode não generalizar para adultos
2. **Geográfico**: Imagens de uma única instituição (China) podem ter características específicas
3. **Temporal**: Dataset estático não captura variações temporais de equipamento/protocolo

**Mitigação**: Resultados devem ser interpretados considerando essas limitações.

---

## Referências da Metodologia

[1] Kermany, D. S., Goldbaum, M., et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 172(5), 1122-1131.

[2] Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*.

[3] He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.

[4] Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). "Densely Connected Convolutional Networks." *CVPR*.

[5] Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*.

[6] Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks." *WACV*.
