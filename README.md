# Chest X-Ray Classification with Transfer Learning and Ensemble Learning

**Link do documento**: https://docs.google.com/document/d/1X3J7e3w0Jrk--bUHL-_0c1Ek9cBb5if6q3s-14et4PA/edit?usp=sharing

Este projeto implementa uma abordagem híbrida que combina Transfer Learning com técnicas de Ensemble Learning para classificação automatizada de imagens de raio-X torácico, distinguindo entre casos normais e patológicos (pneumonia).

## 📋 Resumo

Este estudo propõe uma metodologia que explora o fine-tuning de múltiplas arquiteturas pré-treinadas (EfficientNet, ResNet e DenseNet) e sua integração através de esquemas de votação ponderada. O projeto visa desenvolver um sistema robusto de classificação capaz de mitigar as limitações de dados médicos anotados através de Transfer Learning e aumentar a robustez através de Ensemble Learning.

## 🎯 Objetivo

Desenvolver e validar uma abordagem híbrida que integra Transfer Learning e Ensemble Learning para classificação automatizada de imagens de raio-X torácico, comparando seu desempenho com métodos individuais.

## 📚 Palavras-chave

Transfer Learning, Ensemble Learning, Imagens Médicas, Raio-X Torácico, Deep Learning, Classificação de Imagens

## 🏗️ Estrutura do Projeto

```
xray-classification-ensemble-transfer-learning/
├── configs/
│   └── config.yaml              # Configurações do projeto
├── data/
│   ├── raw/                     # Dados brutos
│   └── processed/               # Dados processados
├── models/
│   └── checkpoints/             # Checkpoints dos modelos
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_ensemble_evaluation.ipynb
│   └── 04_interpretability.ipynb
├── results/
│   ├── figures/                 # Visualizações
│   ├── metrics/                 # Métricas salvas
│   └── logs/                    # Logs de treinamento
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Carregamento e preprocessamento de dados
│   ├── models.py               # Arquiteturas dos modelos
│   ├── trainer.py              # Funções de treinamento
│   ├── evaluation.py           # Métricas e avaliação
│   ├── interpretability.py     # Grad-CAM e visualizações
│   └── utils.py                # Funções utilitárias
├── train.py                    # Script de treinamento
├── ensemble.py                 # Script de ensemble
├── test_robustness.py          # Script de teste de robustez
├── requirements.txt            # Dependências
├── setup.sh                    # Script de configuração
└── README.md                   # Este arquivo
```

## 🔧 Instalação

### Pré-requisitos

- Python 3.8+
- GPU (opcional, mas recomendado):
  - NVIDIA GPU com CUDA (Linux/Windows)
  - Apple Silicon com MPS (macOS)
  - Caso contrário, CPU será utilizada

### Configuração do Ambiente

1. Clone o repositório:

```bash
git clone https://github.com/MathBorgess/xray-classification-ensemble-transfer-learning.git
cd xray-classification-ensemble-transfer-learning
```

2. Execute o script de configuração:

```bash
chmod +x setup.sh
./setup.sh
```

Ou configure manualmente:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Teste o dispositivo disponível:

```bash
python test_device.py
```

O sistema detectará automaticamente:

- **CUDA** (NVIDIA GPUs) - Melhor performance
- **MPS** (Apple Silicon M1/M2/M3) - Boa performance no macOS
- **CPU** - Disponível sempre (mais lento)

## 📊 Dataset

Este projeto utiliza o **Chest X-Ray Dataset** disponível no Kaggle.

### Download do Dataset

1. Acesse: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Baixe o dataset
3. Extraia para: `data/raw/chest_xray/`

### Estrutura Esperada

```
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## 🚀 Uso

### 1. Treinamento de Modelos Individuais

Treine cada arquitetura individualmente:

```bash
# EfficientNetB0
python train.py --model efficientnet_b0

# ResNet50
python train.py --model resnet50

# DenseNet121
python train.py --model densenet121
```

Opções adicionais:

```bash
python train.py --model efficientnet_b0 \
                --config configs/config.yaml \
                --data_dir data/raw/chest_xray \
                --output_dir models
```

### 2. Criação e Avaliação do Ensemble

Após treinar os modelos individuais:

```bash
python ensemble.py --model_dir models \
                   --output_dir results
```

### 3. Teste de Robustez

Teste a robustez dos modelos sob perturbações:

```bash
python test_robustness.py --model efficientnet_b0 \
                          --model_path models/efficientnet_b0_final.pth \
                          --output_dir results
```

## 📓 Notebooks

Execute os notebooks na ordem para análise completa:

1. **01_data_exploration.ipynb**: Exploração e análise dos dados
2. **02_model_training.ipynb**: Treinamento interativo dos modelos
3. **03_ensemble_evaluation.ipynb**: Avaliação do ensemble
4. **04_interpretability.ipynb**: Visualizações Grad-CAM

## 🔬 Metodologia

### 1. Preparação de Dados

- **Dataset**: Chest X-ray Dataset com imagens categorizadas em Normal e Pneumonia
- **Divisão**: 70% treino, 15% validação, 15% teste (estratificada)
- **Pré-processamento**:
  - Redimensionamento para 224×224 pixels
  - Normalização com estatísticas do ImageNet
  - Data augmentation: rotação ±10°, espelhamento horizontal, ajuste de brilho ±10%, zoom 10%

### 2. Transfer Learning

**Arquiteturas**:

- EfficientNetB0 (~5.3M parâmetros)
- ResNet50 (~25.6M parâmetros)
- DenseNet121 (~8M parâmetros)

**Estratégias de Fine-tuning**:

- **Baseline**: Congelamento total exceto classificador (lr=0.001, 15 épocas)
- **Progressive Unfreezing**:
  - Stage 1: Descongelamento das últimas 20 camadas (lr=0.0001, 15 épocas)
  - Stage 2: Descongelamento das últimas 50 camadas (lr=0.00001, 15 épocas)

**Configuração**:

- Otimizador: Adam
- Loss: Binary Cross-Entropy
- Batch size: 32
- Early Stopping: patience=5
- Class weights: aplicados se desbalanceamento > 2:1

### 3. Ensemble Learning

**Abordagens de Combinação**:

1. **Votação Simples**: Média aritmética das predições
2. **Votação Ponderada**: Pesos proporcionais à AUC de validação

### 4. Avaliação

**Métricas**:

- Acurácia
- Sensibilidade (Recall)
- Especificidade
- AUC-ROC
- F1-Score
- Precisão

**Comparações**:

- Baseline vs fine-tuned
- Modelos individuais vs ensemble
- Votação simples vs ponderada

**Teste de Robustez**:

- Ruído gaussiano (σ=10,20)
- Redução de contraste (50%,70%)
- Rotações (±5°,±10°)

**Análises Estatísticas**:

- Teste t-pareado (p<0.05)
- Grad-CAM para interpretabilidade

## 📈 Resultados Esperados

O projeto visa demonstrar:

1. **Superioridade do Ensemble**: Modelos ensemble superam modelos individuais
2. **Eficácia do Transfer Learning**: Fine-tuning progressivo melhora o desempenho
3. **Robustez**: Ensemble mantém desempenho sob perturbações
4. **Interpretabilidade**: Grad-CAM revela regiões relevantes para diagnóstico

## 🛠️ Configuração

Todas as configurações podem ser ajustadas em `configs/config.yaml`:

- Parâmetros de dados (augmentation, batch size, splits)
- Arquiteturas de modelos
- Hiperparâmetros de treinamento
- Métodos de ensemble
- Métricas de avaliação
- Perturbações para teste de robustez

### Configuração de Dispositivo

O sistema detecta automaticamente o melhor dispositivo disponível:

```yaml
# configs/config.yaml
device:
  use_cuda: true # Tentará CUDA (NVIDIA), depois MPS (Apple), depois CPU
  gpu_id: 0 # ID da GPU para CUDA (ignorado para MPS)
```

**Ordem de detecção:**

1. CUDA (NVIDIA GPUs) - Se disponível
2. MPS (Apple Silicon) - Se disponível em macOS
3. CPU - Sempre disponível

Para verificar qual dispositivo está disponível:

```bash
python test_device.py
```

## 👥 Autores

**Jéssica A. L. de Macêdo**  
Engenharia da Computação, CIn - UFPE  
📧 jalm2@cin.ufpe.br

**Matheus Borges Figueirôa**  
Ciência da Computação, CIn - UFPE  
📧 mbf3@cin.ufpe.br

## �📝 Citação

Se você usar este projeto em sua pesquisa, por favor cite:

```bibtex
@misc{xray_classification_ensemble,
  title={Chest X-Ray Classification with Transfer Learning and Ensemble Learning},
  author={de Macêdo, Jéssica A. L. and Figueirôa, Matheus Borges},
  year={2025},
  publisher={GitHub},
  institution={Centro de Informática, Universidade Federal de Pernambuco},
  url={https://github.com/MathBorgess/xray-classification-ensemble-transfer-learning}
}
```

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📧 Contato

Para questões ou sugestões, abra uma issue no GitHub.

## 🙏 Agradecimentos

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- PyTorch e timm para as arquiteturas pré-treinadas
- Comunidade de Deep Learning e Computer Vision

## 📚 Referências

1. ImageNet Large Scale Visual Recognition Challenge
2. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
3. Deep Residual Learning for Image Recognition
4. Densely Connected Convolutional Networks
5. Grad-CAM: Visual Explanations from Deep Networks
6. Transfer Learning for Medical Image Analysis

---

**Nota**: Este projeto foi desenvolvido para fins acadêmicos e de pesquisa. Não deve ser usado como substituto para diagnóstico médico profissional.
