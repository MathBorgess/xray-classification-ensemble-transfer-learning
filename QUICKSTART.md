# Guia Rápido - Chest X-Ray Classification

**Autores:**

- Jéssica A. L. de Macêdo (jalm2@cin.ufpe.br) - Engenharia da Computação, CIn/UFPE
- Matheus Borges Figueirôa (mbf3@cin.ufpe.br) - Ciência da Computação, CIn/UFPE

## 🚀 Quick Start

### 1. Instalação

```bash
# Clone o repositório
git clone https://github.com/MathBorgess/xray-classification-ensemble-transfer-learning.git
cd xray-classification-ensemble-transfer-learning

# Configure o ambiente
chmod +x setup.sh
./setup.sh

# Ative o ambiente virtual
source venv/bin/activate
```

### 2. Teste de Dispositivo (GPU/CPU)

```bash
# Verifique qual acelerador está disponível (CUDA, MPS, ou CPU)
python test_device.py

# O sistema detectará automaticamente:
# - CUDA (NVIDIA GPUs)
# - MPS (Apple Silicon M1/M2/M3)
# - CPU (fallback)
```

```bash
# Prepare a estrutura de diretórios
python prepare_data.py

# Baixe o dataset do Kaggle:
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Extraia para: data/raw/chest_xray/

# Verifique o dataset
python prepare_data.py --check
```

### 4. Treinamento Rápido

#### Treinar um modelo único (EfficientNetB0)

```bash
python train.py --model efficientnet_b0
```

#### Treinar todos os modelos

```bash
python train.py --model efficientnet_b0
python train.py --model resnet50
python train.py --model densenet121
```

#### Criar ensemble

```bash
python ensemble.py
```

#### Testar robustez

```bash
python test_robustness.py \
    --model efficientnet_b0 \
    --model_path models/efficientnet_b0_final.pth
```

## 📊 Análise Exploratória

### Jupyter Notebooks

```bash
# Inicie o Jupyter
jupyter notebook

# Execute os notebooks em ordem:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_model_training.ipynb
# 3. notebooks/03_ensemble_evaluation.ipynb
# 4. notebooks/04_interpretability.ipynb
```

## ⚙️ Configuração Personalizada

### Editar Configurações

Edite `configs/config.yaml` para customizar:

```yaml
# Exemplo: Alterar batch size
data:
  batch_size: 64  # Padrão: 32

# Exemplo: Alterar learning rate
training:
  baseline:
    learning_rate: 0.0005  # Padrão: 0.001

# Exemplo: Adicionar mais épocas
training:
  baseline:
    epochs: 20  # Padrão: 15
```

## 📈 Visualizar Resultados

### TensorBoard (opcional)

```bash
tensorboard --logdir results/logs
```

### Métricas Salvas

- Resultados individuais: `results/*_test_results.txt`
- Comparação ensemble: `results/ensemble_comparison.txt`
- Robustez: `results/*_robustness.txt`
- Figuras: `results/figures/`

## 🔍 Estrutura de Arquivos

```
projeto/
├── src/              # Código fonte modular
├── configs/          # Arquivos de configuração
├── data/            # Datasets
├── models/          # Modelos treinados
├── results/         # Resultados e visualizações
├── notebooks/       # Análises interativas
├── train.py         # Script principal de treinamento
├── ensemble.py      # Script de ensemble
└── test_robustness.py  # Script de teste de robustez
```

## 🐛 Troubleshooting

### Erro: "CUDA out of memory" ou "MPS out of memory"

```yaml
# Reduza o batch size em configs/config.yaml
data:
  batch_size: 16 # ou 8
```

### Verificar qual dispositivo está sendo usado

```bash
# Execute o script de teste
python test_device.py

# Saída esperada:
# ✓ CUDA is available (para NVIDIA GPUs)
# ✓ MPS is available (para Apple Silicon)
# ✓ CPU is always available
```

### macOS com Apple Silicon (M1/M2/M3)

```bash
# O PyTorch detectará automaticamente o MPS
# Saída esperada durante treinamento:
# "Using MPS (Apple Silicon GPU)"

# Se MPS não estiver disponível, atualize o PyTorch:
pip install --upgrade torch torchvision
```

### Erro: "Dataset not found"

```bash
# Verifique a estrutura do dataset
python prepare_data.py --check

# Estrutura esperada:
# data/raw/chest_xray/
#   ├── train/NORMAL/
#   ├── train/PNEUMONIA/
#   ├── val/NORMAL/
#   ├── val/PNEUMONIA/
#   ├── test/NORMAL/
#   └── test/PNEUMONIA/
```

### Erro de importação PyTorch

```bash
# Reinstale os requirements
pip install --upgrade -r requirements.txt
```

## 📝 Comandos Úteis

### Listar modelos disponíveis

```bash
ls -lh models/*.pth
```

### Ver configuração atual

```bash
cat configs/config.yaml
```

### Limpar resultados antigos

```bash
rm -rf results/figures/*
rm -rf results/metrics/*
rm -rf results/logs/*
```

### Backup de modelos

```bash
tar -czf models_backup.tar.gz models/
```

## 🎯 Fluxo de Trabalho Completo

```bash
# 1. Preparar ambiente
./setup.sh
source venv/bin/activate

# 2. Testar dispositivo
python test_device.py

# 3. Preparar dados
python prepare_data.py

# 4. Explorar dados (opcional)
jupyter notebook notebooks/01_data_exploration.ipynb

# 5. Treinar modelos
python train.py --model efficientnet_b0
python train.py --model resnet50
python train.py --model densenet121

# 6. Criar ensemble
python ensemble.py

# 7. Testar robustez
python test_robustness.py --model efficientnet_b0 --model_path models/efficientnet_b0_final.pth

# 8. Analisar resultados
cat results/ensemble_comparison.txt
```

## 💡 Dicas de Performance

### Para treinamento mais rápido:

- **NVIDIA GPU**: Use CUDA (melhor performance)
- **Apple Silicon (M1/M2/M3)**: Use MPS (boa performance)
- **Sem GPU**: Use CPU (mais lento)
- Aumente batch size (se memória permitir)
- Reduza número de épocas para testes

### Configuração por dispositivo:

```yaml
# configs/config.yaml

# Para NVIDIA GPUs:
device:
  use_cuda: true
  gpu_id: 0

# Para Apple Silicon (MPS é detectado automaticamente)
device:
  use_cuda: true  # Tentará CUDA primeiro, depois MPS

# Para CPU apenas:
device:
  use_cuda: false
```

### Para melhor acurácia:

- Use data augmentation agressiva
- Treine por mais épocas
- Ajuste learning rate
- Use ensemble com votação ponderada

### Para interpretabilidade:

- Execute notebook 04 (Grad-CAM)
- Analise as regiões destacadas
- Compare com conhecimento médico

## 📚 Recursos Adicionais

- **Paper de referência**: Ver documento do Google
- **Dataset**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **PyTorch**: https://pytorch.org/docs/
- **timm**: https://github.com/huggingface/pytorch-image-models

## ✉️ Suporte

Para questões ou problemas:

1. Verifique este guia
2. Leia o README.md completo
3. Abra uma issue no GitHub
