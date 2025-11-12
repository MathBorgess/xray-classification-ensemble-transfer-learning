# 🔧 Scripts de Correção Pré-Ensemble

Este diretório contém scripts para implementar as correções críticas identificadas antes da implementação do ensemble.

## 📋 Visão Geral

Antes de implementar o ensemble learning, é necessário corrigir gaps fundamentais no projeto:

1. **Dataset de validação muito pequeno** (16 amostras)
2. **Especificidade extremamente baixa** (12-48%)
3. **Falta de cross-validation**
4. **Augmentation limitada**
5. **Desbalanceamento de classes não resolvido**

## 🚀 Quick Start

### 1. Execute o script de diagnóstico:

```bash
python scripts/quickstart_fixes.py
```

Este script irá:
- ✅ Verificar dependências
- ✅ Validar estrutura de dados
- ✅ Checar modelos treinados
- ✅ Mostrar plano de implementação
- ✅ Listar próximos passos

### 2. Revise a documentação:

```bash
# Abrir documentos em ordem:
open PRE_ENSEMBLE_FIXES.md     # ← COMECE AQUI: Plano detalhado
open progress.md               # Status do projeto
open IMPLEMENTATION_GUIDE.md   # Implementação do ensemble (DEPOIS)
```

## 📂 Estrutura de Scripts

```
scripts/
├── quickstart_fixes.py          # Script de diagnóstico e guia
└── README.md                     # Este arquivo
```

## 🛠️ Scripts a Serem Criados

Seguindo o plano em `PRE_ENSEMBLE_FIXES.md`, você deverá criar:

### Fase 0: Correções Fundamentais

#### 1. Cross-Validation (Dias 1-2)
```
src/cross_validation.py
```
**Função:** Implementar K-Fold stratified cross-validation  
**Output:** Métricas com média ± std ± CI(95%)  
**Comando:** `python -m src.cross_validation`

#### 2. Threshold Optimization (Dia 3)
```
src/threshold_optimization.py
```
**Função:** Otimizar threshold de decisão  
**Output:** Especificidade ≥ 60%  
**Comando:** `python -m src.threshold_optimization`

#### 3. Advanced Augmentation (Dia 4)
```
src/data_loader.py (atualizar)
```
**Função:** Adicionar augmentations específicas para raio-X  
**Output:** 10+ tipos de augmentation

#### 4. Focal Loss (Dia 4)
```
src/losses.py
```
**Função:** Implementar Focal Loss e Class-Balanced Loss  
**Output:** Melhor balanceamento de classes

#### 5. Test-Time Augmentation (Dia 5)
```
src/tta.py
```
**Função:** TTA para reduzir variância  
**Output:** Predições mais estáveis

## 📊 Métricas Esperadas

### Antes das Correções:
| Métrica | Valor |
|---------|-------|
| Especificidade | 12-48% |
| Validation Size | 16 amostras |
| Confidence Intervals | ❌ Não disponível |
| Balanced Accuracy | ~56% |

### Após as Correções:
| Métrica | Valor Target |
|---------|--------------|
| Especificidade | ≥ 60% |
| Validation Size | ~1000 amostras (5 folds) |
| Confidence Intervals | ✅ 95% CI para todas métricas |
| Balanced Accuracy | ≥ 75% |

## 🎯 Ordem de Implementação

**⚠️ IMPORTANTE:** Siga esta ordem rigorosamente!

1. ✅ **Cross-Validation** (Base estatística)
2. ✅ **Threshold Optimization** (Especificidade)
3. ✅ **Advanced Augmentation** (Generalização)
4. ✅ **Focal Loss** (Balanceamento)
5. ✅ **Test-Time Augmentation** (Robustez)
6. ✅ **Consolidação** (Validação)
7. ➡️ **Ensemble** (Após todas as correções)

## 📖 Documentação Relacionada

| Documento | Propósito |
|-----------|-----------|
| `PRE_ENSEMBLE_FIXES.md` | **🔴 PRIORIDADE:** Plano detalhado de correções |
| `progress.md` | Status geral do projeto e roadmap |
| `IMPLEMENTATION_GUIDE.md` | Guia de implementação do ensemble (após fixes) |
| `QUICKSTART.md` | Guia rápido do projeto |
| `README.md` | Documentação geral |

## 🔍 Checklist de Progresso

Marque conforme completa cada etapa:

### Pré-requisitos
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Dataset baixado e estruturado
- [ ] Modelos base treinados (efficientnet_b0, resnet50, densenet121)

### Fase 0: Correções Fundamentais
- [ ] `src/cross_validation.py` implementado
- [ ] Cross-validation executado para 3 modelos
- [ ] `src/threshold_optimization.py` implementado
- [ ] Thresholds otimizados calculados
- [ ] `src/data_loader.py` atualizado com advanced augmentation
- [ ] `src/losses.py` implementado (Focal Loss)
- [ ] `src/tta.py` implementado
- [ ] Modelo re-treinado com Focal Loss
- [ ] Todas as correções validadas

### Fase 1: Ensemble (Após Fase 0)
- [ ] Verificar que Fase 0 está 100% completa
- [ ] Prosseguir com `IMPLEMENTATION_GUIDE.md`

## 💡 Dicas

### Para Cross-Validation:
```bash
# Rodar apenas EfficientNetB0 primeiro (mais rápido)
# Depois adicionar outros modelos
python -m src.cross_validation
```

### Para Threshold Optimization:
```bash
# Testar diferentes métodos:
# - youden: Maximiza J = Sensitivity + Specificity - 1
# - f1: Maximiza F1-score
# - balanced: Balanceia sensibilidade e especificidade
python -m src.threshold_optimization
```

### Para Focal Loss:
```python
# Em config.yaml, ativar:
training:
  use_focal_loss: true
  use_class_weights: false
```

## ⚠️ Avisos Importantes

1. **Não pule etapas:** Cada correção depende da anterior
2. **Valide resultados:** Sempre compare antes/depois
3. **Documente decisões:** Adicione comentários no código
4. **Salve checkpoints:** Backup de modelos treinados
5. **Monitore métricas:** Especificidade deve melhorar

## 🆘 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Data directory not found"
```bash
# Verificar estrutura:
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
└── test/
```

### "Model file not found"
```bash
# Treinar modelos:
python train.py --model efficientnet_b0
python train.py --model resnet50
python train.py --model densenet121
```

## 📞 Suporte

Para dúvidas ou problemas:
1. Revise `PRE_ENSEMBLE_FIXES.md` para detalhes de implementação
2. Consulte `progress.md` para contexto do projeto
3. Verifique `QUICKSTART.md` para setup geral

---

**Autores:** Jéssica A. L. de Macêdo & Matheus Borges Figueirôa  
**Instituição:** CIn - UFPE  
**Última atualização:** 12/11/2025
