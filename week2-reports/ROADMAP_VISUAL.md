# 🎯 Roadmap Visual: Correções Pré-Ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  📊 PROJETO: Chest X-Ray Classification                            │
│  🎓 AUTORES: Jéssica A. L. de Macêdo & Matheus Borges Figueirôa   │
│  🏛️  INSTITUIÇÃO: CIn - UFPE                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 📍 Você Está Aqui

```
┌──────────────────────────────────────────────────────────────────┐
│                        PROGRESSO DO PROJETO                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ✅ FASE -1: Setup e Estrutura          [████████████] 100%    │
│     └─ Código modular, configs, docs                           │
│                                                                  │
│  ✅ FASE 0a: Treinamento Individual     [████████████] 100%    │
│     └─ 3 modelos treinados (Eff, Res, Dense)                   │
│                                                                  │
│  🔴 FASE 0b: Correções Fundamentais      [            ]   0%    │
│     ├─ [ ] Cross-Validation (K=5)                               │
│     ├─ [ ] Threshold Optimization                               │
│     ├─ [ ] Advanced Augmentation                                │
│     ├─ [ ] Focal Loss Implementation                            │
│     └─ [ ] Test-Time Augmentation                               │
│                                                                  │
│  ⏸️  FASE 1: Ensemble Learning           [            ]   0%    │
│     └─ AGUARDANDO: Fase 0b deve estar completa                  │
│                                                                  │
│  ⏸️  FASE 2: Robustness Testing          [            ]   0%    │
│     └─ AGUARDANDO: Fase 1 completa                              │
│                                                                  │
│  ⏸️  FASE 3: Interpretability            [            ]   0%    │
│     └─ AGUARDANDO: Fase 2 completa                              │
│                                                                  │
│  ⏸️  FASE 4: Statistical Validation      [            ]   0%    │
│     └─ AGUARDANDO: Fase 3 completa                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 🚨 Gaps Críticos Identificados

```
┌────────────────────────────────────────────────────────────────────┐
│  GAP                             │ SEVERIDADE │ IMPACTO             │
├──────────────────────────────────┼────────────┼─────────────────────┤
│  Dataset validação (16 samples) │    🔴      │ Métricas instáveis  │
│  Especificidade (12-48%)         │    🔴      │ Inutilizável        │
│  Falta Cross-Validation          │    🟠      │ Sem confiança       │
│  Augmentation limitada           │    🟡      │ Generalização       │
│  Desbalanceamento classe         │    🟠      │ Viés majoritária    │
└────────────────────────────────────────────────────────────────────┘
```

## 🗺️ Plano de Correção (7-10 dias)

```
SEMANA 1: Fundação Estatística
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─ DIA 1-2 ─────────────────────────────────────────────────────┐
│  📊 Cross-Validation (K=5 Stratified)                         │
│  ├─ Implementar: src/cross_validation.py                      │
│  ├─ Treinar: 3 modelos × 5 folds = 15 treinos                 │
│  ├─ Calcular: Média ± Std ± CI(95%)                           │
│  └─ Output: Métricas robustas                                 │
│                                                                │
│  🎯 Target: Validation set ~1000 samples (across folds)       │
│  ⏱️  Tempo: 2 dias                                             │
└────────────────────────────────────────────────────────────────┘

┌─ DIA 3 ───────────────────────────────────────────────────────┐
│  🎚️ Threshold Optimization                                     │
│  ├─ Implementar: src/threshold_optimization.py                │
│  ├─ Métodos: Youden, F1-max, Balanced, Target-Spec            │
│  ├─ Otimizar: Threshold para cada modelo                      │
│  └─ Output: Especificidade ≥ 60%                              │
│                                                                │
│  🎯 Target: Especificidade 60%, Sensibilidade ≥95%            │
│  ⏱️  Tempo: 1 dia                                              │
└────────────────────────────────────────────────────────────────┘


SEMANA 2: Melhorias de Modelo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─ DIA 4 ───────────────────────────────────────────────────────┐
│  🔄 Advanced Augmentation                                      │
│  ├─ Atualizar: src/data_loader.py                             │
│  ├─ Adicionar: Elastic, CLAHE, Noise, Grid, Gamma             │
│  └─ Output: 10+ augmentation types                            │
│                                                                │
│  🔥 Focal Loss                                                 │
│  ├─ Implementar: src/losses.py                                │
│  ├─ Atualizar: src/trainer.py, config.yaml                    │
│  └─ Output: FL(p_t) = -α(1-p_t)^γ log(p_t)                    │
│                                                                │
│  🎯 Target: Melhor balanceamento de classes                   │
│  ⏱️  Tempo: 1 dia                                              │
└────────────────────────────────────────────────────────────────┘

┌─ DIA 5 ───────────────────────────────────────────────────────┐
│  🔄 Re-training com Focal Loss                                 │
│  ├─ Treinar: EfficientNetB0 (melhor baseline)                 │
│  ├─ Validar: Especificidade +5-10%                            │
│  └─ Comparar: vs baseline original                            │
│                                                                │
│  🎯 Target: Especificidade base melhorada                     │
│  ⏱️  Tempo: 1 dia                                              │
└────────────────────────────────────────────────────────────────┘

┌─ DIA 6 ───────────────────────────────────────────────────────┐
│  🎭 Test-Time Augmentation (TTA)                               │
│  ├─ Implementar: src/tta.py                                   │
│  ├─ Testar: Em modelos existentes                             │
│  ├─ Aplicar: 5-10 augmentations por imagem                    │
│  └─ Output: Predições mais estáveis                           │
│                                                                │
│  🎯 Target: Redução de variância                              │
│  ⏱️  Tempo: 1 dia                                              │
└────────────────────────────────────────────────────────────────┘

┌─ DIA 7-10 ────────────────────────────────────────────────────┐
│  ✅ Consolidação e Validação                                   │
│  ├─ Executar: Todos os scripts                                │
│  ├─ Gerar: Relatório consolidado                              │
│  ├─ Validar: Todas as métricas                                │
│  ├─ Comparar: Antes vs Depois                                 │
│  └─ Documentar: Decisões e resultados                         │
│                                                                │
│  🎯 Target: Ready for Ensemble                                │
│  ⏱️  Tempo: 3-4 dias                                           │
└────────────────────────────────────────────────────────────────┘
```

## 📈 Comparação Antes/Depois

```
┌────────────────────────────────────────────────────────────────────┐
│  MÉTRICA              │  ANTES      │  DEPOIS      │  MELHORIA     │
├───────────────────────┼─────────────┼──────────────┼───────────────┤
│  Especificidade       │  12-48%     │  ≥ 60%       │  +20-48% ⬆️   │
│  Validation Size      │  16         │  ~1000       │  +6150% ⬆️    │
│  Balanced Accuracy    │  ~56%       │  ≥ 75%       │  +19% ⬆️      │
│  Confidence Intervals │  ❌         │  ✅ 95% CI   │  NEW ✅       │
│  Augmentation Types   │  4          │  10+         │  +150% ⬆️     │
│  Loss Function        │  CE+weights │  Focal Loss  │  BETTER ✅    │
│  TTA                  │  No         │  Yes (5-10x) │  NEW ✅       │
└────────────────────────────────────────────────────────────────────┘
```

## 📂 Arquivos Criados

```
xray-classification-ensemble-transfer-learning/
│
├── 📄 PRE_ENSEMBLE_FIXES.md         ⭐ COMECE AQUI
│   └─ Plano detalhado com código completo para cada correção
│
├── 📄 FIXES_SUMMARY.md              📊 Sumário visual
│   └─ Visão geral rápida dos gaps e soluções
│
├── 📄 ROADMAP_VISUAL.md             🗺️ Este arquivo
│   └─ Roadmap visual do projeto
│
├── 📄 progress.md                    📈 Status atualizado
│   └─ Roadmap completo com Fase 0b adicionada
│
├── 📄 IMPLEMENTATION_GUIDE.md        ⏸️ Para DEPOIS
│   └─ Implementação do ensemble (usar após correções)
│
└── scripts/
    ├── quickstart_fixes.py           🚀 Script de diagnóstico
    │   └─ Verifica status e mostra próximos passos
    │
    └── README.md                      📖 Guia de scripts
        └─ Como usar os scripts de correção
```

## 🎬 Como Começar AGORA

```bash
# 1️⃣ Execute o diagnóstico
python3 scripts/quickstart_fixes.py

# 2️⃣ Abra a documentação
open PRE_ENSEMBLE_FIXES.md         # ← COMECE AQUI

# 3️⃣ Revise o plano
open FIXES_SUMMARY.md              # Sumário visual
open progress.md                    # Status atualizado

# 4️⃣ Implemente correções (em ordem!)
# Dia 1-2: Cross-Validation
vim src/cross_validation.py        # Copie código do PRE_ENSEMBLE_FIXES.md
python -m src.cross_validation      # Execute

# Dia 3: Threshold Optimization
vim src/threshold_optimization.py  # Copie código
python -m src.threshold_optimization

# Dia 4: Advanced Augmentation + Focal Loss
vim src/data_loader.py             # Atualizar
vim src/losses.py                  # Criar
vim configs/config.yaml            # Atualizar

# Dia 5: Re-training
python train.py --model efficientnet_b0 --use-focal-loss

# Dia 6: TTA
vim src/tta.py                     # Criar
python -m src.tta                  # Testar

# Dia 7-10: Consolidação
# Executar tudo, validar, documentar
```

## ✅ Checklist de Validação

```
ANTES DE PROSSEGUIR PARA ENSEMBLE, VERIFIQUE:

┌─ Implementação ────────────────────────────────────────────────┐
│  [ ] src/cross_validation.py criado                            │
│  [ ] src/threshold_optimization.py criado                      │
│  [ ] src/data_loader.py atualizado (advanced aug)              │
│  [ ] src/losses.py criado (Focal Loss)                         │
│  [ ] src/tta.py criado                                         │
│  [ ] configs/config.yaml atualizado                            │
│  [ ] src/trainer.py atualizado (suporta Focal Loss)            │
└─────────────────────────────────────────────────────────────────┘

┌─ Execução ─────────────────────────────────────────────────────┐
│  [ ] Cross-validation executado (3 modelos, 5 folds cada)      │
│  [ ] Thresholds otimizados calculados                          │
│  [ ] Modelo re-treinado com Focal Loss                         │
│  [ ] TTA testado e validado                                    │
└─────────────────────────────────────────────────────────────────┘

┌─ Resultados ───────────────────────────────────────────────────┐
│  [ ] Especificidade ≥ 60% alcançada                            │
│  [ ] Balanced accuracy ≥ 75%                                   │
│  [ ] Intervalos de confiança calculados (95% CI)               │
│  [ ] Métricas antes/depois documentadas                        │
│  [ ] Gráficos de análise gerados                               │
└─────────────────────────────────────────────────────────────────┘

┌─ Documentação ─────────────────────────────────────────────────┐
│  [ ] Resultados salvos em results/                             │
│  [ ] Figuras salvas em results/figures/                        │
│  [ ] Decisões documentadas no código                           │
│  [ ] progress.md atualizado                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Métricas de Sucesso

```
┌────────────────────────────────────────────────────────────────┐
│                  CRITÉRIOS DE SUCESSO                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✅ OBRIGATÓRIO (Mínimo Aceitável):                            │
│     ├─ Especificidade ≥ 60%                                   │
│     ├─ Sensibilidade ≥ 95%                                    │
│     ├─ Balanced Accuracy ≥ 75%                                │
│     ├─ CI(95%) width < 5% para accuracy                       │
│     └─ Cross-validation completo (5 folds)                    │
│                                                                │
│  ⭐ DESEJÁVEL (Ideal):                                         │
│     ├─ Especificidade ≥ 70%                                   │
│     ├─ Balanced Accuracy ≥ 80%                                │
│     ├─ TTA reduz variância em ≥20%                            │
│     └─ Focal Loss melhora baseline em ≥5%                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## ⚠️ Avisos Críticos

```
╔════════════════════════════════════════════════════════════════╗
║                         ⚠️  ATENÇÃO                            ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  🔴 NÃO PULE PARA ENSEMBLE SEM COMPLETAR ESTAS CORREÇÕES      ║
║                                                                ║
║     Motivos:                                                  ║
║     • Base estatística instável (16 amostras validação)       ║
║     • Métricas não confiáveis (especificidade 12-48%)         ║
║     • Sem intervalos de confiança                             ║
║     • Ensemble será construído sobre areia movediça           ║
║     • Artigo será REJEITADO por falta de rigor                ║
║                                                                ║
║  ✅ SIGA A ORDEM PROPOSTA                                      ║
║                                                                ║
║     1. Cross-Validation (base estatística)                    ║
║     2. Threshold Optimization (especificidade)                ║
║     3. Advanced Augmentation (generalização)                  ║
║     4. Focal Loss (balanceamento)                             ║
║     5. Test-Time Augmentation (robustez)                      ║
║     6. Consolidação (validação)                               ║
║     7. ENTÃO → Ensemble                                       ║
║                                                                ║
║  ⏱️  TEMPO NECESSÁRIO: 7-10 DIAS                               ║
║                                                                ║
║     Este tempo é REAL e NECESSÁRIO.                           ║
║     Melhor gastar agora que refazer depois.                   ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

## 🎓 Justificativa Acadêmica

```
Por que estas correções são CRÍTICAS para publicação?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Dataset de Validação (16 amostras)
   └─ Reviewers rejeitariam imediatamente
   └─ Estatisticamente insignificante
   └─ Não permite early stopping confiável
   └─ Solução: Cross-validation com ~1000 samples

2. Especificidade 12-48%
   └─ 80-90% de falsos positivos
   └─ Inutilizável na prática clínica
   └─ Radiologistas perderiam confiança no sistema
   └─ Solução: Threshold optimization → ≥60%

3. Falta de Intervalos de Confiança
   └─ Impossível avaliar significância
   └─ Não atende padrões científicos
   └─ Comparações não têm validade estatística
   └─ Solução: Bootstrap CI via cross-validation

4. Desbalanceamento Não Resolvido
   └─ Class weights não são suficientes
   └─ Modelos enviesados para classe majoritária
   └─ Focal Loss é estado da arte
   └─ Solução: Implementar Focal Loss

5. Robustez Não Validada
   └─ Predições podem ter alta variância
   └─ Sem TTA, resultados menos confiáveis
   └─ Ensemble sem TTA perde potencial
   └─ Solução: Test-Time Augmentation
```

## 📞 Próximas Ações

```
┌────────────────────────────────────────────────────────────────┐
│  AÇÃO IMEDIATA:                                                │
│  ───────────────                                               │
│                                                                │
│  1. python3 scripts/quickstart_fixes.py                        │
│     └─ Verificar status do sistema                            │
│                                                                │
│  2. open PRE_ENSEMBLE_FIXES.md                                 │
│     └─ Ler plano detalhado                                    │
│                                                                │
│  3. Implementar Dia 1-2: Cross-Validation                      │
│     ├─ Copiar código de PRE_ENSEMBLE_FIXES.md                 │
│     ├─ Criar src/cross_validation.py                          │
│     └─ Executar: python -m src.cross_validation                │
│                                                                │
│  4. Seguir cronograma de 7-10 dias                             │
│                                                                │
│  5. Validar resultados a cada etapa                            │
│                                                                │
│  6. APENAS DEPOIS → Implementar ensemble                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

```
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  🎯 OBJETIVO: Base Estatisticamente Sólida                     ║
║                                                                ║
║  ⏱️  PRAZO: 7-10 dias                                          ║
║                                                                ║
║  🎓 RESULTADO: Artigo Publicável com Rigor Científico          ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

**Autores:** Jéssica A. L. de Macêdo & Matheus Borges Figueirôa  
**Instituição:** CIn - UFPE  
**Data:** 12 de Novembro de 2025  

**Status:** 🔴 IMPLEMENTAÇÃO PENDENTE  
**Prioridade:** 🔴 CRÍTICA  
**Próxima Ação:** `python3 scripts/quickstart_fixes.py`
