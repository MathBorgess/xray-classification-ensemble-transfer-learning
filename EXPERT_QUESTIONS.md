# Questionamentos e Pesquisas para Especialistas em IA

## Perguntas Abertas para Melhorar o Artigo de Pneumonia com Transfer Learning e Ensemble

---

## 📋 Contexto do Projeto

**Título**: Transfer Learning e Ensemble Learning para Classificação de Pneumonia em Raios-X Torácicos

**Resultados Atuais**:

- EfficientNet-B0: 80.29% acc, 99.74% sens, 47.86% spec
- Ensemble (Simple/Weighted Voting): 71.47% acc - NÃO superou individual
- Dataset: 5,863 imagens (desbalanceamento 1:2.89)
- Problema crítico: 52% de falsos positivos

**Objetivo deste documento**: Coletar insights de especialistas em IA/ML para tornar o artigo mais completo, rigoroso e interessante para publicação em venues de alto impacto (MICCAI, TMI, MedIA).

---

## 🔬 SEÇÃO 1: METODOLOGIA E DESIGN EXPERIMENTAL

### 1.1 Arquitetura e Transfer Learning

**Q1.1**: Por que EfficientNet-B0 superou ResNet-50 e DenseNet-121 em 13-14%?

- Nossa hipótese: Compound scaling é mais eficiente que residual/dense connections
- **Pergunta para especialistas**:
  - Há evidências na literatura de que compound scaling é superior para imagens médicas?
  - Seria mais apropriado comparar com modelos de capacidade similar (params)?
  - ResNet-50 (25.6M) vs EfficientNet-B0 (5.3M) - comparação justa?

**Q1.2**: Progressive Unfreezing - estratégia ideal?

- Usamos: 5 épocas classifier-only + 20 épocas full fine-tuning
- **Pergunta para especialistas**:
  - Essa proporção (1:4) é padrão? Literatura sugere outras?
  - Deveríamos usar gradual unfreezing (camada por camada) em vez de two-stage?
  - Layer-wise Learning Rate Decay (LLRD) seria melhor? Ex: LR × 0.95^(max_layer - current_layer)

**Q1.3**: ImageNet Pre-training vs Medical Pre-training

- Todos os modelos foram pré-treinados em ImageNet (fotos naturais)
- **Pergunta para especialistas**:
  - Vale a pena pré-treinar em ChestX-ray14 ou MIMIC-CXR antes de fine-tunar?
  - Self-supervised (SimCLR, MoCo, MAE) em raios-X seria mais eficaz?
  - Há estudos comparando ImageNet vs Medical pre-training para pneumonia?

**Q1.4**: Arquiteturas modernas não testadas

- Não testamos: Vision Transformers (ViT, Swin, BEiT), ConvNeXt, MaxViT
- **Pergunta para especialistas**:
  - ViT requer datasets maiores (literatura sugere >10K). Com 5.2K samples, overfittaria?
  - Swin Transformer seria mais apropriado (hierarchical, menor receptive field inicial)?
  - Hybrid CNN-Transformer (ConvNeXt, CoAtNet) seria o melhor dos dois mundos?

---

### 1.2 Ensemble Learning

**Q2.1**: Por que Simple Voting = Weighted Voting (resultados idênticos)?

- Pesos calculados por AUC: 34.26%, 33.36%, 32.38% (diferença de apenas 2%)
- **Pergunta para especialistas**:
  - Isso indica que os modelos têm performance similar demais para weighted voting funcionar?
  - Deveríamos usar métricas mais discriminativas para pesos? (Ex: Specificity, Balanced Acc)
  - Pesos adaptativos por imagem (confidence-based) seriam melhores?

**Q2.2**: Por que ensemble não superou o melhor individual (-8.82%)?

- Nossa análise: dominância de modelos fracos + correlação de erros + pesos ineficazes
- **Pergunta para especialistas**:
  - Essa falha de ensemble é comum em imagens médicas? Há casos similares na literatura?
  - Diversity metrics (Q-statistic, correlation coefficient, disagreement) deveriam ser calculados antes?
  - Negative Correlation Learning (NCL) durante treinamento forçaria diversidade?

**Q2.3**: Stacking vs Voting - qual implementar?

- Planejamos implementar stacking com meta-learner (Logistic Regression, XGBoost, LightGBM)
- **Pergunta para especialistas**:
  - Qual meta-learner é mais robusto para datasets médicos pequenos?
  - Devemos treinar meta-learner em validation set ou usar cross-validation nested?
  - Feature engineering no meta-learner (ex: concatenar [predictions, confidence, disagreement]) ajuda?

**Q2.4**: Ensemble de quê? CNN + Transformer?

- Atualmente: 3 CNNs (EfficientNet, ResNet, DenseNet) - arquiteturas similares
- **Pergunta para especialistas**:
  - Ensemble de CNN + Vision Transformer teria mais diversidade?
  - Ensemble de modelos com diferentes resoluções (224×224, 384×384, 512×512)?
  - Ensemble de modelos treinados com diferentes augmentations?

---

### 1.3 Loss Functions e Balanceamento

**Q3.1**: Focal Loss - hiperparâmetros ideais

- Planejamos: α=[1.945, 0.673], γ=2.0
- **Pergunta para especialistas**:
  - γ=2.0 é o padrão de Lin et al. (2017), mas há estudos tuning γ para imagens médicas?
  - α deveria ser igual aos class weights ou ajustado separadamente?
  - Focal Loss funciona melhor com qual optimizer? (Adam vs AdamW vs SGD)

**Q3.2**: Class-Balanced Loss vs Focal Loss

- Temos duas opções para lidar com desbalanceamento
- **Pergunta para especialistas**:
  - Class-Balanced Loss (CB Loss) é superior a Focal Loss em datasets médicos?
  - Combinar ambos (CB-Focal Loss) como em Cui et al. (2019) faz sentido?
  - LDAM Loss (Label-Distribution-Aware Margin) seria melhor?

**Q3.3**: Outros approaches para desbalanceamento

- Não testamos: SMOTE, Undersampling, Cost-sensitive learning explícito
- **Pergunta para especialistas**:
  - SMOTE (oversampling sintético) funciona bem em imagens médicas?
  - Undersampling da classe majoritária (jogar fora dados) é recomendado?
  - Two-stage training (balance primeiro, imbalance depois) ajudaria?

---

### 1.4 Data Augmentation

**Q4.1**: Augmentation específico para raio-X

- Planejamos: CLAHE, Elastic Deformation, Grid Distortion, etc.
- **Pergunta para especialistas**:
  - Há augmentations específicos para raio-X que não estamos considerando?
  - RandAugment ou AutoAugment (busca automática) funcionam em imagens médicas?
  - Augmentation baseado em física (simulação de diferentes energias de raio-X)?

**Q4.2**: Mixup/CutMix em imagens médicas

- Mixup interpola imagens; CutMix cola regiões
- **Pergunta para especialistas**:
  - Mixup é válido clinicamente? (misturar pneumonia + normal → diagnóstico ambíguo)
  - CutMix preserva melhor anatomia que Mixup?
  - Mosaic augmentation (grid de 4 imagens) usado em YOLO seria útil?

**Q4.3**: Test-Time Augmentation - quantas augmentations?

- Planejamos: 5 augmentations (horizontal flip, rotate ±5°, shift, brightness)
- **Pergunta para especialistas**:
  - 5 é suficiente ou 10-20 seria melhor? (trade-off tempo vs performance)
  - Quais augmentations são mais eficazes em TTA para raio-X?
  - TTA com voting (modal class) vs averaging (probabilidades) - qual melhor?

---

## 🎯 SEÇÃO 2: AVALIAÇÃO E MÉTRICAS

### 2.1 Métricas de Performance

**Q5.1**: Balanced Accuracy vs F1-Score - qual priorizar?

- Balanced Acc = (Sens + Spec) / 2
- F1 = 2 × (Prec × Rec) / (Prec + Rec)
- **Pergunta para especialistas**:
  - Para aplicação clínica de screening, qual métrica é mais informativa?
  - Há métricas específicas para imbalanced medical imaging (ex: G-mean)?
  - Matthews Correlation Coefficient (MCC) seria melhor que F1 para imbalance?

**Q5.2**: Threshold Optimization - qual método?

- Planejamos: Youden's J, F1-max, Target-Specificity, Cost-Sensitive
- **Pergunta para especialistas**:
  - Para pneumonia, qual método é clinicamente mais justificável?
  - Como definir custos em cost-sensitive? Custo(FN) = 10 × Custo(FP) é razoável?
  - Threshold deve ser otimizado por fold (cross-validation) ou global?

**Q5.3**: Calibração de probabilidades

- Não analisamos calibração (reliability diagrams, ECE, Brier score)
- **Pergunta para especialistas**:
  - Calibração é crítica para aplicação clínica? (médicos confiam em "90% de certeza"?)
  - Temperature Scaling, Platt Scaling ou Isotonic Regression - qual melhor para calibrar?
  - Focal Loss naturalmente descalibra modelos? (foco em hard examples)

---

### 2.2 Validação Estatística

**Q6.1**: Testes estatísticos - estamos usando os corretos?

- Usamos: McNemar's test (accuracy), Bootstrap CI (intervalos de confiança)
- **Pergunta para especialistas**:
  - DeLong's test para comparar AUCs seria mais apropriado?
  - Paired t-test para múltiplas métricas (F1, Balanced Acc)?
  - Bonferroni correction para múltiplas comparações (5 modelos)?

**Q6.2**: Cross-validation - nested ou não?

- Planejamos: K=5 stratified CV (outer loop)
- **Pergunta para especialistas**:
  - Nested CV (inner loop para tuning de hiperparâmetros) é necessário?
  - GroupKFold seria melhor? (agrupar imagens do mesmo paciente)
  - Repeated CV (5×2 ou 10×10) aumenta confiabilidade? Trade-off com custo computacional?

**Q6.3**: Significância clínica vs estatística

- Diferença estatisticamente significativa (p<0.001) nem sempre é clinicamente relevante
- **Pergunta para especialistas**:
  - Qual diferença mínima de Specificity é clinicamente importante? (5%, 10%, 15%?)
  - Non-inferiority test seria apropriado? (ensemble não é inferior a individual)
  - Como reportar efeito prático além de p-value? (Cohen's d, odds ratio)

---

### 2.3 Interpretabilidade e Explicabilidade

**Q7.1**: Grad-CAM - suficiente para validação clínica?

- Planejamos: Grad-CAM++ para visualizar regiões importantes
- **Pergunta para especialistas**:
  - Grad-CAM é o estado da arte ou há métodos melhores? (LayerCAM, ScoreCAM, XGrad-CAM)
  - Attention maps de Transformers são mais interpretáveis que Grad-CAM de CNNs?
  - Como validar quantitativamente? (Intersection over Union com máscaras de radiologistas?)

**Q7.2**: Attention mechanisms - devem ser adicionados?

- Não usamos: Self-Attention, Channel Attention (SE, CBAM), Spatial Attention
- **Pergunta para especialistas**:
  - Attention modules melhoram interpretabilidade e performance simultaneamente?
  - Squeeze-and-Excitation (SE) já está no EfficientNet - suficiente?
  - Attention Branch Network (ABN) que força atenção durante treino seria útil?

**Q7.3**: Saliency maps e perturbation-based methods

- Não testamos: RISE, LIME, SHAP, Integrated Gradients
- **Pergunta para especialistas**:
  - Esses métodos são complementares a Grad-CAM ou redundantes?
  - SHAP (SHapley Additive exPlanations) é mais rigoroso matematicamente?
  - Perturbation tests (ocluir regiões, medir drop de performance) são necessários?

---

## 🏥 SEÇÃO 3: APLICAÇÃO CLÍNICA E VALIDAÇÃO

### 3.1 Trade-off Sensitivity-Specificity

**Q8.1**: Qual threshold para aplicação real?

- Atualmente: Sens=99.74%, Spec=47.86%
- Meta: Sens≥95%, Spec≥65%
- **Pergunta para especialistas**:
  - Em screening de pneumonia pediátrica, qual trade-off é aceitável clinicamente?
  - Há guidelines de sociedades médicas (ATS, ERS, RSNA) sobre thresholds mínimos?
  - Different thresholds para diferentes contextos? (emergency vs routine screening)

**Q8.2**: Impacto de falsos positivos vs falsos negativos

- FP: Paciente normal recebe tratamento desnecessário (antibióticos, exames adicionais)
- FN: Paciente com pneumonia não é tratado (risco de morte)
- **Pergunta para especialistas**:
  - Como quantificar custos de FP vs FN? (financeiro, tempo, qualidade de vida)
  - Análise de decisão (decision curve analysis) seria apropriada?
  - Net Benefit metric para avaliar utilidade clínica?

**Q8.3**: Comparação com performance humana

- Não comparamos com radiologistas (inter-rater agreement)
- **Pergunta para especialistas**:
  - Como obter baseline de radiologistas? (anotar subset do test set?)
  - Comparar com radiologista individual ou consenso de 3+ especialistas?
  - AI deve superar ou ser "não-inferior" a humanos? (regulatory perspective)

---

### 3.2 Generalização e Robustez

**Q9.1**: External validation - quais datasets?

- Nosso dataset: Guangzhou (China), pediátrico (1-5 anos)
- **Pergunta para especialistas**:
  - Quais datasets públicos são apropriados para validação externa?
    - ChestX-ray14 (adultos, multi-patologia)
    - MIMIC-CXR (adultos, hospital único)
    - PadChest (adultos, Espanha)
    - VinBigData (adultos, Vietnã)
  - Zero-shot generalization (testar sem re-treinar) ou fine-tune?
  - Quão diferente pode ser a população e ainda ser considerado "external validation"?

**Q9.2**: Robustness testing - quais perturbações?

- Planejamos: Gaussian noise, contrast reduction, rotation
- **Pergunta para especialistas**:
  - Há perturbações específicas de raio-X? (scatter, beam hardening, grid artifacts)
  - Adversarial attacks (FGSM, PGD) são relevantes para imagens médicas?
  - Robustness benchmarks estabelecidos? (ImageNet-C, ImageNet-A equivalentes para medical)

**Q9.3**: Domain shift e dataset bias

- Nosso dataset pode ter biases (equipamento, protocolo, população)
- **Pergunta para especialistas**:
  - Como detectar dataset bias quantitativamente?
  - Domain adaptation techniques (DANN, CORAL) deveriam ser usados?
  - Multi-source domain generalization (treinar em múltiplos hospitais) é viável?

---

### 3.3 Aspectos Regulatórios e Éticos

**Q10.1**: Regulatory approval - FDA, CE Mark, ANVISA

- Sistema seria classificado como Class II ou III medical device?
- **Pergunta para especialistas**:
  - Quais evidências são necessárias para approval? (prospective study, RCT?)
  - 510(k) pathway (equivalência a dispositivo existente) seria aplicável?
  - Software as Medical Device (SaMD) guidelines específicos?

**Q10.2**: Fairness e viés demográfico

- Não analisamos performance por: sexo, idade, etnia, severidade
- **Pergunta para especialistas**:
  - Análise de subgrupos é obrigatória? (disparate impact)
  - Como garantir fairness quando dados demográficos não estão disponíveis?
  - Fairness metrics (demographic parity, equalized odds) aplicam-se a medical AI?

**Q10.3**: Explicabilidade para stakeholders

- Diferentes stakeholders: radiologistas, médicos generalistas, pacientes, reguladores
- **Pergunta para especialistas**:
  - Nível de explicação varia por stakeholder? (heatmap para médico, "90% certeza" para paciente)
  - Counterfactual explanations ("se essa opacidade fosse menor, seria normal") são úteis?
  - Right to explanation (GDPR) aplica-se a AI médica?

---

## 💡 SEÇÃO 4: INOVAÇÃO E CONTRIBUIÇÕES

### 4.1 Novelty e Originalidade

**Q11.1**: O que torna nosso trabalho novel?

- Nossa análise: EfficientNet superior, ensemble underperformance, trade-off analysis
- **Pergunta para especialistas**:
  - Comparação de 3 arquiteturas em setup controlado é contribuição suficiente?
  - Análise de _por que_ ensemble falha é mais interessante que simplesmente reportar?
  - Há gap na literatura que estamos preenchendo especificamente?

**Q11.2**: Como enquadrar ensemble underperformance?

- Pode ser visto como: resultado negativo, lição aprendida, ou insight importante
- **Pergunta para especialistas**:
  - Journals/conferences aceitam bem "negative results"? (ex: ensemble não funcionou)
  - Devemos enquadrar como "quando NOT usar ensemble" (prescriptive guidance)?
  - Meta-análise de ensemble failures em medical imaging seria paper separado?

**Q11.3**: Contribuições metodológicas vs aplicadas

- Metodológica: Comparação de técnicas, análise de falhas
- Aplicada: Sistema funcional para pneumonia
- **Pergunta para especialistas**:
  - Venues de alto impacto (MICCAI, TMI) preferem metodologia ou aplicação?
  - "Better mousetrap" (85% vs 80% acc) é suficiente ou precisa de inovação técnica?
  - Framework generalizável (aplicável a outras doenças) aumenta impacto?

---

### 4.2 Comparação com Estado da Arte

**Q12.1**: Como comparar com Kermany et al. (2018) que atingiu 92.8%?

- Nossa accuracy: 80.29% (EfficientNet) vs 92.8% (Kermany Inception-v3)
- Nossa sensitivity: 99.74% vs 93.2% (superamos!)
- Nossa specificity: 47.86% vs 90.1% (muito inferior)
- **Pergunta para especialistas**:
  - Diferenças de split de dados invalidam comparação direta?
  - Deveríamos re-implementar método de Kermany com nosso split?
  - Como reportar comparação quando não há código/splits oficiais?

**Q12.2**: Benchmarks e leaderboards

- Não há leaderboard oficial para Chest X-Ray Pneumonia dataset
- **Pergunta para especialistas**:
  - Vale a pena criar leaderboard (ex: via Papers With Code)?
  - Standardized splits e evaluation protocol deveriam ser propostos?
  - Como garantir reprodutibilidade? (seeds, hardware, versões de bibliotecas)

**Q12.3**: Comparação multi-dataset

- Nosso modelo só foi testado em um dataset
- **Pergunta para especialistas**:
  - É válido comparar modelos treinados/testados em datasets diferentes?
  - Meta-analysis agregando resultados de múltiplos papers seria útil?
  - Transfer learning cross-dataset (treinar em A, testar em B) é avaliação melhor?

---

### 4.3 Impacto e Relevância

**Q13.1**: Quem se beneficia deste trabalho?

- Pesquisadores, clínicos, desenvolvedores de sistemas, reguladores, pacientes
- **Pergunta para especialistas**:
  - Para maximizar impacto, qual audiência priorizar no paper?
  - Resultados negativos (ensemble failure) interessam a pesquisadores mas não clínicos?
  - Framework open-source + modelos pré-treinados aumentam citações?

**Q13.2**: Deployment viability

- Nossa solução é viável para deployment real?
- **Pergunta para especialistas**:
  - EfficientNet-B0 (5.3M params) é leve o suficiente para edge devices? (tablets, mobile)
  - Quantização (INT8) mantém performance? (TensorRT, ONNX Runtime)
  - Cloud vs edge deployment - qual mais apropriado para pneumonia screening?

**Q13.3**: Socioeconomic impact

- Pneumonia é prevalente em países de baixa renda (sub-Saharan Africa, Southeast Asia)
- **Pergunta para especialistas**:
  - Como adaptar modelo para settings com recursos limitados?
  - Offline models (sem internet) são críticos para regiões remotas?
  - Cost-effectiveness analysis deveria fazer parte do paper?

---

## 📊 SEÇÃO 5: APRESENTAÇÃO E COMUNICAÇÃO

### 5.1 Estrutura do Paper

**Q14.1**: Ordem e ênfase das seções

- Atualmente: Abstract, Intro, Methods, Results, Discussion, Conclusion
- **Pergunta para especialistas**:
  - Ensemble failure deve estar em "Results" ou "Discussion"?
  - Trade-off analysis merece seção separada ou integrar em "Results"?
  - Limitations devem ser seção separada ou final de "Discussion"?

**Q14.2**: Visualizações e figuras

- Planejamos: ROC curves, bar charts, confusion matrix, Grad-CAM
- **Pergunta para especialistas**:
  - Quais figuras são essenciais vs nice-to-have?
  - Saliency maps de casos corretos vs incorretos (error analysis) adicionam valor?
  - Diagrams de arquitetura (flowcharts, network diagrams) são esperados?

**Q14.3**: Suplementary material

- O que colocar no paper vs supplement?
- **Pergunta para especialistas**:
  - Detalhes de hyperparameter tuning vão para supplement?
  - Ablation studies (remover componentes, medir impacto) são suplementares?
  - Additional experiments (ViT, Mixup) que não cabem no main paper?

---

### 5.2 Reprodutibilidade

**Q15.1**: Checklist de reprodutibilidade

- NeurIPS, ICLR, ICML têm checklists obrigatórios
- **Pergunta para especialistas**:
  - MICCAI, TMI, MedIA têm checklists similares?
  - Quais informações são _essenciais_ para reprodução? (seeds, hardware, library versions)
  - Docker container + scripts é suficiente ou código linha-por-linha?

**Q15.2**: Código e modelos - onde hospedar?

- Opções: GitHub, Papers With Code, Hugging Face, Zenodo
- **Pergunta para especialistas**:
  - GitHub é suficiente ou deve ter DOI (Zenodo)?
  - Pre-trained weights devem ser disponibilizados? (copyright, size)
  - Interactive demo (Gradio, Streamlit) aumenta impacto?

**Q15.3**: Dados - questões de privacidade

- Dataset é público (Kaggle) mas tem restrições?
- **Pergunta para especialistas**:
  - Re-distribuir data splits (train/val/test indices) viola ToS do Kaggle?
  - Synthetic data generation (GANs) para augmentar dataset é eticamente aceitável?
  - Federated learning (treinar sem centralizar dados) seria alternativa?

---

### 5.3 Escrita e Linguagem

**Q16.1**: Tom e estilo

- Paper deve ser: técnico, acessível, ou híbrido?
- **Pergunta para especialistas**:
  - Medical imaging venues preferem jargão médico ou ML-friendly language?
  - Abstract deve focar em métrica (80% acc) ou impacto ("reduz sobrecarga de radiologistas")?
  - Primeira pessoa ("we propose") ou passivo ("it is proposed")?

**Q16.2**: Claims e assertiveness

- Quão forte podem ser as afirmações?
- **Pergunta para especialistas**:
  - "EfficientNet is superior" vs "EfficientNet shows promising results" - qual aceito?
  - Claims sobre aplicação clínica sem validação prospectiva - permitido?
  - Diferença entre "demonstrates", "suggests", "indicates" - importa?

**Q16.3**: Limitações - quão honestos ser?

- Temos muitas limitações (dataset pequeno, validação externa, etc.)
- **Pergunta para especialistas**:
  - Listar todas as limitações enfraquece o paper ou demonstra rigor?
  - Como balancear honestidade vs "selling" o trabalho?
  - Reviewers penalizam se não mencionarmos limitações óbvias?

---

## 🎯 SEÇÃO 6: VENUE E PUBLICAÇÃO

### 6.1 Target Venue

**Q17.1**: Qual venue é mais apropriado?

- Opções:
  - **Conferences**: MICCAI, IPMI, CVPR Medical Workshop, NeurIPS Medical
  - **Journals**: IEEE TMI, Medical Image Analysis, Computer Methods in Biomedicine, JMIR
- **Pergunta para especialistas**:
  - Para primeiro paper, conference ou journal é melhor? (timeline, prestígio)
  - MICCAI: mais competitivo mas mais visibilidade?
  - Journal: mais espaço para detalhes mas menos networking?

**Q17.2**: Timeliness e deadline

- MICCAI 2026: deadline ~março 2026
- TMI: rolling submission
- **Pergunta para especialistas**:
  - 4 meses (nov→mar) é suficiente para melhorias + escrita + revisões?
  - Submit early (draft inicial) ou wait (após todas as melhorias)?
  - Preprint (arXiv) antes ou depois de submission?

**Q17.3**: Resubmission strategy

- Se rejeitado, como adaptar?
- **Pergunta para especialistas**:
  - Feedback de reviewers deve guiar próximas experiências?
  - Resubmit para venue similar ou pivot para aplicação diferente?
  - Quanto tempo esperar entre submissions? (ethical implications)

---

### 6.2 Review Process

**Q18.1**: Common reviewer concerns

- Com base em experiência, quais objeções esperar?
- **Pergunta para especialistas**:
  - "Dataset is small" - como rebater? (cross-validation, data augmentation)
  - "No external validation" - é blocker ou limitation aceitável?
  - "Ensemble failure" - será visto como fraqueza ou insight?

**Q18.2**: Rebuttal strategies

- Como responder a críticas durante rebuttal?
- **Pergunta para especialistas**:
  - Adicionar experiências durante rebuttal period (1-2 semanas) é viável?
  - Tone: defensivo vs colaborativo vs agradecido?
  - Quais críticas são "deal-breakers" vs negociáveis?

**Q18.3**: Revision scope

- Se aceito com major revisions, quanto trabalho adicionar?
- **Pergunta para especialistas**:
  - External validation pode ser adicionada em revision? (2-3 meses)
  - Treinar novos modelos (ViT) conta como "within scope" ou "new paper"?
  - Como negociar com area chair se reviewer pede muito?

---

## 🧪 SEÇÃO 7: EXPERIMENTOS ADICIONAIS

### 7.1 Ablation Studies

**Q19.1**: Quais componentes ablacionar?

- Progressive unfreezing, class weights, data augmentation, early stopping
- **Pergunta para especialistas**:
  - Ablation de todos os componentes é necessário ou subset representativo?
  - Como reportar? (tabela com checkmarks) ou (gráficos de impacto)
  - Ablation deve usar best model (EfficientNet) ou all models?

**Q19.2**: Hyperparameter sensitivity

- LR, batch size, epochs, weight decay, dropout
- **Pergunta para especialistas**:
  - Grid search vs random search vs Bayesian optimization - qual reportar?
  - Sensitivity plots (performance vs hyperparameter) são esperados?
  - Optimal hyperparameters são dataset-specific ou generalizáveis?

---

### 7.2 Análises Adicionais

**Q20.1**: Error analysis - o que analisar?

- Casos difíceis, failure modes, confusion entre bacterial vs viral
- **Pergunta para especialistas**:
  - Análise qualitativa (mostrar imagens) ou quantitativa (características de erros)?
  - Clustering de erros (UMAP, t-SNE) revela padrões?
  - Radiologist annotation de casos incorretos (por que modelo errou)?

**Q20.2**: Feature analysis

- Visualizar embeddings, ativações, feature importance
- **Pergunta para especialistas**:
  - Feature space analysis (PCA, t-SNE) adiciona insights?
  - CKA (Centered Kernel Alignment) para comparar representações entre modelos?
  - Probing classifiers para entender o que cada camada aprende?

**Q20.3**: Confidence calibration

- Reliability diagrams, Expected Calibration Error (ECE)
- **Pergunta para especialistas**:
  - Calibração é subestimada em medical imaging papers?
  - Overconfidence vs underconfidence - qual é pior clinicamente?
  - Selective prediction (abstain em casos incertos) deveria ser implementado?

---

## 📚 SEÇÃO 8: LITERATURA E CONTEXTO

### 8.1 Related Work

**Q21.1**: Quais papers são must-cite?

- EfficientNet (Tan & Le), ResNet (He et al.), DenseNet (Huang et al.)
- Kermany et al. (mesmo dataset), Rajpurkar CheXNet, Wang ChestX-ray14
- **Pergunta para especialistas**:
  - Há survey papers de medical image analysis que devemos citar?
  - Seminal works de ensemble learning (Dietterich, Breiman)?
  - Recent works (2023-2025) em pneumonia detection?

**Q21.2**: Posicionamento vs literatura

- Como posicionar nosso trabalho?
- **Pergunta para especialistas**:
  - "First to compare EfficientNet vs ResNet vs DenseNet for pneumonia" - defensável?
  - "First to analyze why ensemble fails" - há precedentes?
  - Citation searching: backward (references) vs forward (cited by) - qual priorizar?

---

### 8.2 Future Work

**Q22.1**: Quais direções propor?

- Multi-task (pneumonia + severity + etiology), multi-modal (CXR + CT + clinical data)
- **Pergunta para especialistas**:
  - Future work deve ser realistic (podemos fazer) ou aspirational (alguém deveria fazer)?
  - Specific (treinar ViT) vs vague (explore deep learning)?
  - Quantas direções propor? (3? 5? 10?)

**Q22.2**: Longitudinal e temporal analysis

- Tracking pneumonia progression over time (múltiplas imagens do mesmo paciente)
- **Pergunta para especialistas**:
  - Temporal models (RNNs, Transformers) para sequências de raios-X?
  - Predicting treatment response vs apenas diagnóstico?
  - Clinical decision support system (CDSS) - próximo passo lógico?

---

## 💬 SEÇÃO 9: PERGUNTAS META

### 9.1 Sobre este Documento

**Q23.1**: Relevância das perguntas

- Algumas perguntas podem ser irrelevantes ou muito específicas
- **Pergunta para especialistas**:
  - Quais destas perguntas são CRÍTICAS para responder antes de submeter?
  - Quais são interesting-to-know mas não blockers?
  - Há perguntas importantes que não incluímos?

**Q23.2**: Priorização

- Não temos tempo para responder todas as 70+ perguntas
- **Pergunta para especialistas**:
  - Top 10 perguntas que mais impactam qualidade do paper?
  - Quais podem ser respondidas com literatura vs experimentos?
  - Quais podem ficar como "future work" sem enfraquecer o paper?

---

### 9.2 Colaboração e Feedback

**Q24.1**: Coautoria

- Este trabalho pode se beneficiar de coautores especialistas?
- **Pergunta para especialistas**:
  - Radiologista como coautor (validação clínica) é necessário?
  - Estatístico (análise rigorosa) agregaria valor?
  - Como abordar potenciais colaboradores? (via email, conference, Twitter/X)

**Q24.2**: Peer feedback informal

- Antes de submissão formal, buscar feedback
- **Pergunta para especialistas**:
  - Lab reading groups são úteis? (apresentar draft, receber críticas)
  - Postar em Twitter/X, LinkedIn, Reddit (r/MachineLearning) para feedback?
  - Preprint em arXiv - vantagens (feedback early) vs desvantagens (scooping)?

---

## 🎓 SEÇÃO 10: RECURSOS E REFERÊNCIAS

### Para Especialistas que Responderem

**Agradecemos imensamente seu tempo e expertise!**

Suas respostas ajudarão a:

- ✅ Tornar o paper mais rigoroso metodologicamente
- ✅ Posicionar contribuições de forma mais impactante
- ✅ Evitar erros comuns e objeções de reviewers
- ✅ Direcionar experimentos futuros de forma mais eficiente

**Como responder**:

1. Escolha perguntas de seu domínio de expertise
2. Não precisa responder todas - qualquer insight é valioso!
3. Referencie papers, datasets, código se possível
4. Indique nível de confiança (opinião vs consenso da área)

**Formato sugerido**:

```
Q[número]: [Pergunta]
R: [Sua resposta]
Confiança: [Alta/Média/Baixa]
Referências: [Paper/Link se aplicável]
```

---

## 📬 Contato e Contribuições

**Documento vivo**: Este documento será atualizado conforme recebermos respostas e novos insights.

**Para contribuir**:

- GitHub Issue: [link do repositório]
- Email: [seu email]
- Twitter/X: [seu handle]

**Próximos passos após feedback**:

1. Compilar respostas em FAQ
2. Priorizar experimentos baseados em consenso
3. Atualizar paper draft
4. Iterar com especialistas antes de submission

---

**Versão**: 1.0  
**Data**: 14 de Novembro de 2025  
**Autores**: Matheus Borges (+ colaboradores)  
**Status**: Aberto para feedback

**Keywords**: Transfer Learning, Ensemble Learning, Medical Image Analysis, Pneumonia Detection, Deep Learning, EfficientNet, Expert Consultation

---

## 📝 Apêndice: Resumo das Áreas de Questionamento

1. **Metodologia** (Q1-Q4): Arquitetura, ensemble, loss functions, augmentation
2. **Avaliação** (Q5-Q7): Métricas, validação estatística, interpretabilidade
3. **Aplicação** (Q8-Q10): Trade-offs clínicos, generalização, aspectos regulatórios
4. **Inovação** (Q11-Q13): Novelty, comparação com SOTA, impacto
5. **Comunicação** (Q14-Q16): Estrutura, reprodutibilidade, escrita
6. **Publicação** (Q17-Q18): Venue selection, review process
7. **Experimentos** (Q19-Q20): Ablations, análises adicionais
8. **Literatura** (Q21-Q22): Related work, future directions
9. **Meta** (Q23-Q24): Priorização, colaboração

**Total**: 24 áreas principais, 70+ perguntas específicas

---

## 🙏 Agradecimentos Antecipados

Agradecemos antecipadamente a:

- Pesquisadores em Medical Image Analysis
- Especialistas em Deep Learning e Computer Vision
- Radiologistas e profissionais de saúde
- Revisores e membros de program committees
- Comunidade open-source (PyTorch, timm, Albumentations)

**Seu conhecimento é essencial para elevar a qualidade e impacto deste trabalho!**
