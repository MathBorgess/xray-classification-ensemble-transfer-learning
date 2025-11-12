# 📊 Ensemble e Transfer Learning para Classificação de Imagens Médicas

## Contextualização do Projeto

Este projeto tem como objetivo aplicar **Transfer Learning** e **Ensemble Learning** na classificação de imagens de raio-X torácico, distinguindo entre casos **Normais** e **Pneumonia**.  
A proposta combina arquiteturas pré-treinadas (ResNet50, DenseNet121 e EfficientNetB0) com esquemas de votação simples e ponderada, buscando maior robustez e interpretabilidade.

### Motivação

- Pneumonia é responsável por milhões de mortes anuais (OMS).
- Escassez de radiologistas em regiões críticas aumenta a necessidade de sistemas CAD.
- Transfer Learning mitiga a limitação de dados médicos anotados.
- Ensemble Learning aumenta robustez e estabilidade das previsões.

---

## 🖥️ Resultados Individuais dos Modelos

| Modelo             | Test Accuracy | Test AUC   | F1-Score   | Sensibilidade | Especificidade |
| ------------------ | ------------- | ---------- | ---------- | ------------- | -------------- |
| **ResNet50**       | 0.6715        | 0.9230     | 0.7915     | 0.9974        | 0.1282         |
| **DenseNet121**    | 0.6891        | 0.9505     | 0.8008     | 1.0000        | 0.1709         |
| **EfficientNetB0** | **0.8029**    | **0.9761** | **0.8635** | 0.9974        | **0.4786**     |

🔎 **Insights:**

- Todos os modelos alcançaram **alta sensibilidade (~100%)**, indicando excelente capacidade de detectar pneumonia.
- O **EfficientNetB0** se destacou em acurácia, AUC e especificidade, mostrando maior equilíbrio entre classes.
- ResNet e DenseNet apresentaram boa performance em recall, mas baixa especificidade (tendência a falso positivo).

---

## 🚀 Estado Atual do Desenvolvimento

1. **Treinamento Individual Concluído**

   - ResNet50, DenseNet121 e EfficientNetB0 foram treinados com _progressive unfreezing_ e early stopping.
   - Resultados consolidados em métricas de teste.

2. **Documentação e Estruturação da Metodologia**

   - Artigo já descreve claramente etapas: preparação de dados, fine-tuning, ensemble e avaliação.
   - Cronograma está sendo seguido com documentação semanal.

3. **Alinhamento com Objetivo Final**
   - O foco agora é **integrar os modelos em um ensemble** (votação simples e ponderada).
   - Pesos para votação ponderada serão definidos proporcionalmente ao AUC de validação.

---

## 📌 Próximos Passos

- **Semana 4 (atual):**

  - Coletar predições dos três modelos no conjunto de teste.
  - Implementar **ensemble simples** e **ensemble ponderado**.
  - Comparar métricas (acurácia, AUC, F1, sensibilidade, especificidade).

- **Semana 5:**

  - Testar robustez sob perturbações (ruído, contraste, rotação).
  - Avaliar significância estatística com teste t-pareado.

- **Semana 6:**
  - Organizar relatório final e apresentação.
  - Gerar visualizações interpretáveis (Grad-CAM) para explicar decisões dos modelos.

---

## 🎯 Alinhamento Estratégico

- **Objetivo imediato:** validar se o ensemble supera os modelos individuais em equilíbrio entre sensibilidade e especificidade.
- **Objetivo final:** entregar um sistema robusto, interpretável e documentado, pronto para ser apresentado como tese da disciplina.
- **Risco atual:** baixo número de amostras de validação (16 imagens) pode limitar estabilidade estatística → mitigação via _cross-validation_ ou _bootstrapping_.

---

## 📂 Conclusão

O desenvolvimento está **bem alinhado com o cronograma** e já apresenta resultados promissores, especialmente com o EfficientNetB0.  
O próximo marco crítico será a **implementação do ensemble**, que deve consolidar ganhos de robustez e equilibrar métricas de desempenho.
