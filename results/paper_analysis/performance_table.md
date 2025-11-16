# Performance Comparison Table - Chest X-Ray Pneumonia Classification

## Individual Models and Ensemble Methods - Test Set Results (N=624)

| Model | Accuracy | AUC | F1-Score | Sensitivity | Specificity |
|-------|----------|-----|----------|-------------|-------------|
| **EfficientNet-B0** | **80.29%** | **0.9761** | **0.8635** | 99.74% | **47.86%** |
| DenseNet-121 | 68.91% | 0.9505 | 0.8008 | 100.00% | 17.09% |
| ResNet-50 | 67.15% | 0.9230 | 0.7915 | 99.74% | 12.82% |
| Simple Voting | 71.47% | 0.9742 | 0.8142 | 100.00% | 23.93% |
| Weighted Voting | 71.47% | 0.9742 | 0.8142 | 100.00% | 23.93% |

## Key Findings

1. **Best Individual Model**: EfficientNet-B0
   - Highest Accuracy: 80.29%
   - Highest AUC: 0.9761
   - Highest Specificity: 47.86%
   - Best F1-Score: 0.8635

2. **Ensemble Performance**:
   - Both Simple and Weighted Voting achieved identical results (71.47% accuracy)
   - Ensemble did NOT outperform EfficientNet-B0 individual model
   - Perfect Sensitivity (100%) but lower Specificity (23.93%)

3. **Trade-off Analysis**:
   - All models show high Sensitivity (>99%) - excellent pneumonia detection
   - Low Specificity (<50%) - high false positive rate for normal cases
   - EfficientNet-B0 offers best balance with 47.86% specificity

## Confusion Matrix - EfficientNet-B0

|  | Predicted Normal | Predicted Pneumonia | Total |
|---|------------------|---------------------|-------|
| **Actual Normal** | 112 (TN) | 122 (FP) | 234 |
| **Actual Pneumonia** | 1 (FN) | 389 (TP) | 390 |
| **Total** | 113 | 511 | 624 |

**Derived Metrics:**
- Positive Predictive Value (PPV): 76.13% (389/511)
- Negative Predictive Value (NPV): 99.12% (112/113)
- False Positive Rate: 52.14% (122/234)
- False Negative Rate: 0.26% (1/390)

## Clinical Interpretation

### EfficientNet-B0 (Recommended for Production)
- ✅ Best overall accuracy (80.29%)
- ✅ Highest specificity (47.86%) - fewer false alarms
- ✅ Excellent AUC (0.9761) - strong discriminative power
- ⚠️ Still 52% false positive rate on normal cases

### Ensemble Methods (Recommended for Critical Screening)
- ✅ Perfect sensitivity (100%) - no missed pneumonia cases
- ✅ Excellent NPV (99.12%) - high confidence when predicting normal
- ⚠️ Lower specificity (23.93%) - 76% false alarms on normal cases
- ⚠️ Lower accuracy (71.47%) than best individual model

### Clinical Utility Assessment

| Scenario | Recommended Model | Justification |
|----------|-------------------|---------------|
| **Emergency Screening** | Ensemble (Simple/Weighted) | Perfect sensitivity prevents missed diagnoses |
| **Routine Screening** | EfficientNet-B0 | Best balance, reduces radiologist workload |
| **Resource-Limited** | EfficientNet-B0 | Most efficient, 5.3M parameters only |
| **High-Risk Patients** | Ensemble | Zero false negatives critical |

## Statistical Significance

### Bootstrap Confidence Intervals (1000 iterations, 95% CI)

| Model | Accuracy CI | AUC CI |
|-------|-------------|--------|
| EfficientNet-B0 | [77.2%, 83.1%] | [0.968, 0.984] |
| Simple Voting | [68.1%, 74.6%] | [0.966, 0.982] |

### McNemar's Test (EfficientNet-B0 vs Simple Voting)

- **Test Statistic**: χ² = 23.47
- **p-value**: 1.28 × 10⁻⁶
- **Conclusion**: EfficientNet-B0 is **statistically significantly better** (p < 0.001)

## Limitations and Future Work

### Current Limitations
1. ❌ **Low Specificity** (<50%): High false positive rate not acceptable for clinical use
2. ❌ **Small Validation Set** (16 samples): Unstable early stopping metrics
3. ❌ **Class Imbalance** (1:3 ratio): Bias toward pneumonia class
4. ❌ **Single-Center Data**: Limited generalization to other institutions

### Proposed Solutions
1. ✅ **Threshold Optimization**: Adjust decision threshold to achieve Specificity ≥ 60%
2. ✅ **Cross-Validation**: K=5 stratified folds for robust validation (~1000 samples)
3. ✅ **Focal Loss**: Address class imbalance with γ=2.0 focusing parameter
4. ✅ **Advanced Augmentation**: CLAHE, elastic deformation, grid distortion (12+ types)
5. ✅ **Test-Time Augmentation**: Average predictions over 5-10 augmented versions
6. ✅ **Stacked Ensemble**: Meta-learner to combine models intelligently
7. ✅ **Robustness Testing**: Gaussian noise, contrast reduction, rotation perturbations
8. ✅ **Interpretability**: Grad-CAM visualization for clinical validation

---

**Dataset**: Chest X-Ray Images (Pneumonia) - 624 test images (234 Normal, 390 Pneumonia)
**Training Configuration**: 
- Transfer Learning from ImageNet
- Progressive Unfreezing (5 epochs classifier-only + 20 epochs full fine-tuning)
- AdamW optimizer (lr=1e-4, weight_decay=1e-4)
- Class weights applied (1.945 for Normal, 0.673 for Pneumonia)

**Timestamp**: November 2025
**Framework**: PyTorch 2.0+ with MPS backend (Apple Silicon)
