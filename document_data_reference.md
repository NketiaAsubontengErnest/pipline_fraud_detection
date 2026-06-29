# Fraud Detection Research — Document Data Reference

**Project:** Real-Time Fraud Detection Pipeline (ML + Kafka + FastAPI)  
**Document:** Research - Solo Formart - Gemini - claude expanded C2 - backup-13-06-26.docx  
**Last updated:** 2026-06-28  
**Source of truth:** `metrics.json` + actual training run outputs

---

## 1. Dataset Overview (Table 2)

| Metric                   | Value              |
|--------------------------|--------------------|
| Total Transactions       | 100,002            |
| Total Legitimate (Class 0) | 91,499 (91.50%) |
| Total Fraudulent (Class 1) | 8,503  (8.50%)  |

---

## 2. Dataset Split

| Split      | Ratio | Transactions |
|------------|-------|-------------|
| Training   | 70%   | ~70,001     |
| Validation | 15%   | ~15,000     |
| Test       | 15%   | 15,001      |

### Test Set Breakdown (Stratified)

| Class       | Count  | % of Test |
|-------------|--------|-----------|
| Legitimate  | 13,725 | 91.49%    |
| Fraudulent  | 1,276  | 8.51%     |
| **Total**   | **15,001** | 100% |

---

## 3. Resampling — K-Means SMOTE-ENN (Table 4)

| Class           | Before Resampling | After Resampling | Percentage |
|-----------------|-------------------|------------------|------------|
| 0 (Legitimate)  | 91,499            | 43,147*          | 50%        |
| 1 (Fraudulent)  | 8,503             | 43,147*          | 50%        |
| **Total**       | **100,002**       | **86,294***      | 100%       |

> *From previous training run. Update if retraining produces different balanced counts.

---

## 4. Model Performance Comparison (Table 5)

| Algorithm                              | Precision | Recall  | F1-Score | AUC-ROC |
|----------------------------------------|-----------|---------|----------|---------|
| **Extreme Gradient Boosting (Winner)** | 0.6313    | 0.7406  | 0.6816   | 0.9494  |
| Light Gradient Boosting Machine        | 0.6335    | 0.7234  | 0.6754   | 0.9489  |
| Random Forest                          | 0.6253    | 0.7179  | 0.6684   | 0.9456  |

### Full Precision Values (from metrics.json)

| Algorithm                    | Precision (full)   | Recall (full)      | F1-Score (full)    | AUC-ROC (full)     |
|------------------------------|--------------------|--------------------|--------------------|--------------------|
| Extreme Gradient Boosting    | 0.6312625250501002 | 0.7405956112852664 | 0.6815723043635052 | 0.9494282565622306 |
| Light Gradient Boosting Machine | 0.6334934797529169 | 0.7233542319749217 | 0.6754482253933407 | 0.9488563989242338 |
| Random Forest                | 0.6252559726962458 | 0.7178683385579937 | 0.6683692083181321 | 0.9455636066715772 |

### Winner Reasoning

- **XGBoost wins** because it has the **highest AUC-ROC (0.9494)** and **highest Recall (74.06%)**
- In fraud detection, recall is critical — missing a fraud case is more costly than a false alarm
- XGBoost also has the **lowest false negatives (331)** — fewest undetected fraud cases
- LightGBM has slightly higher precision (63.35%) but loses on AUC and Recall

---

## 5. Best Model Display Metrics (Figure 8 Bar Chart)

These are the 4-metric values shown on the `/metrics` dashboard bar chart (XGBoost):

| Metric    | Display Value |
|-----------|--------------|
| Precision | 0.631        |
| Recall    | 0.741        |
| F1-score  | 0.682        |
| AUC-ROC   | 0.949        |

---

## 6. Meta-Learner (Ensemble Stack)

| Metric    | Value              |
|-----------|--------------------|
| Precision | 0.6612 (66.12%)    |
| Recall    | 0.6959 (69.59%)    |
| F1-Score  | 0.6781 (67.81%)    |
| AUC-ROC   | 0.9484             |
| Threshold | 0.12               |

> Note: The meta-learner stacks predictions from all 3 models. Its AUC (0.9484) is slightly below XGBoost's (0.9494), so XGBoost is reported as the Winner individually. The meta-learner is used in production for final decisions.

---

## 7. Confusion Matrix — Random Forest (Table 6 / Figure 11)

**Decision Threshold:** 0.17

|                         | Predicted Legitimate (0) | Predicted Fraudulent (1) |
|-------------------------|--------------------------|--------------------------|
| **Actual Legitimate (0)** | **13,176** (TN)        | **549** (FP)             |
| **Actual Fraudulent (1)** | **360** (FN)           | **916** (TP)             |

### Derived Metrics

| Metric      | Calculation                           | Value    |
|-------------|---------------------------------------|----------|
| Accuracy    | (13,176 + 916) / 15,001               | 93.94%   |
| Precision   | 916 / (916 + 549) = 916 / 1,465      | 62.53%   |
| Recall      | 916 / (916 + 360) = 916 / 1,276      | 71.79%   |
| F1-Score    | 2 × 0.6253 × 0.7179 / (0.6253+0.7179)| 0.6684   |
| AUC-ROC     | —                                     | 0.9456   |
| Specificity | 13,176 / 13,725                       | 95.99%   |
| FPR         | 549 / 13,725                          | 4.00%    |

**Verify:** Legit total = 13,176 + 549 = 13,725 ✓ | Fraud total = 360 + 916 = 1,276 ✓

---

## 8. Confusion Matrix — LightGBM (Table 7 / Figure 12)

**Decision Threshold:** 0.06

|                         | Predicted Legitimate (0) | Predicted Fraudulent (1) |
|-------------------------|--------------------------|--------------------------|
| **Actual Legitimate (0)** | **13,191** (TN)        | **534** (FP)             |
| **Actual Fraudulent (1)** | **353** (FN)           | **923** (TP)             |

### Derived Metrics

| Metric      | Calculation                           | Value    |
|-------------|---------------------------------------|----------|
| Accuracy    | (13,191 + 923) / 15,001               | 94.09%   |
| Precision   | 923 / (923 + 534) = 923 / 1,457      | 63.35%   |
| Recall      | 923 / (923 + 353) = 923 / 1,276      | 72.34%   |
| F1-Score    | 2 × 0.6335 × 0.7234 / (0.6335+0.7234)| 0.6754   |
| AUC-ROC     | —                                     | 0.9489   |
| Specificity | 13,191 / 13,725                       | 96.11%   |
| FPR         | 534 / 13,725                          | 3.89%    |

**Verify:** Legit total = 13,191 + 534 = 13,725 ✓ | Fraud total = 353 + 923 = 1,276 ✓

---

## 9. Confusion Matrix — XGBoost / Winner (Table 8 / Figure 13)

**Decision Threshold:** 0.075

|                         | Predicted Legitimate (0) | Predicted Fraudulent (1) |
|-------------------------|--------------------------|--------------------------|
| **Actual Legitimate (0)** | **13,173** (TN)        | **552** (FP)             |
| **Actual Fraudulent (1)** | **331** (FN)           | **945** (TP)             |

### Derived Metrics

| Metric      | Calculation                           | Value    |
|-------------|---------------------------------------|----------|
| Accuracy    | (13,173 + 945) / 15,001               | 94.11%   |
| Precision   | 945 / (945 + 552) = 945 / 1,497      | 63.13%   |
| Recall      | 945 / (945 + 331) = 945 / 1,276      | **74.06% ← HIGHEST** |
| F1-Score    | 2 × 0.6313 × 0.7406 / (0.6313+0.7406)| **0.6816 ← HIGHEST** |
| AUC-ROC     | —                                     | **0.9494 ← HIGHEST** |
| Specificity | 13,173 / 13,725                       | 95.98%   |
| FPR         | 552 / 13,725                          | 4.02%    |
| False Negatives | 331                               | **331 ← LOWEST** |

**Verify:** Legit total = 13,173 + 552 = 13,725 ✓ | Fraud total = 331 + 945 = 1,276 ✓

---

## 10. Model Comparison Summary

| Model      | TP  | FP  | TN     | FN  | Threshold | FN Rank |
|------------|-----|-----|--------|-----|-----------|---------|
| XGBoost    | 945 | 552 | 13,173 | 331 | 0.075     | 1st (least missed fraud) |
| LightGBM   | 923 | 534 | 13,191 | 353 | 0.06      | 2nd |
| Random Forest | 916 | 549 | 13,176 | 360 | 0.17   | 3rd |

---

## 11. Figures in the Document

| Figure | Title | Chapter/Section |
|--------|-------|-----------------|
| Figure 1  | Data Preprocessing Pipeline | 3.3 |
| Figure 2  | Class Distribution Before Balancing | 3.4 |
| Figure 3  | Hybrid Resampling Pipeline | 3.4.2 |
| Figure 4  | Full Architecture — Real-Time Fraud Detection | 3.7.1 |
| Figure 5  | Visualization of Raw Class Distribution | 4.1.1 |
| Figure 6  | CSV Preview of Transaction Feature Space | 4.1.2 |
| Figure 7  | Comparative Charts Before vs After Balancing | 4.2.2 |
| Figure 8  | Model Evaluation Metrics Bar Chart (XGBoost) | 4.3.1 |
| Figure 9  | F1-Score Curve and AUC-ROC Curve | 4.3.2 |
| Figure 10 | Recall Curve — Model Sensitivity Across Decision Thresholds | 4.3.2 |
| Figure 11 | Heatmap Visualization of the Random Forest Confusion Matrix | 4.4.1 |
| Figure 12 | Heatmap Visualization of the LightGBM Confusion Matrix | 4.5.1 |
| Figure 13 | Heatmap Visualization of the XGBoost Confusion Matrix | 4.5.2 |

> Figures 10, 12, 13 have empty image placeholders in the document.
> Insert the PNG files from the `/static/` folder.

---

## 12. Tables in the Document

| Table | Title | Chapter/Section |
|-------|-------|-----------------|
| Table 1 | Sample Credit Card Transaction Dataset | 3.2.1 |
| Table 2 | Original Dataset Overview | 4.1.1 |
| Table 3 | Data Descriptive Statistics | 4.1.2 |
| Table 4 | Distribution Summary via K-Means SMOTE-ENN | 4.2.2 |
| Table 5 | Model Performance Comparison (Algorithm Evaluation) | 4.3.1 |
| Table 6 | Confusion Matrix Results (Random Forest) | 4.4.1 |
| Table 7 | Confusion Matrix Results (LightGBM) | 4.5.1 |
| Table 8 | Confusion Matrix Results (XGBoost) | 4.5.2 |

---

## 13. Key Reference Values

| What to cite | Value | Where used |
|---|---|---|
| Best AUC-ROC | 0.9494 (XGBoost) | Throughout Ch.4 |
| AUC on chart/dashboard | 0.949 | Figure 8 / Figure 9 caption |
| Best F1-score | 0.6816 | Table 5, text comparisons |
| Best Recall | 74.06% (0.7406) | Table 5, XGBoost justification |
| Highest Precision | 63.35% (LightGBM) | Table 5 |
| XGBoost threshold | 0.075 | Table 8, Section 4.5.2 |
| LightGBM threshold | 0.06 | Table 7, Section 4.5.1 |
| RF threshold | 0.17 | Table 6, Section 4.4.1 |
| Meta-learner threshold | 0.12 | System description |
| Lowest false negatives | 331 (XGBoost) | Section 4.5.2 |
| Best accuracy | 94.11% (XGBoost) | Table 8 |
| Test set total | 15,001 | Sections 4.4, 4.5 |
| Legit in test | 13,725 | All confusion matrices |
| Fraud in test | 1,276 | All confusion matrices |

---

## 14. Critical Rules — Do NOT Get These Wrong

1. **Winner = XGBoost** — Never say LightGBM is the winner
2. **No model has Precision 1.0000** — All precision values are approximately 0.63
3. **AUC-ROC = 0.9494** — NOT 0.996 or 0.9922
4. **F1-score = 0.6816** — NOT 0.989
5. **Test set is 13,725 legit : 1,276 fraud** — NOT 7,500 : 7,500
6. **Total transactions = 100,002** — NOT 100,000
7. **LightGBM advantage = highest Precision (63.35%)** — not any other metric
8. **XGBoost advantage = highest Recall + highest AUC + lowest false negatives**
9. **All three models share the same test set** — 13,725 legitimate, 1,276 fraudulent, total 15,001

---

## 15. Static Image Files (in `/static/`)

| File | Description | Figure in doc |
|------|-------------|---------------|
| `static/confusion_matrix.png` | XGBoost heatmap | Figure 13 |
| `static/auc_chart.png` | AUC-ROC curve | Figure 9 |
| `static/f1_chart.png` | F1-score curve | Figure 9 |
| `static/recall_chart.png` | Recall curve | Figure 10 |
| `static/roc_curve.png` | ROC curve | Figure 9 |
| `static/distribution_comparison.png` | Before/after resampling | Figure 7 |
| `static/precision_chart.png` | Precision curve | — |
