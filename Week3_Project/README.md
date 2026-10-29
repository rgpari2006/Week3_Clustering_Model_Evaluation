# Week 3: Unsupervised Learning & Model Evaluation

Clustering (K-Means, Hierarchical), dimensionality reduction (PCA) and model evaluation
(cross-validation, confusion matrix, precision, recall, F1-score, ROC-AUC, hyperparameter tuning)
on the scikit-learn **Wine** dataset (178 samples, 13 features, 3 cultivars).

## Contents
| File | Description |
|---|---|
| `Week3_Clustering_Model_Evaluation.ipynb` | Full walkthrough notebook |
| `analysis.py` | Same pipeline as a script; saves figures to `figures/` and metrics to `results.json` |
| `Week3_Model_Evaluation_Report.docx` | Written report with plots and conclusions |
| `figures/` | Generated plots |
| `results.json` | All numeric results |

## Run
```bash
pip install -r requirements.txt
python analysis.py            # or: jupyter notebook Week3_Clustering_Model_Evaluation.ipynb
```

## Method
1. Standardise features; PCA for variance analysis and 2-D visualisation.
2. K-Means: elbow method and silhouette score for k = 2..10.
3. Hierarchical clustering: Ward dendrogram and comparison of ward / complete / average / single linkage.
4. Validate clusters against the true labels with the Adjusted Rand Index and cross-tabulations.
5. Classification: Logistic Regression and Random Forest, 5-fold stratified cross-validation,
   `GridSearchCV` tuning, then confusion matrix, precision, recall, F1 and one-vs-rest ROC-AUC on a 25% hold-out set.

## Key results
| Item | Result |
|---|---|
| Best k (silhouette and elbow) | 3 |
| K-Means (k=3): silhouette / ARI | 0.285 / 0.897 |
| Hierarchical Ward (k=3): silhouette / ARI | 0.277 / 0.790 |
| PCA: variance in 2 components | 55.4% (8 components for 90%) |
| Tuned models, hold-out test (45 wines) | accuracy, F1 and ROC-AUC all 1.00 |
| Tuned models, 5-fold CV on all data | accuracy 0.972 (LogReg) / 0.978 (RF) |

The 45-sample test set is small, so the cross-validated figures are the more reliable estimate.
Random seed 42 is used throughout.
