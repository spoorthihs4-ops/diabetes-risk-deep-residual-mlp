# Deep Residual MLP for Diabetes Risk Prediction

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP%20%2B%20IG-6D4AB8)

Can a deep network screen for diabetes from **survey answers alone**? This project builds a residual multilayer perceptron in PyTorch on **253,680 CDC BRFSS 2015 records**, handles the 13.9% class imbalance without leaking information into evaluation, chooses clinically meaningful decision thresholds, and explains every prediction with SHAP and Integrated Gradients.

> MSc Data Science project, University of Hertfordshire. A full written report is in [`reports/`](reports/Deep_Residual_MLP_Report.pdf).

---

## Highlights

| | |
|---|---|
| **Discrimination** | Test **AUC 0.9015**, average precision 0.6703 |
| **Screening mode** | **81.4% sensitivity** with negative predictive value **0.965** (threshold 0.35, chosen on validation) |
| **Calibration** | Brier score **0.0937** |
| **No leakage** | Split → scale → SMOTE on train only; every validation-test gap below 0.01 |
| **Explainability** | SHAP and Integrated Gradients agree: BMI, general health, high blood pressure, age, cholesterol |
| **Honest ablations** | 20 architectures and 13 regularisation variants show capacity is not the bottleneck - dropout is what matters |

## Methodology

![Methodology flowchart](reports/figures/methodology_flowchart.png)

**[Explore the interactive methodology](https://spoorthihs4-ops.github.io/diabetes-risk-deep-residual-mlp/)** – click any stage to see what it does, why it matters, the result it produced and the techniques behind it.

<details>
<summary><b>Pipeline as a live diagram (zoom and pan on GitHub)</b></summary>

```mermaid
flowchart TB
    A["1 · Clinical problem<br/>253,680 records · 21 features"]:::c1 --> B["2 · Leakage-free split<br/>70/15/15 · scaler on train"]:::c2
    B --> C["3 · SMOTE on train only<br/>13.9% → 33.3% positives"]:::c3
    C --> D["4 · Clinical EDA<br/>distributions · correlation · PCA"]:::c4
    D --> E["5 · Deep residual MLP<br/>21 → 512 × 5 → 1"]:::c5
    E --> F["6 · Depth × width ablation<br/>20 configs within 0.003 AUC"]:::c6
    F --> G["7 · Train<br/>focal loss · AdamW · cosine"]:::c7
    G --> H["8 · Clinical thresholds<br/>F1-optimal · 80% sensitivity"]:::c8
    H --> I["9 · Evaluate & explain<br/>AUC 0.9015 · SHAP · IG"]:::c9
    I --> J["10 · Regularisation ablation<br/>loss · activation · dropout"]:::c10

    classDef c1 fill:#2B59C3,stroke:#1d3f8f,color:#fff
    classDef c2 fill:#0E9384,stroke:#0a6b60,color:#fff
    classDef c3 fill:#7A4FD1,stroke:#5a37a3,color:#fff
    classDef c4 fill:#B5338A,stroke:#862566,color:#fff
    classDef c5 fill:#E25A1C,stroke:#a8420f,color:#fff
    classDef c6 fill:#D99A06,stroke:#a37304,color:#fff
    classDef c7 fill:#1D7A8C,stroke:#135563,color:#fff
    classDef c8 fill:#D1335B,stroke:#9c2443,color:#fff
    classDef c9 fill:#16A34A,stroke:#0f7a37,color:#fff
    classDef c10 fill:#1F2A44,stroke:#0f1626,color:#fff
```
</details>

## Results

### Test set (38,052 records, real 13.9% prevalence)

| Metric | Default (0.50) | F1-optimal (0.45) | Screening (0.35) |
|---|---|---|---|
| Sensitivity | 0.521 | 0.629 | **0.814** |
| Specificity | 0.962 | 0.933 | 0.827 |
| Precision | 0.691 | 0.603 | 0.431 |
| NPV | 0.926 | 0.940 | **0.965** |
| F1 (diabetic) | 0.594 | **0.616** | 0.564 |
| MCC | 0.546 | **0.553** | 0.504 |

Threshold-free: **AUC 0.9015 · average precision 0.6703 · Brier 0.0937**. Both thresholds were chosen on validation data before the test set was used.

<p align="center">
  <img src="reports/figures/evaluation_panels.png" width="49%" alt="Evaluation panels">
  <img src="reports/figures/shap_importance.png" width="49%" alt="SHAP feature importance">
</p>

### What the ablations showed

| Study | Finding |
|---|---|
| Depth × width (20 configs) | All within **0.003 AUC** (0.9006-0.9036); a 12k-parameter model scores 0.9031 |
| Dropout | Removing dropout drops AUC to **0.8722**; 0.3-0.4 works best |
| Loss | Focal loss and standard BCE tie on AUC (0.9022); weighted BCE needs threshold tuning |
| Activation | GELU, ReLU, Mish and SELU all within 0.002 AUC |

The takeaway: with 21 self-reported features, **performance is limited by the information in the data, not the size of the network**. The biggest practical gains came from leakage-free evaluation and a clinically chosen threshold.

<p align="center">
  <img src="reports/figures/depth_width_heatmaps.png" width="80%" alt="Depth x width ablation">
</p>

## Notebooks

| Notebook | Contents |
|---|---|
| [`00_full_pipeline_run_all`](notebooks/00_full_pipeline_run_all.ipynb) | **Everything, top to bottom - run this to reproduce** |
| [`01_data_preprocessing_and_eda`](notebooks/01_data_preprocessing_and_eda.ipynb) | Objectives, dataset, leakage-free split, scaling, SMOTE, EDA |
| [`02_model_and_depth_width_ablation`](notebooks/02_model_and_depth_width_ablation.ipynb) | Residual MLP, focal loss, utilities, 20-configuration ablation |
| [`03_training_and_threshold_optimisation`](notebooks/03_training_and_threshold_optimisation.ipynb) | Full training, curve dashboard, clinical thresholds |
| [`04_evaluation_and_explainability`](notebooks/04_evaluation_and_explainability.ipynb) | Test metrics, evaluation panels, SHAP, Integrated Gradients, val-test audit |
| [`05_regularisation_ablation_and_conclusions`](notebooks/05_regularisation_ablation_and_conclusions.ipynb) | Loss, activation and dropout ablations, limitations, conclusions |

## Reproduce

1. Download `diabetes_binary_health_indicators_BRFSS2015.csv` from the [Diabetes Health Indicators dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and place it next to the notebook.
2. `pip install -r requirements.txt`
3. Run [`00_full_pipeline_run_all.ipynb`](notebooks/00_full_pipeline_run_all.ipynb) top to bottom. It runs on CPU; the full training took about 29 minutes.

## Limitations

Self-reported, cross-sectional survey data with no laboratory biomarkers (HbA1c, glucose); pre-diabetes and diabetes are merged into one label; exact duplicate survey rows are not removed before splitting; subgroup fairness has not yet been audited.

## Repository structure

```
diabetes-risk-deep-residual-mlp/
├── notebooks/            # 00 full pipeline + 01-05 part notebooks (with outputs)
├── reports/              # written report (PDF + Word) and figures
├── docs/index.html       # interactive methodology (GitHub Pages)
├── requirements.txt
└── README.md
```

## Author

**Dr. Spoorthi H S** · MSc Data Science, University of Hertfordshire  
[LinkedIn](https://linkedin.com/in/dr-spoorthi-2005b6351) · [GitHub](https://github.com/spoorthihs4-ops)
