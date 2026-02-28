# Clinical Phenotyping Pipeline

End-to-end pipeline for identifying clinical subphenotypes in COVID-19 patients using the **MIMIC-IV** database. Covers data extraction, preprocessing, missing data imputation, unsupervised clustering, and SHAP-based explainability.

To run this pipeline with another dataset ignore the SQL scripts and start from the preprocessing notebooks.

---

## Project Structure

```
clinical-phenotyping-pipeline/
│
├── postgresql-scripts/          # SQL data extraction from MIMIC-IV
│   ├── 01_demographics.sql      # Base demographics + comorbidities
│   ├── 02_filter_2017_2022.sql  # Temporal filtering (2017–2022)
│   ├── 03_covid_admission.sql   # COVID cohort identification
│   ├── 04_select_lab_items.sql  # Top 61 lab tests selection
│   ├── 05_control_groups.sql    # Pre/post-COVID control cohorts
│   ├── 06_lab_data.sql          # Lab event extraction
│   ├── 07_first_lab_values.sql  # First lab value per admission
│   ├── 08–11_future_*.sql       # Readmission tracking
│   ├── 12_download.sql          # CSV export queries
│   └── README.md                # Detailed SQL pipeline docs
│
├── data/                        # Data preprocessing & imputation
│   ├── mimic_data/              # Raw MIMIC-IV exports
│   │   ├── raw_data/            # Unprocessed CSV downloads
│   │   ├── complete_data/       # Cleaned datasets
│   │   └── preprocessing-*.ipynb # Cohort preprocessing notebooks
│   ├── missing-analysis.ipynb   # Missing value analysis
│   ├── mice-col-imputation.ipynb # MICE column-wise imputation
│   ├── pmm-imputation.ipynb     # Predictive Mean Matching
│   ├── mixed-imputation.ipynb   # Mixed imputation strategy
│   ├── mice-col/ pmm/ mixed/   # Imputed dataset outputs
│   └── final-data/              # Final datasets for analysis
│
├── Modules/                     # Python analysis modules
│   ├── config.py                # Paths, reference values, feature mappings
│   ├── DataAnalysisModule.py    # Exploratory data analysis
│   ├── ImputationModule.py      # MICE imputation framework
│   ├── AssociationModule.py     # Association rule mining
│   ├── ClusterBaseModule.py     # Base class: stats, PCA/UMAP, visualisation
│   ├── ClusterKmeansModule.py   # K-Means clustering
│   ├── ClusterHierarchicalModule.py # Agglomerative clustering
│   ├── ClusterGmmModule.py      # Gaussian Mixture Models
│   ├── ClusterSpectralModule.py # Spectral clustering
│   ├── ClusterDBSCANModule.py   # DBSCAN clustering
│   ├── ClusterHDBSCANModule.py  # HDBSCAN clustering
│   ├── ClusterSHADEModule.py    # Autoencoder-based (SHADE)
│   ├── ClusterMetricsModule.py  # Cluster validation metrics (SWC, S-Dbw, DBCV, DSI, DISCO)
│   ├── SHAPClassifierModule.py  # SHAP explainability + Optuna hyperparameter tuning
│   ├── FutureAnalysisModule.py  # Readmission & trajectory analysis
│   └── ExternalModules/         # Third-party CVI, DISCO, SHADE implementations
│
├── notebooks/                   # Analysis workflow (run in order)
│   ├── 00-statistics/           # EDA, dimensionality reduction, comorbidity correlations
│   ├── 01-shap/                 # SHAP feature importance classification
│   ├── 02-kmeans/               # K-Means: all features, DR features, top-death features
│   ├── 03-hierarchical/         # Hierarchical clustering experiments
│   ├── 04-gmm/                  # GMM clustering experiments
│   ├── 05-spectral/             # Spectral clustering experiments
│   ├── 06-dbscan/               # DBSCAN clustering experiments
│   ├── 07-hdbscan/              # HDBSCAN clustering experiments
│   ├── 08-shade/                # SHADE deep clustering experiments
│   ├── 09-best-results/         # Best phenotype results comparison
│   └── log_best.csv             # Performance log across all methods
│
├── export/                      # Generated figures and reports
├── requirements.txt             # Python dependencies
└── .gitignore
```

---

## Pipeline Overview

```
 MIMIC-IV Database
       │
       ▼
 ┌─────────────────┐
 │  SQL Extraction  │  postgresql-scripts/
 │  (12 scripts)    │  Demographics, labs, control groups, readmissions
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Preprocessing   │  data/mimic_data/
 │  & Cleaning      │  Cohort-specific preprocessing notebooks
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Missing Data    │  data/
 │  Imputation      │  MICE, PMM, Mixed strategies
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Clustering &    │  notebooks/00–08
 │  Phenotyping     │  7 algorithms × multiple feature sets
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │  Evaluation &    │  notebooks/09, Modules/
 │  Explainability  │  Cluster metrics, SHAP, readmission analysis
 └─────────────────┘
```

### Cohorts

| Cohort | Description | Selection |
|--------|-------------|-----------|
| **COVID** | Patients with ICD-10 U07.1 | First COVID admission, 2020–2022 |
| **Control Pre** | Pre-pandemic controls | Sampled 20K patients, 2017–2019 |
| **Control Post** | Post-pandemic non-COVID | Sampled 20K patients, 2020–2022 |

### Clustering Algorithms

| Algorithm | Module | Key Characteristics |
|-----------|--------|---------------------|
| K-Means | `ClusterKmeansModule` | Centroid-based, Optuna-tuned |
| Hierarchical | `ClusterHierarchicalModule` | Agglomerative, dendrogram analysis |
| GMM | `ClusterGmmModule` | Probabilistic, soft assignments |
| Spectral | `ClusterSpectralModule` | Graph-based, nonlinear boundaries |
| DBSCAN | `ClusterDBSCANModule` | Density-based, noise detection |
| HDBSCAN | `ClusterHDBSCANModule` | Hierarchical density, variable density |
| SHADE | `ClusterSHADEModule` | Autoencoder + clustering |

All modules include **Optuna hyperparameter optimisation** and support **PCA/UMAP** dimensionality reduction for visualisation.

### Validation Metrics

Implemented in `ClusterMetricsModule.py`:
- **SWC** (Silhouette Width Criterion)
- **S-Dbw** (Scatter and Density Between)
- **DBCV** (Density-Based Cluster Validation)
- **VIASCKDE** (Internal Validation via KDE)
- **DSI** (Density Separation Index)
- **CVNN** (Cross-Validated Nearest Neighbours)
- **DISCO** (Distribution-based Cluster Overlap)

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 12+ with MIMIC-IV loaded

### Installation

```bash
git clone https://github.com/GLeyte/clinical-phenotyping-pipeline.git
cd clinical-phenotyping-pipeline

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Running the Pipeline

1. **Extract data**: Execute SQL scripts in `postgresql-scripts/` in numerical order against your MIMIC-IV database (see [SQL README](postgresql-scripts/README.md))
2. **Preprocess**: Run the preprocessing notebooks in `data/mimic_data/`
3. **Impute**: Run imputation notebooks in `data/` to generate `final-data/`
4. **Analyse**: Follow notebooks `00` through `09` in `notebooks/`

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `scikit-learn` | Clustering, scaling, evaluation |
| `umap-learn` | UMAP dimensionality reduction |
| `hdbscan` | HDBSCAN algorithm |
| `xgboost` | XGBoost classifier for SHAP |
| `shap` | Feature importance explanations |
| `optuna` | Hyperparameter optimisation |
| `miceforest` | Multiple imputation (MICE) |
| `torch` | SHADE autoencoder backend |
| `mlxtend` | Association rule mining |
| `pandas`, `numpy`, `matplotlib`, `seaborn` | Data manipulation & visualisation |

---

## License & Citation

This pipeline uses the **MIMIC-IV** database. Users must complete the required CITI training, sign the PhysioNet Data Use Agreement, and cite MIMIC-IV in publications:

> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023).
> MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
