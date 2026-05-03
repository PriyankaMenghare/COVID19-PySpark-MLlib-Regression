# COVID-19 Death Prediction — PySpark MLlib 🦠📊

A PySpark MLlib pipeline analyzing COVID-19 case data from India to predict daily 
deaths using Linear Regression, Random Forest, and Gradient Boosted Trees. Built 
on Databricks with SparkSession, StandardScaler, and comprehensive model evaluation.

---

## 🧩 Objective

- Apply Machine Learning models on COVID-19 dataset using **MLlib - PySpark**
- Build, train and evaluate regression models on **Databricks**
- Predict daily death counts and evaluate model performance

---

## 📦 Dataset

- **File:** `covid_dataset.csv`
- **Size:** 150 rows × 5 columns
- **Coverage:** India — daily COVID-19 records

### Features
| Column | Description |
|--------|-------------|
| Date | Date of record |
| Country | Country name (India) |
| Confirmed | Total confirmed cases |
| Recovered | Total recovered cases |
| Deaths | Total deaths ← **Target** |

### Key Correlations
| Feature Pair | Correlation |
|-------------|:-----------:|
| Confirmed ↔ Recovered | 0.9984 |
| Confirmed ↔ Deaths | 0.9926 |
| Recovered ↔ Deaths | 0.9849 |

> All features are highly correlated — indicating a consistent relationship 
> between case progression and outcomes.

---

## 🔬 Methodology

### 1. EDA & Insights
- 5-row sanity check, schema and description
- Aggregated totals for India:
  - **Confirmed:** 262,574,045
  - **Recovered:** 193,560,294
  - **Deaths:** 5,003,949
- Bar chart (log scale) for case type comparison
- Correlation analysis via Spark `.corr()` + seaborn heatmap

### 2. Preprocessing
- **Missing values:** None found across all columns
- **Normalization:** Not applied — count data follows Poisson distribution; normalization would distort it
- **Feature Engineering:**
  - Extracted `Year`, `Month`, `Day` from `Date` column
  - `VectorAssembler` → combined `[Year, Month, Day, Confirmed, Recovered]`
  - `StandardScaler` → standardized feature vector

### 3. Data Preparation
- **Features (X):** `scaled_features` (Year, Month, Day, Confirmed, Recovered)
- **Target (Y):** `Deaths`
- **Train/Test Split:** 80% / 20%

---

## 🤖 Models & Results

### Training Metrics

| Model | R² (Train) | RMSE (Train) | MAE (Train) |
|-------|:----------:|:------------:|:-----------:|
| **Linear Regression** | **0.9993** | 751.29 | 602.41 |
| Random Forest | 0.9987 | 1022.04 | 763.06 |
| Gradient Boosted Trees | 0.99974 | 459.02 | 346.94 |

### Test Metrics

| Model | R² (Test) | RMSE (Test) | MAE (Test) |
|-------|:---------:|:-----------:|:----------:|
| **Linear Regression** | **0.9988** ✅ | 840.64 | 711.97 |
| Random Forest | 0.9976 | 1213.04 | 945.11 |
| Gradient Boosted Trees | 0.9975 | 1241.82 | 911.15 |

### Confusion Matrices (Threshold: 100 deaths)

```
Linear Regression:
                 Predicted Positive  Predicted Negative
Actual Positive          23                  0
Actual Negative           1                  0

Random Forest:
                 Predicted Positive  Predicted Negative
Actual Positive          24                  0
Actual Negative           0                  0
```

### Sample Predictions (Linear Regression)
| Actual Deaths | Predicted Deaths |
|:------------:|:----------------:|
| 1,391 | -292.28 |
| 1,889 | 613.48 |
| 2,649 | 2,201.46 |
| 9,900 | 11,183.05 |

---

## 📝 Key Findings

- **Linear Regression** — best generalizing model; consistent train/test performance, no overfitting
- **Gradient Boosted Trees** — signs of overfitting; excellent on train but higher test error
- **Random Forest** — middle ground; better generalization than GBT but weaker than LR on test data
- **Conclusion:** Linear Regression is recommended when generalization to unseen data is the priority

---

## 🛠️ Setup & Usage

### Prerequisites
- Databricks account (recommended) or local PySpark setup
- Python 3.8+

### Install Dependencies
```bash
pip install pyspark pandas numpy matplotlib seaborn
```

### Run on Databricks
1. Upload `covid_dataset.csv` to `/FileStore/tables/`
2. Import and run `COVID19_PySpark.ipynb`

### Run Locally
```bash
jupyter notebook COVID19_PySpark.ipynb
```

> Update the file path from `/FileStore/tables/covid_dataset.csv` to your local path.

---

## 📁 File Structure

```
COVID19-PySpark-MLlib-Regression/
├── COVID19_PySpark.ipynb     # Main notebook
├── covid_dataset.csv         # COVID-19 India dataset
└── README.md
```

---

## 📝 License

MIT License — for academic use.
