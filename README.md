# 🎬 Movie Rating Prediction

A machine learning pipeline that predicts **IMDb movie ratings** for Indian films using regression techniques applied to metadata features such as genre, director, actors, duration, and votes.

---

## Project Structure

```
Movie Rating/
├── IMDb Movies India.csv       # Raw dataset
├── movie_rating_model.py       # Main ML pipeline
├── requirements.txt            # Python dependencies
├── rating_distribution.png     # KDE plot: actual vs. predicted ratings
├── actual_vs_predicted.png     # Scatter plot with regression line
├── feature_importances.png     # Permutation feature importance chart
└── model_comparison.png        # Side-by-side model metric comparison
```

---

## Dataset

**IMDb Movies India.csv** — 15,509 Indian movies with the following columns:

| Column    | Description                          |
|-----------|--------------------------------------|
| Name      | Movie title                          |
| Year      | Release year (raw: `(2019)`)         |
| Duration  | Runtime in minutes (raw: `120 min`) |
| Genre     | Comma-separated genres               |
| Rating    | IMDb rating 1–10 **(target)**        |
| Votes     | Number of user votes                 |
| Director  | Director name                        |
| Actor 1–3 | Top-billed cast members              |

> 7,919 rows have a valid `Rating` and are used for training/evaluation.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python movie_rating_model.py
```

---

## Pipeline Overview

### Data Preprocessing
- **Year** extracted from string format `(YYYY)` → numeric
- **Duration** extracted from `"N min"` → numeric (minutes)
- **Votes** cleaned of commas → numeric; then **log-transformed** (`log1p`) to reduce right-skew
- Rows with missing `Rating` are dropped (target variable cannot be imputed)
- Missing categorical values filled with `"Unknown"`; missing numerics filled with column median

### Feature Engineering
| Feature        | Description                                    |
|----------------|------------------------------------------------|
| `Year`         | Release year                                   |
| `Duration`     | Movie runtime (minutes)                        |
| `Log_Votes`    | Log-transformed vote count                     |
| `Genre_count`  | Number of genres a movie belongs to            |
| `Actor_count`  | Number of credited actors (1–3)                |
| `Genre_encoded`| Target-encoded genre                           |
| `Director_encoded` | Target-encoded director                   |
| `Actor 1/2/3_encoded` | Target-encoded cast members           |

**Target Encoding** (sklearn `TargetEncoder` with 5-fold cross-fitting) is used for high-cardinality categoricals (Director: 3,139 unique values; Actors: 2,500–3,000) to prevent data leakage and avoid an explosion of one-hot columns.

---

## Models & Results

All models trained on 80% of data, evaluated on held-out 20%:

| Model                   | MAE    | RMSE   | R²     |
|-------------------------|--------|--------|--------|
| Ridge Regression        | 0.8772 | 1.1300 | 0.3132 |
| Random Forest           | 0.8022 | 1.0516 | 0.4052 |
| Gradient Boosting       | 0.7921 | 1.0459 | 0.4116 |
| **Hist Gradient Boosting** | **0.7855** | **1.0370** | **0.4215** |

**Best model: Hist Gradient Boosting** with R² = 0.4215, meaning the model explains ~42% of the variance in movie ratings — a reasonable result given that subjective audience taste and content quality are not captured in metadata alone.

---

## Output Visualisations

| File | Description |
|------|-------------|
| `rating_distribution.png` | KDE density curves of actual vs. predicted rating distributions |
| `actual_vs_predicted.png` | Scatter plot showing model predictions against ground truth |
| `feature_importances.png` | Permutation feature importance — which features most affect predictions |
| `model_comparison.png`    | Side-by-side bar charts comparing MAE, RMSE, and R² across all models |

---

## Key Insights

- **Log_Votes** is the single most important feature — movies with more votes tend to have more stable, higher ratings (popularity signal)
- **Director** and **Actor** encodings carry meaningful signal, validating the role of creative talent in rating outcomes
- **Year** provides a weak but present trend — newer films may have rating inflation or deflation effects
- Genre complexity (`Genre_count`) gives a small but consistent boost

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Python 3.9+ required.
