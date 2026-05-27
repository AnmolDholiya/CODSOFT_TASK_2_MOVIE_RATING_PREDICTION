"""
============================================================
  Movie Rating Prediction — IMDb India Dataset
  Full Pipeline: EDA → Preprocessing → Feature Engineering
                 → Model Training → Evaluation → Visualisation
  v3: + CatBoost + NLP Title Features
============================================================
"""

import os
import sys
import re
import warnings
import joblib

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.preprocessing import TargetEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ─────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(BASE_DIR, "IMDb Movies India.csv")
OUTPUT_DIR  = BASE_DIR
MODEL_PATH  = os.path.join(BASE_DIR, "saved_model.joblib")

CATEGORICAL_COLS = ["Director", "Actor 1", "Actor 2", "Actor 3"]
NUMERIC_FEATURES = [
    "Year", "Duration", "Log_Votes", "Decade",
    "Genre_Count", "Actor_Count",
    # NLP title features
    "Is_Sequel", "Title_Length", "Title_Word_Count", "English_Word_Ratio",
]
SEED = 42

sns.set_theme(style="whitegrid", palette="muted")

# ── Common English words for English ratio computation ─────────
ENGLISH_VOCAB = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might","shall",
    "not","no","it","its","this","that","these","those","he","she","they","we",
    "you","i","me","him","her","us","them","my","your","his","our","their",
    "what","which","who","when","where","why","how","all","some","any","each",
    "love","life","day","time","one","two","three","man","woman","men","women",
    "world","great","good","bad","new","old","first","last","long","little",
    "own","same","other","than","then","now","only","just","more","most","over",
    "under","again","out","up","down","into","through","before","after","between",
    "dream","fire","heart","night","dark","light","blood","war","true","story",
    "return","returns","beyond","rise","fall","lost","found","super","ultra",
}


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_data(path):
    for enc in ["utf-8", "latin-1", "utf-8-sig"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"  [OK] Loaded dataset ({enc}): {df.shape[0]:,} rows x {df.shape[1]} columns")
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot read: {path}")


# ─────────────────────────────────────────────────────────────
# 2. NLP TITLE FEATURES
# ─────────────────────────────────────────────────────────────
SEQUEL_PATTERN = re.compile(
    r"(\d+|II|III|IV|VI|VII|VIII|IX|XI|XII|\bpart\b|\breturns?\b|\bagain\b)",
    re.IGNORECASE
)

def extract_title_features(name_series):
    """Extract NLP-based features from movie title strings."""
    names = name_series.fillna("").astype(str)

    # Is Sequel: does the title contain a number or sequel keyword?
    is_sequel = names.apply(lambda x: int(bool(SEQUEL_PATTERN.search(x))))

    # Title Length: number of characters (excluding whitespace)
    title_length = names.apply(lambda x: len(x.replace(" ", "")))

    # Title Word Count: number of words
    title_word_count = names.apply(lambda x: len(x.split()))

    # English Word Ratio: fraction of words found in the English vocabulary
    def english_ratio(text):
        words = [w.lower().strip(".,!?\"'()-") for w in text.split()]
        if not words:
            return 0.0
        return sum(1 for w in words if w in ENGLISH_VOCAB) / len(words)

    english_word_ratio = names.apply(english_ratio)

    return pd.DataFrame({
        "Is_Sequel":          is_sequel,
        "Title_Length":       title_length,
        "Title_Word_Count":   title_word_count,
        "English_Word_Ratio": english_word_ratio,
    }, index=name_series.index)


# ─────────────────────────────────────────────────────────────
# 3. PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def preprocess_base(df):
    df = df.copy()

    before = len(df)
    df = df.dropna(subset=["Rating"])
    print(f"  [OK] Dropped {before - len(df):,} rows with missing Rating → {len(df):,} remain")

    df["Year"]     = df["Year"].astype(str).str.extract(r"(\d{4})").astype(float)
    df["Duration"] = df["Duration"].astype(str).str.extract(r"(\d+)").astype(float)
    df["Votes"]    = (df["Votes"].astype(str)
                      .str.replace(",", "", regex=False)
                      .str.extract(r"(\d+)").astype(float))

    for col in ["Year", "Duration", "Votes"]:
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS + ["Genre"]:
        df[col] = df[col].fillna("Unknown")

    # Basic engineered features
    df["Decade"]      = (df["Year"] // 10) * 10
    df["Log_Votes"]   = np.log1p(df["Votes"])
    df["Genre_Count"] = df["Genre"].apply(lambda x: len(str(x).split(",")))
    df["Actor_Count"] = df.apply(
        lambda r: sum([r["Actor 1"] != "Unknown",
                       r["Actor 2"] != "Unknown",
                       r["Actor 3"] != "Unknown"]), axis=1)
    df["Primary_Genre"] = df["Genre"].apply(lambda x: str(x).split(",")[0].strip())

    # NLP title features
    title_feats = extract_title_features(df["Name"])
    df = pd.concat([df, title_feats], axis=1)

    sequels_found = int(df["Is_Sequel"].sum())
    print(f"  [OK] NLP features extracted — {sequels_found} sequels/series detected in dataset")
    return df


def extract_unique_genres(df):
    genres = set()
    for val in df["Genre"].dropna():
        for g in str(val).split(","):
            genres.add(g.strip())
    return sorted(list(genres))


def add_genre_one_hot(df, genre_list):
    for genre in genre_list:
        df[f"Genre_{genre}"] = df["Genre"].apply(
            lambda x: 1 if genre in [g.strip() for g in str(x).split(",")] else 0
        )
    return df


# ─────────────────────────────────────────────────────────────
# 4. EDA PLOTS
# ─────────────────────────────────────────────────────────────
def plot_eda(df, output_dir):
    print("\n[2/6] Generating EDA dashboard...")

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle("IMDb India Movies — Exploratory Data Analysis",
                 fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)

    # 1. Rating distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["Rating"], bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax1.axvline(df["Rating"].mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean={df['Rating'].mean():.2f}")
    ax1.set_title("Rating Distribution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Rating"); ax1.set_ylabel("Count")
    ax1.legend(fontsize=9)

    # 2. Avg rating by decade
    ax2 = fig.add_subplot(gs[0, 1])
    decade_rating = df.groupby("Decade")["Rating"].mean().reset_index()
    ax2.bar(decade_rating["Decade"].astype(int), decade_rating["Rating"],
            color="#55A868", edgecolor="white", width=7)
    ax2.set_title("Avg Rating by Decade", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Decade"); ax2.set_ylabel("Avg Rating")
    ax2.set_xticks(decade_rating["Decade"].astype(int))
    ax2.set_xticklabels(decade_rating["Decade"].astype(int), rotation=45, fontsize=8)

    # 3. Top 10 genres by count
    ax3 = fig.add_subplot(gs[0, 2])
    top_genres = df["Primary_Genre"].value_counts().head(10)
    ax3.barh(top_genres.index[::-1], top_genres.values[::-1], color="#C44E52")
    ax3.set_title("Top 10 Genres (Count)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Number of Movies")

    # 4. Top 10 genres by avg rating
    ax4 = fig.add_subplot(gs[1, 0])
    genre_rating = df.groupby("Primary_Genre")["Rating"].mean().nlargest(10).sort_values()
    ax4.barh(genre_rating.index, genre_rating.values, color="#8172B2")
    ax4.set_title("Top 10 Genres by Avg Rating", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Avg Rating")

    # 5. Log(Votes) vs Rating scatter
    ax5 = fig.add_subplot(gs[1, 1])
    sample = df.sample(min(2000, len(df)), random_state=SEED)
    ax5.scatter(sample["Log_Votes"], sample["Rating"],
                alpha=0.3, s=10, color="#4C72B0")
    ax5.set_title("Log(Votes) vs Rating", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Log(Votes)"); ax5.set_ylabel("Rating")

    # 6. Avg rating by duration bucket
    ax6 = fig.add_subplot(gs[1, 2])
    dur_bins   = pd.cut(df["Duration"],
                        bins=[0, 60, 90, 120, 150, 180, 400],
                        labels=["<60", "60-90", "90-120", "120-150", "150-180", "180+"])
    dur_rating = df.groupby(dur_bins, observed=True)["Rating"].mean()
    ax6.bar(dur_rating.index.astype(str), dur_rating.values, color="#CCB974")
    ax6.set_title("Avg Rating by Duration (min)", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Duration (minutes)"); ax6.set_ylabel("Avg Rating")

    # 7. Correlation heatmap
    ax7 = fig.add_subplot(gs[2, :2])
    corr_cols = ["Rating", "Year", "Duration", "Log_Votes", "Decade",
                 "Genre_Count", "Actor_Count",
                 "Is_Sequel", "Title_Length", "Title_Word_Count", "English_Word_Ratio"]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax7, annot_kws={"size": 8})
    ax7.set_title("Feature Correlation Heatmap (incl. NLP)", fontsize=12, fontweight="bold")

    # 8. Top directors by avg rating (>=5 movies)
    ax8 = fig.add_subplot(gs[2, 2])
    active_dirs = df[df["Director"] != "Unknown"].groupby("Director").filter(lambda x: len(x) >= 5)
    top_dir = active_dirs.groupby("Director")["Rating"].mean().nlargest(8).sort_values()
    ax8.barh(top_dir.index, top_dir.values, color="#64B5CD")
    ax8.set_title("Top Directors by Avg Rating\n(>=5 movies)", fontsize=12, fontweight="bold")
    ax8.set_xlabel("Avg Rating")

    # 9. Sequel vs Non-Sequel avg rating
    ax9 = fig.add_subplot(gs[3, 0])
    sequel_grp = df.groupby("Is_Sequel")["Rating"].mean()
    labels = ["Non-Sequel", "Sequel"]
    ax9.bar(labels, [sequel_grp.get(0, 0), sequel_grp.get(1, 0)],
            color=["#4C72B0", "#C44E52"], edgecolor="white")
    ax9.set_title("Avg Rating: Sequel vs Non-Sequel", fontsize=12, fontweight="bold")
    ax9.set_ylabel("Avg Rating")
    for i, v in enumerate([sequel_grp.get(0, 0), sequel_grp.get(1, 0)]):
        ax9.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")

    # 10. English word ratio distribution
    ax10 = fig.add_subplot(gs[3, 1])
    ax10.hist(df["English_Word_Ratio"], bins=20, color="#55A868", edgecolor="white")
    ax10.set_title("English Word Ratio in Titles", fontsize=12, fontweight="bold")
    ax10.set_xlabel("Ratio (0=Non-English, 1=Full English)")
    ax10.set_ylabel("Count")

    # 11. Title word count distribution
    ax11 = fig.add_subplot(gs[3, 2])
    ax11.hist(df["Title_Word_Count"].clip(upper=10), bins=10, color="#8172B2", edgecolor="white")
    ax11.set_title("Title Word Count Distribution", fontsize=12, fontweight="bold")
    ax11.set_xlabel("Words in Title")
    ax11.set_ylabel("Count")

    path = os.path.join(output_dir, "01_EDA_Analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] EDA dashboard saved -> {path}")


# ─────────────────────────────────────────────────────────────
# 5. MODELS DEFINITION
# ─────────────────────────────────────────────────────────────
def get_sklearn_models():
    """Models that work on fully numeric feature matrices."""
    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=12,
        min_samples_split=5, random_state=SEED, n_jobs=-1
    )
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5, random_state=SEED
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=6,
        l2_regularization=1.5, random_state=SEED
    )
    xgb = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
        random_state=SEED, verbosity=0
    )
    return {
        "Ridge Regression":      ridge,
        "Random Forest":         rf,
        "Gradient Boosting":     gb,
        "Hist Gradient Boosting": hgb,
        "XGBoost":               xgb,
    }


# ─────────────────────────────────────────────────────────────
# 6. RESULTS PLOTS
# ─────────────────────────────────────────────────────────────
def plot_results(results, y_test, output_dir):
    print("\n[5/6] Generating results plots...")

    model_names = list(results.keys())
    short_names = [n.replace(" ", "\n") for n in model_names]
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#E07B39"]
    colors = palette[:len(model_names)]

    rmse_vals = [results[m]["RMSE"] for m in model_names]
    r2_vals   = [results[m]["R2"]   for m in model_names]
    cv_vals   = [results[m]["CV_R2"]for m in model_names]

    best_name  = max(results, key=lambda k: results[k]["R2"])
    best_preds = results[best_name]["preds"]
    residuals  = np.array(y_test) - best_preds

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Model Evaluation Results", fontsize=16, fontweight="bold")

    # A. RMSE comparison
    bars = axes[0, 0].bar(short_names, rmse_vals, color=colors, edgecolor="white", width=0.55)
    axes[0, 0].set_title("RMSE Comparison (lower = better)", fontweight="bold")
    axes[0, 0].set_ylabel("RMSE")
    for bar, v in zip(bars, rmse_vals):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, v + 0.003,
                        f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # B. R2 + CV R2 grouped bars
    x     = np.arange(len(model_names))
    width = 0.35
    bars1 = axes[0, 1].bar(x - width/2, r2_vals, width, label="Test R²",  color=colors, edgecolor="white")
    axes[0, 1].bar(x + width/2, cv_vals, width, label="CV R²", color=colors, edgecolor="white", alpha=0.55)
    axes[0, 1].set_title("R² Score: Test vs Cross-Validation", fontweight="bold")
    axes[0, 1].set_ylabel("R²")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(short_names, fontsize=8)
    axes[0, 1].legend()
    for bar, v in zip(bars1, r2_vals):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, v + 0.003,
                        f"{v:.3f}", ha="center", fontsize=8)

    # C. Actual vs Predicted (best model)
    axes[1, 0].scatter(y_test, best_preds, alpha=0.3, s=12, color="#4C72B0")
    mn, mx = float(np.array(y_test).min()), float(np.array(y_test).max())
    axes[1, 0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect")
    axes[1, 0].set_title(f"Actual vs Predicted — {best_name}", fontweight="bold")
    axes[1, 0].set_xlabel("Actual Rating"); axes[1, 0].set_ylabel("Predicted Rating")
    axes[1, 0].legend()

    # D. Residual distribution
    axes[1, 1].hist(residuals, bins=40, color="#55A868", edgecolor="white")
    axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1, 1].set_title(f"Residual Distribution — {best_name}", fontweight="bold")
    axes[1, 1].set_xlabel("Residual (Actual - Predicted)")
    axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    path = os.path.join(output_dir, "02_Model_Results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Results plot saved -> {path}")


def plot_feature_importance(best_model, best_name, X_test, y_test, output_dir):
    print("  Computing permutation feature importances...")
    result = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=10, random_state=SEED, n_jobs=-1
    )
    idx    = result.importances_mean.argsort()
    top_n  = min(20, len(idx))
    top_idx = idx[-top_n:]

    imp_df = pd.DataFrame(
        result.importances[top_idx].T,
        columns=X_test.columns[top_idx]
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(data=imp_df, orient="h", palette="viridis", errorbar="sd", ax=ax)
    ax.set_title(f"Top {top_n} Permutation Feature Importances — {best_name}",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Decrease in R²"); ax.set_ylabel("Feature")
    plt.tight_layout()
    path = os.path.join(output_dir, "03_Feature_Importances.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Feature importance saved -> {path}")


def save_metrics_csv(results, output_dir):
    rows = []
    for name, r in results.items():
        rows.append({"Model": name,
                     "RMSE":  round(r["RMSE"],  4),
                     "MAE":   round(r["MAE"],   4),
                     "R2":    round(r["R2"],    4),
                     "CV_R2": round(r["CV_R2"], 4)})
    metrics_df = (pd.DataFrame(rows)
                  .sort_values("R2", ascending=False)
                  .reset_index(drop=True))
    path = os.path.join(output_dir, "04_Model_Metrics.csv")
    metrics_df.to_csv(path, index=False)
    print(f"  [OK] Metrics CSV saved -> {path}")
    print(f"\n{metrics_df.to_string(index=False)}")
    return metrics_df


# ─────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 62)
    print("  Movie Rating Prediction Pipeline — IMDb India  (v3)")
    print("=" * 62)

    # ── Load & preprocess ──────────────────────────────────────
    print("\n[1/6] Loading & preprocessing data...")
    df = preprocess_base(load_data(CSV_PATH))

    # ── EDA ───────────────────────────────────────────────────
    plot_eda(df, OUTPUT_DIR)

    # ── Genre list & one-hot ───────────────────────────────────
    genre_list = extract_unique_genres(df)
    print(f"  [OK] Found {len(genre_list)} unique genres")

    # ── Split ─────────────────────────────────────────────────
    print("\n[3/6] Preparing features and splitting data...")
    X = df.drop(columns=["Rating"])
    y = df["Rating"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Experience counts (computed from train only) ───────────
    director_counts = X_train["Director"].value_counts().to_dict()
    actor_counts = {}
    for col in ["Actor 1", "Actor 2", "Actor 3"]:
        for actor, count in X_train[col].value_counts().items():
            if actor != "Unknown":
                actor_counts[actor] = actor_counts.get(actor, 0) + count

    def add_experience_features(d):
        d = d.copy()
        d["Director_Movie_Count"]   = d["Director"].map(director_counts).fillna(0)
        d["Actor1_Movie_Count"]     = d["Actor 1"].map(actor_counts).fillna(0)
        d["Actor2_Movie_Count"]     = d["Actor 2"].map(actor_counts).fillna(0)
        d["Actor3_Movie_Count"]     = d["Actor 3"].map(actor_counts).fillna(0)
        d["Cast_Movie_Count_Mean"]  = d[["Actor1_Movie_Count",
                                         "Actor2_Movie_Count",
                                         "Actor3_Movie_Count"]].mean(axis=1)
        return d

    X_train_eng = add_experience_features(X_train)
    X_test_eng  = add_experience_features(X_test)

    # ── Genre one-hot ──────────────────────────────────────────
    X_train_eng = add_genre_one_hot(X_train_eng, genre_list)
    X_test_eng  = add_genre_one_hot(X_test_eng,  genre_list)

    # ── Target encoding for sklearn models ─────────────────────
    encoder  = TargetEncoder(smooth="auto", cv=5, random_state=SEED)
    enc_cols = [f"{c}_encoded" for c in CATEGORICAL_COLS]

    X_tr_cat = pd.DataFrame(
        encoder.fit_transform(X_train_eng[CATEGORICAL_COLS], y_train),
        columns=enc_cols, index=X_train_eng.index
    )
    X_te_cat = pd.DataFrame(
        encoder.transform(X_test_eng[CATEGORICAL_COLS]),
        columns=enc_cols, index=X_test_eng.index
    )

    genre_cols   = [f"Genre_{g}" for g in genre_list]
    exp_features = ["Director_Movie_Count", "Actor1_Movie_Count",
                    "Actor2_Movie_Count",   "Actor3_Movie_Count",
                    "Cast_Movie_Count_Mean"]

    X_train_sk = pd.concat([X_train_eng[NUMERIC_FEATURES + genre_cols + exp_features], X_tr_cat], axis=1)
    X_test_sk  = pd.concat([X_test_eng[NUMERIC_FEATURES  + genre_cols + exp_features], X_te_cat], axis=1)

    # Cast aggregate encodings
    cast_enc_cols = ["Actor 1_encoded", "Actor 2_encoded", "Actor 3_encoded"]
    for split_X in [X_train_sk, X_test_sk]:
        split_X["Cast_Rating_Mean"] = split_X[cast_enc_cols].mean(axis=1)
        split_X["Cast_Rating_Max"]  = split_X[cast_enc_cols].max(axis=1)
        split_X["Cast_Rating_Min"]  = split_X[cast_enc_cols].min(axis=1)

    final_features_list = list(X_train_sk.columns)

    # ── CatBoost feature matrix (raw categoricals, no target enc) ─
    cb_cat_features = CATEGORICAL_COLS   # e.g. ["Director","Actor 1","Actor 2","Actor 3"]
    cb_keep = NUMERIC_FEATURES + genre_cols + exp_features + CATEGORICAL_COLS

    X_train_cb = X_train_eng[cb_keep].copy()
    X_test_cb  = X_test_eng[cb_keep].copy()
    # Ensure categoricals are strings for CatBoost
    for col in cb_cat_features:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col]  = X_test_cb[col].astype(str)

    # ── Train models ───────────────────────────────────────────
    print("\n[4/6] Training models...")
    sk_models = get_sklearn_models()
    results   = {}

    header = f"  {'Model':<32} {'RMSE':>7} {'MAE':>7} {'Test R2':>8} {'CV R2':>8}"
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))

    # ── sklearn models ─────────────────────────────────────────
    for name, model in sk_models.items():
        model.fit(X_train_sk, y_train)
        preds  = model.predict(X_test_sk)
        rmse   = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae    = float(mean_absolute_error(y_test, preds))
        r2     = float(r2_score(y_test, preds))
        cv_r2  = float(cross_val_score(model, X_train_sk, y_train,
                                       cv=5, scoring="r2", n_jobs=-1).mean())
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2,
                          "CV_R2": cv_r2, "preds": preds, "model": model}
        print(f"  {name:<32} {rmse:>7.4f} {mae:>7.4f} {r2:>8.4f} {cv_r2:>8.4f}")

    # ── CatBoost ───────────────────────────────────────────────
    from catboost import Pool, cv as catboost_cv

    cb_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function="RMSE",
        cat_features=cb_cat_features,
        random_seed=SEED,
        verbose=0,
    )
    cb_model.fit(X_train_cb, y_train,
                 eval_set=(X_test_cb, y_test),
                 early_stopping_rounds=30,
                 verbose=False)

    cb_preds = cb_model.predict(X_test_cb)
    cb_rmse  = float(np.sqrt(mean_squared_error(y_test, cb_preds)))
    cb_mae   = float(mean_absolute_error(y_test, cb_preds))
    cb_r2    = float(r2_score(y_test, cb_preds))

    # Use CatBoost's native CV (avoids sklearn clone incompatibility)
    cb_pool = Pool(X_train_cb, label=y_train, cat_features=cb_cat_features)
    cb_cv_params = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3,
        "loss_function": "RMSE",
        "random_seed": SEED,
        "verbose": 0,
    }
    cb_cv_results = catboost_cv(
        params=cb_cv_params,
        pool=cb_pool,
        fold_count=5,
        verbose=False,
        plot=False,
    )
    # Best mean RMSE across folds → convert to R²-like metric via 1 - (RMSE²/Var)
    best_cv_rmse = cb_cv_results["test-RMSE-mean"].min()
    y_var = float(np.var(y_train))
    cb_cv = float(1.0 - (best_cv_rmse ** 2) / y_var) if y_var > 0 else 0.0

    results["CatBoost"] = {"RMSE": cb_rmse, "MAE": cb_mae, "R2": cb_r2,
                            "CV_R2": cb_cv, "preds": cb_preds, "model": cb_model,
                            "is_catboost": True}
    print(f"  {'CatBoost':<32} {cb_rmse:>7.4f} {cb_mae:>7.4f} {cb_r2:>8.4f} {cb_cv:>8.4f}")

    # ── Voting Ensemble (sklearn models only — same feature space) ──
    hgb = sk_models["Hist Gradient Boosting"]
    xgb = sk_models["XGBoost"]
    rf  = sk_models["Random Forest"]

    ensemble = VotingRegressor(
        estimators=[("HGB", hgb), ("XGB", xgb), ("RF", rf)],
        n_jobs=-1
    )
    ensemble.fit(X_train_sk, y_train)
    ens_preds = ensemble.predict(X_test_sk)
    ens_rmse  = float(np.sqrt(mean_squared_error(y_test, ens_preds)))
    ens_mae   = float(mean_absolute_error(y_test, ens_preds))
    ens_r2    = float(r2_score(y_test, ens_preds))
    ens_cv    = float(cross_val_score(ensemble, X_train_sk, y_train,
                                      cv=5, scoring="r2", n_jobs=-1).mean())
    results["Voting Ensemble (HGB+XGB+RF)"] = {
        "RMSE": ens_rmse, "MAE": ens_mae, "R2": ens_r2,
        "CV_R2": ens_cv, "preds": ens_preds, "model": ensemble
    }
    print(f"  {'Voting Ensemble (HGB+XGB+RF)':<32} {ens_rmse:>7.4f} {ens_mae:>7.4f} {ens_r2:>8.4f} {ens_cv:>8.4f}")

    # ── Best model ─────────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]["R2"])
    best_info  = results[best_name]
    best_model = best_info["model"]
    best_preds = best_info["preds"]
    print(f"\n  Best model: {best_name}  (R2 = {best_info['R2']:.4f})")

    errors = np.abs(np.array(y_test) - best_preds)
    print(f"\n  Within +/-0.5 stars : {(errors<=0.5).mean()*100:.1f}%")
    print(f"  Within +/-1.0 star  : {(errors<=1.0).mean()*100:.1f}%")
    print(f"  Within +/-2.0 stars : {(errors<=2.0).mean()*100:.1f}%")

    # ── Plots ─────────────────────────────────────────────────
    plot_results(results, y_test, OUTPUT_DIR)

    # Feature importance on best model
    is_catboost_best = best_info.get("is_catboost", False)
    if is_catboost_best:
        plot_feature_importance(best_model, best_name, X_test_cb, y_test, OUTPUT_DIR)
    else:
        plot_feature_importance(best_model, best_name, X_test_sk, y_test, OUTPUT_DIR)

    # ── Metrics CSV ───────────────────────────────────────────
    print("\n[6/6] Saving metrics...")
    save_metrics_csv(results, OUTPUT_DIR)

    # ── Save best model with full metadata ────────────────────
    medians = {col: float(df[col].median()) for col in ["Year", "Duration", "Votes"]}

    save_dict = {
        "model":               best_model,
        "encoder":             encoder,
        "medians":             medians,
        "best_name":           best_name,
        "genre_list":          genre_list,
        "director_counts":     director_counts,
        "actor_counts":        actor_counts,
        "final_features_list": final_features_list,
        "categorical_cols":    CATEGORICAL_COLS,
        "numeric_features":    NUMERIC_FEATURES,
        "cb_cat_features":     cb_cat_features,
        "cb_keep":             cb_keep,
        "is_catboost":         is_catboost_best,
        "exp_features":        exp_features,
        "genre_cols":          genre_cols,
        # For autocomplete in Streamlit
        "unique_directors":    sorted(df["Director"].unique().tolist()),
        "unique_actors":       sorted(list(
            set(df["Actor 1"].unique()) |
            set(df["Actor 2"].unique()) |
            set(df["Actor 3"].unique())
        )),
        "unique_genres":       genre_list,
    }
    joblib.dump(save_dict, MODEL_PATH)
    print(f"  [OK] Best model & pipeline metadata saved -> {MODEL_PATH}")

    print("\n" + "=" * 62)
    print("  Pipeline complete! Output files:")
    print("    01_EDA_Analysis.png")
    print("    02_Model_Results.png")
    print("    03_Feature_Importances.png")
    print("    04_Model_Metrics.csv")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
