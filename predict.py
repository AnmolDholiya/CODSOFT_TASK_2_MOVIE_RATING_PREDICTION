"""
predict.py — Interactive Movie Rating Predictor (v3)
=====================================================
Supports CatBoost + NLP title features.
Usage:
    python predict.py
"""

import os
import sys
import re
import warnings
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.joblib")

# ── Sequel detection pattern (shared with training) ───────────
SEQUEL_PATTERN = re.compile(
    r"(\d+|II|III|IV|VI|VII|VIII|IX|XI|XII|\bpart\b|\breturns?\b|\bagain\b)",
    re.IGNORECASE
)

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
# NLP Title Feature Extraction
# ─────────────────────────────────────────────────────────────
def extract_title_features_single(name: str) -> dict:
    """Extract NLP title features for a single movie name string."""
    name = name.strip() if name else ""

    is_sequel = int(bool(SEQUEL_PATTERN.search(name)))

    title_length = len(name.replace(" ", ""))
    title_word_count = len(name.split())

    words = [w.lower().strip(".,!?\"'()-") for w in name.split()]
    english_word_ratio = (
        sum(1 for w in words if w in ENGLISH_VOCAB) / len(words)
        if words else 0.0
    )

    return {
        "Is_Sequel":          is_sequel,
        "Title_Length":       title_length,
        "Title_Word_Count":   title_word_count,
        "English_Word_Ratio": english_word_ratio,
    }


# ─────────────────────────────────────────────────────────────
# Train & save if model file is missing
# ─────────────────────────────────────────────────────────────
def train_and_save():
    print("\n  No saved model found — training now (~60 seconds)...")
    import movie_rating_model
    movie_rating_model.main()


def load_or_train():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    return joblib.load(MODEL_PATH)


# ─────────────────────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────────────────────
def _ask(prompt, default=None, cast=None):
    hint = f" [{default}]" if default is not None else ""
    raw  = input(f"  {prompt}{hint}: ").strip()
    if raw == "" and default is not None:
        return default
    if cast:
        try:
            return cast(raw)
        except ValueError:
            print(f"    -> Invalid, using default: {default}")
            return default
    return raw if raw else default


def collect_inputs(medians):
    print("\n" + "-" * 56)
    print("  Enter movie details  (press Enter to use defaults)")
    print("-" * 56)
    name     = _ask("Movie title         (e.g. Dhoom 2)",        "Unknown Movie")
    year     = _ask("Release year        (e.g. 2019)",            int(medians["Year"]),     int)
    duration = _ask("Duration in minutes (e.g. 120)",             int(medians["Duration"]), int)
    votes    = _ask("Number of votes     (e.g. 5000)",            int(medians["Votes"]),    int)
    genre    = _ask("Genre               (e.g. Drama, Romance)",  "Drama")
    director = _ask("Director name",                              "Unknown")
    actor1   = _ask("Actor 1",                                    "Unknown")
    actor2   = _ask("Actor 2  (optional)",                        "Unknown")
    actor3   = _ask("Actor 3  (optional)",                        "Unknown")

    return {
        "Name":     name,
        "Year":     year,
        "Duration": duration,
        "Votes":    votes,
        "Genre":    genre,
        "Director": director,
        "Actor 1":  actor1,
        "Actor 2":  actor2,
        "Actor 3":  actor3,
    }


# ─────────────────────────────────────────────────────────────
# Build Feature Row
# ─────────────────────────────────────────────────────────────
def build_feature_row(bundle, raw_inputs):
    """
    Convert raw inputs dictionary into the exact feature DataFrame
    expected by whichever model (CatBoost or sklearn ensemble) is saved.
    """
    model            = bundle["model"]
    encoder          = bundle["encoder"]
    genre_list       = bundle["genre_list"]
    director_counts  = bundle["director_counts"]
    actor_counts     = bundle["actor_counts"]
    categorical_cols = bundle["categorical_cols"]
    numeric_features = bundle["numeric_features"]
    exp_features     = bundle["exp_features"]
    genre_cols       = bundle["genre_cols"]
    is_catboost      = bundle.get("is_catboost", False)

    name     = raw_inputs.get("Name", "")
    year     = raw_inputs["Year"]
    duration = raw_inputs["Duration"]
    votes    = raw_inputs["Votes"]
    genre    = raw_inputs["Genre"]
    director = raw_inputs["Director"]
    actor1   = raw_inputs["Actor 1"]
    actor2   = raw_inputs["Actor 2"]
    actor3   = raw_inputs["Actor 3"]

    # ── NLP title features ──────────────────────────────────
    nlp = extract_title_features_single(name)

    # ── Base numeric features ───────────────────────────────
    row = {
        "Year":               float(year),
        "Duration":           float(duration),
        "Log_Votes":          np.log1p(float(votes)),
        "Decade":             float((year // 10) * 10),
        "Genre_Count":        float(len([g.strip() for g in genre.split(",") if g.strip()])),
        "Actor_Count":        float(sum([a != "Unknown" for a in [actor1, actor2, actor3]])),
        "Is_Sequel":          float(nlp["Is_Sequel"]),
        "Title_Length":       float(nlp["Title_Length"]),
        "Title_Word_Count":   float(nlp["Title_Word_Count"]),
        "English_Word_Ratio": float(nlp["English_Word_Ratio"]),
        # Categorical (as strings for CatBoost compatibility)
        "Director": str(director),
        "Actor 1":  str(actor1),
        "Actor 2":  str(actor2),
        "Actor 3":  str(actor3),
        "Genre":    genre,
    }

    # ── Experience counts ───────────────────────────────────
    row["Director_Movie_Count"]  = float(director_counts.get(director, 0))
    row["Actor1_Movie_Count"]    = float(actor_counts.get(actor1, 0))
    row["Actor2_Movie_Count"]    = float(actor_counts.get(actor2, 0))
    row["Actor3_Movie_Count"]    = float(actor_counts.get(actor3, 0))
    row["Cast_Movie_Count_Mean"] = np.mean([
        row["Actor1_Movie_Count"],
        row["Actor2_Movie_Count"],
        row["Actor3_Movie_Count"],
    ])

    # ── Genre one-hot ────────────────────────────────────────
    input_genres = {g.strip() for g in genre.split(",") if g.strip()}
    for g in genre_list:
        row[f"Genre_{g}"] = float(1 if g in input_genres else 0)

    df_row = pd.DataFrame([row])

    # ── CatBoost path (raw categoricals, no target enc) ──────
    if is_catboost:
        cb_keep = bundle["cb_keep"]
        for col in categorical_cols:
            df_row[col] = df_row[col].astype(str)
        return df_row[cb_keep]

    # ── sklearn path (target-encode categoricals) ─────────────
    enc_cols = [f"{c}_encoded" for c in categorical_cols]
    cat_enc  = pd.DataFrame(
        encoder.transform(df_row[categorical_cols]),
        columns=enc_cols, index=df_row.index
    )

    keep_num = numeric_features + genre_cols + exp_features
    df_final = pd.concat([df_row[keep_num], cat_enc], axis=1)

    # Cast aggregate encodings
    cast_cols = ["Actor 1_encoded", "Actor 2_encoded", "Actor 3_encoded"]
    df_final["Cast_Rating_Mean"] = df_final[cast_cols].mean(axis=1)
    df_final["Cast_Rating_Max"]  = df_final[cast_cols].max(axis=1)
    df_final["Cast_Rating_Min"]  = df_final[cast_cols].min(axis=1)

    # Align columns to training order exactly
    final_features_list = bundle["final_features_list"]
    df_final = df_final[final_features_list]
    return df_final


# ─────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────
def predict(bundle, raw_inputs):
    df_final = build_feature_row(bundle, raw_inputs)
    pred_val = bundle["model"].predict(df_final)[0]
    return float(np.clip(pred_val, 1.0, 10.0))


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 56)
    print("  Movie Rating Predictor — IMDb India  (v3)")
    print("=" * 56)

    bundle    = load_or_train()
    medians   = bundle["medians"]
    best_name = bundle["best_name"]
    nlp_flag  = "✓ NLP title features active"
    cb_flag   = "✓ CatBoost" if bundle.get("is_catboost") else "✓ sklearn ensemble"
    print(f"  Model ready! [{cb_flag}] [{nlp_flag}]")
    print(f"  Powered by: {best_name}")

    while True:
        raw_inputs = collect_inputs(medians)
        nlp        = extract_title_features_single(raw_inputs["Name"])
        rating     = predict(bundle, raw_inputs)

        stars_filled = int(round(rating / 2))
        star_str     = "★" * stars_filled + "☆" * (5 - stars_filled)

        print("\n" + "=" * 56)
        print(f"  Movie  : {raw_inputs['Name']}")
        print(f"  Sequel : {'Yes' if nlp['Is_Sequel'] else 'No'}")
        print(f"  English word ratio: {nlp['English_Word_Ratio']:.2f}")
        print(f"  Predicted IMDb Rating: {rating:.2f} / 10")
        print(f"  {star_str}")
        print("=" * 56)

        again = input("\n  Predict another movie? (y/n) [y]: ").strip().lower()
        if again in ("n", "no"):
            print("\n  Goodbye!\n")
            break


if __name__ == "__main__":
    main()
