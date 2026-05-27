import os
import sys
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

st.set_page_config(
    page_title="Indian Movie Rating Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #1e293b; border-radius: 8px 8px 0 0;
    color: #94a3b8; padding: 10px 20px; font-weight: 600;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
}

div[data-testid="stMetric"] {
    background: #1e293b; border-radius: 10px;
    padding: 14px 16px; border: 1px solid #334155;
}

.rating-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 16px; padding: 28px;
    border-left: 6px solid #f59e0b;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    text-align: center; color: white;
}
.nlp-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 0.78rem;
    font-weight: 600; margin: 3px;
}
.badge-yes  { background: #065f46; color: #6ee7b7; border: 1px solid #059669; }
.badge-no   { background: #1e3a5f; color: #93c5fd; border: 1px solid #3b82f6; }
.badge-info { background: #3f2c5e; color: #c4b5fd; border: 1px solid #7c3aed; }
</style>
""", unsafe_allow_html=True)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "IMDb Movies India.csv")
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.joblib")

sys.path.insert(0, BASE_DIR)
from predict import predict, extract_title_features_single


@st.cache_resource
def load_bundle_cached():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None


# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding-bottom:16px;">
    <h1 style="margin:0; font-size:2rem; color:#f59e0b;">🎬 IMDb India</h1>
    <p style="color:#94a3b8; margin:4px 0 0;">Rating Prediction Engine</p>
    <p style="color:#64748b; font-size:0.78rem;">v3 · CatBoost + NLP Features</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

bundle = load_bundle_cached()

if bundle:
    st.sidebar.success("✅ Model Loaded & Ready")
    st.sidebar.markdown(f"**Best model:** {bundle.get('best_name', '—')}")
    cb_active = "✅ CatBoost" if bundle.get("is_catboost") else "⬜ sklearn Ensemble"
    st.sidebar.markdown(f"**Architecture:** {cb_active}")
    st.sidebar.markdown(f"**Genres tracked:** {len(bundle.get('genre_list', []))}")
    st.sidebar.markdown(f"**Directors:** {len(bundle.get('unique_directors', []))}")
    st.sidebar.markdown(f"**Actors:** {len(bundle.get('unique_actors', []))}")
else:
    st.sidebar.error("⚠️ No saved model found")

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Train / Re-train")
if st.sidebar.button("Run Training Pipeline", use_container_width=True):
    with st.spinner("Training all models... (~60–90 seconds)"):
        import movie_rating_model
        try:
            movie_rating_model.main()
            st.sidebar.success("Done! Reloading…")
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.82rem; color:#64748b; line-height:1.7;">
<b>Features used:</b><br/>
• Release Year & Decade<br/>
• Duration · Log(Votes)<br/>
• 23 Genre binary flags<br/>
• Director & cast target encodings<br/>
• Experience counts (movies in DB)<br/>
• <b>NLP:</b> Is_Sequel, Title_Length,<br/>
  &nbsp;&nbsp;Word_Count, English_Word_Ratio
</div>
""", unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────
st.title("🎬 Movie Rating Prediction")
st.markdown("Estimate a film's **IMDb rating** using metadata, cast, and movie title analysis.")

if bundle is None:
    st.warning("👈 Click **Run Training Pipeline** in the sidebar to train the model first.")
    st.stop()

genre_list       = bundle["genre_list"]
unique_directors = bundle["unique_directors"]
unique_actors    = bundle["unique_actors"]
medians          = bundle["medians"]

tab1, tab2, tab3 = st.tabs(["🔮  Predict Rating", "📊  Data Insights (EDA)", "📈  Model Evaluation"])


# ═══════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Movie Details")

    left_col, right_col = st.columns([2, 1.3], gap="large")

    with left_col:
        with st.container(border=True):
            # ── Movie Title ────────────────────────────────
            movie_name = st.text_input(
                "🎬 Movie Title",
                value="My Movie",
                placeholder="e.g. Dhoom 2, Krrish 3, URI: The Surgical Strike",
                help="The title is analyzed for sequel indicators, word count, language style etc."
            )

            st.markdown("---")
            # ── Year / Duration / Votes ────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                year = st.slider("Release Year", 1920, 2026, int(medians["Year"]), 1)
            with c2:
                duration = st.number_input("Duration (min)", 10, 400, int(medians["Duration"]), 5)
            with c3:
                votes = st.number_input("Votes", 1, 1_000_000, int(medians["Votes"]), 50)

            # ── Genre ──────────────────────────────────────
            genres = st.multiselect(
                "Genres", genre_list, default=["Drama"],
                help="Select all applicable genres."
            )
            genre_str = ", ".join(genres) if genres else "Unknown"

            # ── Director ───────────────────────────────────
            dir_opts = ["Enter Custom Name..."] + unique_directors
            sel_dir  = st.selectbox("Director", dir_opts,
                                    index=dir_opts.index("Unknown") if "Unknown" in dir_opts else 0)
            director = st.text_input("Custom Director", "Unknown") if sel_dir == "Enter Custom Name..." else sel_dir

            # ── Cast ───────────────────────────────────────
            st.markdown("**Cast Members**")
            ac1, ac2, ac3 = st.columns(3)
            actor_opts = ["Enter Custom Name..."] + unique_actors
            def_idx    = actor_opts.index("Unknown") if "Unknown" in actor_opts else 0

            with ac1:
                s1 = st.selectbox("Actor 1 (Lead)", actor_opts, index=def_idx, key="a1")
                actor1 = st.text_input("Custom Actor 1", "Unknown", key="a1c") if s1 == "Enter Custom Name..." else s1
            with ac2:
                s2 = st.selectbox("Actor 2", actor_opts, index=def_idx, key="a2")
                actor2 = st.text_input("Custom Actor 2", "Unknown", key="a2c") if s2 == "Enter Custom Name..." else s2
            with ac3:
                s3 = st.selectbox("Actor 3", actor_opts, index=def_idx, key="a3")
                actor3 = st.text_input("Custom Actor 3", "Unknown", key="a3c") if s3 == "Enter Custom Name..." else s3

    # ── Build raw inputs dict & predict ────────────────────────
    raw_inputs = {
        "Name":     movie_name,
        "Year":     year,
        "Duration": duration,
        "Votes":    votes,
        "Genre":    genre_str,
        "Director": director,
        "Actor 1":  actor1,
        "Actor 2":  actor2,
        "Actor 3":  actor3,
    }

    nlp    = extract_title_features_single(movie_name)
    rating = predict(bundle, raw_inputs)

    stars_filled = int(round(rating / 2))
    star_str     = "★" * stars_filled + "☆" * (5 - stars_filled)

    # Colour the rating badge
    if rating >= 7.5:
        badge_col, badge_label = "#065f46", "🟢 High"
    elif rating >= 5.5:
        badge_col, badge_label = "#78350f", "🟡 Average"
    else:
        badge_col, badge_label = "#7f1d1d", "🔴 Low"

    sequel_badge = (
        '<span class="nlp-badge badge-yes">🔢 Sequel / Series</span>'
        if nlp["Is_Sequel"]
        else '<span class="nlp-badge badge-no">🎬 Original Title</span>'
    )
    lang_pct = nlp["English_Word_Ratio"] * 100
    if lang_pct >= 70:
        lang_label = "English"
    elif lang_pct >= 30:
        lang_label = "Mixed"
    else:
        lang_label = "Non-English"
    lang_badge = f'<span class="nlp-badge badge-info">🌐 {lang_label} ({lang_pct:.0f}% EN words)</span>'

    with right_col:
        st.markdown(f"""
        <div class="rating-card">
            <p style="margin:0;color:#94a3b8;font-size:.85rem;text-transform:uppercase;letter-spacing:1px;">
                Predicted IMDb Rating
            </p>
            <h1 style="font-size:4.5rem;margin:8px 0;color:#f59e0b;line-height:1;font-weight:800;">
                {rating:.2f}
                <span style="font-size:1.4rem;color:#64748b;font-weight:400;">/ 10</span>
            </h1>
            <div style="font-size:2rem;color:#f59e0b;letter-spacing:3px;margin-bottom:10px;">
                {star_str}
            </div>
            <div style="display:inline-block;padding:4px 16px;border-radius:20px;
                        background:{badge_col};font-size:.9rem;font-weight:600;margin-bottom:14px;">
                {badge_label}
            </div>
            <hr style="border-color:#334155;margin:14px 0;"/>
            <div style="text-align:left;">
                <p style="margin:0 0 8px;font-size:.88rem;color:#94a3b8;font-weight:600;">
                    🧠 NLP Title Analysis
                </p>
                {sequel_badge}
                {lang_badge}
                <br/>
                <span class="nlp-badge badge-info">📏 {nlp['Title_Word_Count']} words · {nlp['Title_Length']} chars</span>
            </div>
            <hr style="border-color:#334155;margin:14px 0;"/>
            <div style="text-align:left;font-size:.85rem;color:#cbd5e1;line-height:2;">
                🎬 <b>{movie_name}</b><br/>
                📅 {year} &nbsp;·&nbsp; ⏱ {duration} min &nbsp;·&nbsp; 👥 {votes:,} votes<br/>
                🎭 {genre_str}<br/>
                🎥 {director}<br/>
                ⭐ {", ".join(a for a in [actor1,actor2,actor3] if a!="Unknown") or "—"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Experience box
        dir_exp  = bundle["director_counts"].get(director, 0)
        act1_exp = bundle["actor_counts"].get(actor1, 0)
        act2_exp = bundle["actor_counts"].get(actor2, 0)
        act3_exp = bundle["actor_counts"].get(actor3, 0)
        st.markdown(f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;
                    padding:14px 16px;margin-top:14px;font-size:.85rem;color:#94a3b8;">
            📚 <b>Dataset Experience</b><br/>
            Director <i>{director}</i>: <b>{dir_exp}</b> movies<br/>
            {actor1}: <b>{act1_exp}</b> &nbsp;|&nbsp;
            {actor2}: <b>{act2_exp}</b> &nbsp;|&nbsp;
            {actor3}: <b>{act3_exp}</b>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    st.markdown("""
    This section explores the historical patterns in the IMDb India dataset. 
    Understanding these trends helps explain how features like **Genres**, **Votes**, and **Release Year** influence the model's predictions.
    """)

    if os.path.exists(CSV_PATH):
        try:
            df_raw = pd.read_csv(CSV_PATH, encoding="latin-1").dropna(subset=["Rating"])
            
            # 1. Key Metrics
            st.markdown("### 📌 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Movies", f"{len(df_raw):,}")
            col2.metric("Average Rating", f"{df_raw['Rating'].mean():.1f} / 10")
            
            # Clean Year to find oldest/newest
            valid_years = df_raw['Year'].astype(str).str.extract(r'(\d{4})').dropna()[0].astype(int)
            if not valid_years.empty:
                col3.metric("Oldest Movie", f"{valid_years.min()}")
                col4.metric("Newest Movie", f"{valid_years.max()}")
            
            st.markdown("---")
            
            # 2. Interactive Charts
            st.markdown("### 📈 Key Distributions")
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**Rating Distribution**")
                st.caption("Observe that most Indian movies fall between a 5.0 and 7.5 rating.")
                # Create a simple histogram DataFrame for st.bar_chart
                hist_values, bin_edges = np.histogram(df_raw['Rating'], bins=15, range=(1,10))
                bin_labels = [f"{round(bin_edges[i], 1)}" for i in range(len(hist_values))]
                hist_df = pd.DataFrame({'Number of Movies': hist_values}, index=bin_labels)
                st.bar_chart(hist_df, height=350)
                
            with c2:
                st.markdown("**Top 10 Genres**")
                st.caption("Drama, Action, and Romance dominate the Indian movie industry.")
                # Explode genres and count
                genres = df_raw['Genre'].dropna().str.split(',').explode().str.strip()
                top_genres = genres.value_counts().head(10)
                st.bar_chart(top_genres, height=350)

            st.markdown("---")
            
            # 3. Raw Data
            with st.expander("🔍 View Raw Dataset Sample", expanded=False):
                st.caption(f"Showing 10 random movies out of **{df_raw.shape[0]:,}** valid records.")
                st.dataframe(df_raw.sample(min(10, len(df_raw)), random_state=42), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading data insights: {e}")

    # 3. Votes vs. Rating Scatter Plot
    st.markdown("### 🎬 Votes vs. Rating")
    st.caption("Do movies with more votes tend to have higher ratings? (Showing a sample of movies)")
    if os.path.exists(CSV_PATH):
        try:
            if 'Votes' in df_raw.columns:
                votes_clean = pd.to_numeric(df_raw['Votes'].astype(str).str.replace(',', ''), errors='coerce')
                scatter_df = pd.DataFrame({'Rating': df_raw['Rating'], 'Votes': votes_clean}).dropna()
                
                # Sample the data if it's very large for performance
                if len(scatter_df) > 1500:
                    scatter_df = scatter_df.sample(1500, random_state=42)
                    
                st.scatter_chart(scatter_df, x='Rating', y='Votes', height=350)
        except Exception as e:
            st.error(f"Error displaying scatter chart: {e}")


# ═══════════════════════════════════════════════════════════
# TAB 3 — MODEL EVALUATION
# ═══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Regression Model Comparison")
    st.caption("All models evaluated on a held-out 20% test set. Cross-validated R² (5-fold) is also shown.")

    metrics_path = os.path.join(BASE_DIR, "04_Model_Metrics.csv")
    if os.path.exists(metrics_path):
        mdf = pd.read_csv(metrics_path)
        # Highlight best R2 row
        best_r2_idx = mdf["R2"].idxmax()
        st.dataframe(
            mdf.style.highlight_max(subset=["R2","CV_R2"], color="#065f46")
                     .highlight_min(subset=["RMSE","MAE"],   color="#065f46"),
            use_container_width=True
        )
    else:
        st.warning("Metrics CSV not found. Run the training pipeline first.")

    st.markdown("---")
    p1, p2 = st.columns(2)
    with p1:
        st.subheader("Performance Metrics")
        img = os.path.join(BASE_DIR, "actual_vs_predicted.png")
        if os.path.exists(img):
            st.image(img, caption="Actual vs Predicted", use_container_width=True)
    with p2:
        st.subheader("Feature Importances")
        img = os.path.join(BASE_DIR, "feature_importances.png")
        if os.path.exists(img):
            st.image(img, caption="Permutation importances — top 20 features", use_container_width=True)
