"""
Skill Exchange & Service Evaluation — Streamlit app
Ridge Regression price estimator based on ridge_base.ipynb.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Place fiverr.csv in the same folder as this file, or upload it in the app.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

RANDOM_STATE = 42
DEFAULT_DATA = Path(__file__).with_name("fiverr.csv")

st.set_page_config(page_title="Skill Exchange Value Estimator", page_icon="🔁", layout="wide")

st.title("🔁 Skill Exchange & Service Evaluation")
st.caption("Ridge Regression base model · Estimates service listing prices to help compare barter offers.")
st.info(
    "This model was trained on Fiverr listing starting prices in euros. "
    "Use the result as a rough reference, not a guaranteed market price. "
    "It has not been calibrated to Kenyan service prices."
)

def parse_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Fiverr columns using the notebook's cleaning logic."""
    required = {"Category", "Subcat", "name", "stars", "price"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    df = raw_df.copy()
    df = df.drop(columns=[c for c in ["Category-href", "Subcat-href"] if c in df.columns])

    price_text = df["price"].astype("string")
    price_match = price_text.str.extract(r"€\s*([\d,]+\.?\d*)", expand=False)
    df["price_eur"] = pd.to_numeric(
        price_match.str.replace(",", "", regex=False), errors="coerce"
    )

    stars_text = df["stars"].astype("string")
    parts = stars_text.str.extract(r"^([\d.]+)\(([\d.]+)(k?)\+?\)$")
    df["rating"] = pd.to_numeric(parts[0], errors="coerce")
    review_count = pd.to_numeric(parts[1], errors="coerce")
    df["reviews"] = review_count * np.where(parts[2].fillna("").eq("k").to_numpy(dtype=bool), 1000, 1)
    df["reviews"] = df["reviews"].fillna(0)
    df["has_reviews"] = raw_df["stars"].notna().astype(int)

    df["Category"] = df["Category"].astype("string").fillna("").str.strip()
    df["Subcat"] = (
        df["Subcat"].astype("string").fillna("")
        .str.replace(r"(?<=[a-z])(New|Beta)$", "", regex=True)
        .str.strip()
    )
    df["service"] = (
        df["name"].fillna("").astype(str).str.lower()
        .str.replace(r"^i will\s+", "", regex=True).str.strip()
    )
    df = df.drop(columns=[c for c in ["name", "stars", "price"] if c in df.columns])
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=["price_eur", "Category", "Subcat", "service"])
    df["log_price"] = np.log1p(df["price_eur"])
    df["log_reviews"] = np.log1p(df["reviews"])
    return df

def make_preprocessor():
    numeric_cols = ["rating", "log_reviews", "has_reviews"]
    categorical_cols = ["Category", "Subcat"]
    text_col = "service"
    return ColumnTransformer([
        ("num", Pipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("txt", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=3000, stop_words="english"
        ), text_col),
    ])

@st.cache_resource(show_spinner="Training the Ridge Regression model…")
def train_model(csv_bytes: bytes):
    import io
    raw = pd.read_csv(io.BytesIO(csv_bytes))
    clean = parse_dataset(raw)
    feature_cols = ["Category", "Subcat", "service", "rating", "log_reviews", "has_reviews"]
    X = clean[feature_cols]
    y = clean["log_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    pipe = Pipeline([("prep", make_preprocessor()), ("model", Ridge())])
    params = {"model__alpha": [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipe, params, cv=cv, scoring="neg_root_mean_squared_error",
        n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    pred_log = best_model.predict(X_test)
    actual_eur = np.expm1(y_test)
    pred_eur = np.maximum(0, np.expm1(pred_log))
    metrics = {
        "Best alpha": float(search.best_params_["model__alpha"]),
        "CV RMSE (log)": float(-search.best_score_),
        "Test RMSE (log)": float(np.sqrt(mean_squared_error(y_test, pred_log))),
        "R²": float(r2_score(y_test, pred_log)),
        "MAE (€)": float(mean_absolute_error(actual_eur, pred_eur)),
        "Median error (€)": float(median_absolute_error(actual_eur, pred_eur)),
        "Rows used": int(len(clean)),
        "Train rows": int(len(X_train)),
        "Test rows": int(len(X_test)),
    }
    return best_model, clean, metrics

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload fiverr.csv", type=["csv"])
    st.caption("If no file is uploaded, the app looks for fiverr.csv beside app.py.")
    st.divider()
    st.markdown("**Model features**")
    st.write("Category, sub-category, service title, rating and review count.")

if uploaded is not None:
    csv_bytes = uploaded.getvalue()
elif DEFAULT_DATA.exists():
    csv_bytes = DEFAULT_DATA.read_bytes()
else:
    csv_bytes = None

if csv_bytes is None:
    st.warning("Upload your `fiverr.csv` dataset in the sidebar to train and use the model.")
    st.markdown("Expected columns: `Category`, `Subcat`, `name`, `stars`, and `price`.")
    st.stop()

try:
    with st.spinner("Preparing data and fitting the model. The first run can take a little while…"):
        model, clean, metrics = train_model(csv_bytes)
except Exception as exc:
    st.error(f"Could not prepare/train the model: {exc}")
    st.markdown("Check that the uploaded file is the Fiverr CSV used by the notebook and contains the expected columns.")
    st.stop()

tab_predict, tab_compare, tab_data = st.tabs(["💡 Estimate a service", "⚖️ Compare a swap", "📊 Dataset & model"])

def service_form(prefix: str, default_category: str, default_subcat: str):
    categories = sorted(clean["Category"].dropna().astype(str).unique().tolist())
    if default_category not in categories:
        default_category = categories[0]
    category = st.selectbox("Service category", categories, index=categories.index(default_category), key=f"{prefix}_cat")
    subs = sorted(clean.loc[clean["Category"].astype(str) == category, "Subcat"].dropna().astype(str).unique().tolist())
    if not subs:
        subs = sorted(clean["Subcat"].dropna().astype(str).unique().tolist())
    subcat = st.selectbox("Sub-category", subs, index=subs.index(default_subcat) if default_subcat in subs else 0, key=f"{prefix}_sub")
    title = st.text_input("Describe the service", placeholder="e.g. build a responsive business website", key=f"{prefix}_title")
    c1, c2 = st.columns(2)
    with c1:
        rating = st.slider("Seller rating", min_value=0.0, max_value=5.0, value=4.8, step=0.1, key=f"{prefix}_rating")
    with c2:
        reviews = st.number_input("Number of reviews", min_value=0, max_value=1000000, value=10, step=1, key=f"{prefix}_reviews")
    return category, subcat, title.strip(), rating, int(reviews)

def estimate(category, subcat, title, rating, reviews):
    service = re.sub(r"^i will\s+", "", title.lower()).strip()
    row = pd.DataFrame([{
        "Category": category,
        "Subcat": subcat,
        "service": service,
        "rating": float(rating) if reviews > 0 else np.nan,
        "log_reviews": np.log1p(reviews),
        "has_reviews": int(reviews > 0),
    }])
    prediction = float(np.expm1(model.predict(row)[0]))
    return max(0.0, prediction)

with tab_predict:
    st.subheader("Estimate the value of one service")
    st.write("Enter the service details to estimate its listing starting price.")
    with st.form("single_estimate_form"):
        # Form widgets are collected after submit. Use dynamic sub-category lists from the dataset.
        categories = sorted(clean["Category"].dropna().astype(str).unique().tolist())
        category = st.selectbox("Service category", categories, key="single_cat")
        subcats = sorted(clean.loc[clean["Category"].astype(str) == category, "Subcat"].dropna().astype(str).unique().tolist())
        subcat = st.selectbox("Sub-category", subcats or sorted(clean["Subcat"].unique()), key="single_sub")
        title = st.text_input("Describe the service", placeholder="e.g. design a logo for a small business", key="single_title")
        c1, c2 = st.columns(2)
        with c1:
            rating = st.slider("Seller rating", 0.0, 5.0, 4.8, 0.1, key="single_rating")
        with c2:
            reviews = st.number_input("Number of reviews", min_value=0, max_value=1000000, value=10, step=1, key="single_reviews")
        submitted = st.form_submit_button("Estimate service value", type="primary")
    if submitted:
        if not title.strip():
            st.warning("Enter a short description of the service first.")
        else:
            value = estimate(category, subcat, title, rating, int(reviews))
            st.metric("Estimated starting price", f"€{value:,.2f}")
            st.caption("Prediction is in euros because the model was trained on Fiverr prices.")
            st.write("This is a model estimate, not a verified quote or a direct conversion to Kenyan shillings.")

with tab_compare:
    st.subheader("Compare two services for a barter")
    st.write("The difference estimates how much extra value may be needed to balance the exchange.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Service A")
        with st.form("service_a_form"):
            cats = sorted(clean["Category"].dropna().astype(str).unique().tolist())
            a_cat = st.selectbox("Category", cats, key="a_cat")
            a_subs = sorted(clean.loc[clean["Category"].astype(str) == a_cat, "Subcat"].dropna().astype(str).unique().tolist())
            a_sub = st.selectbox("Sub-category", a_subs or sorted(clean["Subcat"].unique()), key="a_sub")
            a_title = st.text_input("Service description", placeholder="e.g. tutor Python basics", key="a_title")
            a_rating = st.slider("Rating", 0.0, 5.0, 4.8, 0.1, key="a_rating")
            a_reviews = st.number_input("Reviews", 0, 1000000, 10, 1, key="a_reviews")
            submit_a = st.form_submit_button("Save Service A")
    with right:
        st.markdown("#### Service B")
        with st.form("service_b_form"):
            cats = sorted(clean["Category"].dropna().astype(str).unique().tolist())
            b_cat = st.selectbox("Category", cats, key="b_cat")
            b_subs = sorted(clean.loc[clean["Category"].astype(str) == b_cat, "Subcat"].dropna().astype(str).unique().tolist())
            b_sub = st.selectbox("Sub-category", b_subs or sorted(clean["Subcat"].unique()), key="b_sub")
            b_title = st.text_input("Service description", placeholder="e.g. create a social media plan", key="b_title")
            b_rating = st.slider("Rating", 0.0, 5.0, 4.8, 0.1, key="b_rating")
            b_reviews = st.number_input("Reviews", 0, 1000000, 10, 1, key="b_reviews")
            submit_b = st.form_submit_button("Save Service B")
    # Persist submitted form values in session state so either form can be saved independently.
    if submit_a:
        st.session_state["saved_a"] = (a_cat, a_sub, a_title, a_rating, int(a_reviews))
    if submit_b:
        st.session_state["saved_b"] = (b_cat, b_sub, b_title, b_rating, int(b_reviews))
    if "saved_a" in st.session_state and "saved_b" in st.session_state:
        sa = st.session_state["saved_a"]
        sb = st.session_state["saved_b"]
        if sa[2].strip() and sb[2].strip():
            va = estimate(*sa)
            vb = estimate(*sb)
            ca, cb, cc = st.columns(3)
            ca.metric("Service A estimate", f"€{va:,.2f}")
            cb.metric("Service B estimate", f"€{vb:,.2f}")
            gap = abs(va - vb)
            cc.metric("Estimated value gap", f"€{gap:,.2f}")
            if abs(va - vb) < 0.01:
                st.success("The estimated values are effectively equal.")
            elif va > vb:
                st.info(f"Service B is estimated to be worth about €{gap:,.2f} less. The receiver of Service A could offer a top-up or extra service of similar estimated value.")
            else:
                st.info(f"Service A is estimated to be worth about €{gap:,.2f} less. The receiver of Service B could offer a top-up or extra service of similar estimated value.")
            st.caption("This comparison uses predicted listing starting prices, not the time, effort, quality, or locally agreed value of either service.")
        else:
            st.warning("Save descriptions for both services to calculate the comparison.")

with tab_data:
    st.subheader("Model performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows used", f"{metrics['Rows used']:,}")
    m2.metric("Selected alpha", f"{metrics['Best alpha']:g}")
    m3.metric("Test R²", f"{metrics['R²']:.3f}")
    m4.metric("Mean absolute error", f"€{metrics['MAE (€)']:.2f}")

    metrics_df = pd.DataFrame([
        {"Metric": "5-fold CV RMSE (log)", "Value": metrics["CV RMSE (log)"]},
        {"Metric": "Test RMSE (log)", "Value": metrics["Test RMSE (log)"]},
        {"Metric": "Test R²", "Value": metrics["R²"]},
        {"Metric": "Mean absolute error (€)", "Value": metrics["MAE (€)"]},
        {"Metric": "Median absolute error (€)", "Value": metrics["Median error (€)"]},
    ])
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    st.caption("R² describes how much price variation is explained on the held-out test set. Lower error is better; R² near 0 means little explanatory power.")

    st.subheader("Dataset preview")
    st.write(f"Cleaned dataset: {len(clean):,} rows. The app removes duplicates and parses starting prices, ratings and review counts.")
    st.dataframe(clean[["Category", "Subcat", "service", "rating", "reviews", "price_eur"]].head(25), use_container_width=True)
    st.subheader("Average starting price by category")
    chart_data = clean.groupby("Category", as_index=False)["price_eur"].median().sort_values("price_eur")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(chart_data["Category"], chart_data["price_eur"])
    ax.set_xlabel("Median starting price (€)")
    ax.set_ylabel("")
    ax.set_title("Median listing starting price by category")
    fig.tight_layout()
    st.pyplot(fig)
    st.download_button(
        "Download cleaned dataset",
        data=clean.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_fiverr_services.csv",
        mime="text/csv",
    )

with st.expander("How this model works and its limitations"):
    st.markdown("""
    - **Algorithm:** Ridge Regression with regularisation; `alpha` is selected using 5-fold cross-validation.
    - **Inputs:** service category, sub-category, title text, seller rating, review count and whether reviews exist.
    - **Text handling:** TF-IDF features from one- and two-word phrases in the service title.
    - **Target:** `log(1 + starting price)`; predictions are converted back to euros.
    - **Limitations:** the notebook's test R² is around 0.34, so many price differences remain unexplained. Prices reflect Fiverr's starting packages, not necessarily a full project's cost or Kenyan market value. Hours, expertise, complexity and local demand are not directly measured.
    """)
