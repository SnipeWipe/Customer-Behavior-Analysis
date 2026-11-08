"""
RetailPulse — Customer Shopping Intelligence Platform
Run:
    pip install -r requirements.txt
    streamlit run app.py

Data source:
  Upload customer_shopping_behavior.csv (or another compatible CSV file) in the sidebar.

Expected dataset fields (case/spacing differences are normalized):
customer_id, age, gender, item_purchased, category, purchase_amount_(usd),
location, size, color, season, review_rating, subscription_status,
shipping_type, discount_applied, promo_code_used, previous_purchases,
payment_method, frequency_of_purchases
"""

from __future__ import annotations

import re
import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="RetailPulse | Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- THEME / HELPERS -----------------------------

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
        background: rgba(128,128,128,.08);
        border: 1px solid rgba(128,128,128,.18);
        padding: 14px 16px; border-radius: 12px;
      }
      .hero {
        padding: 1.25rem 1.4rem; border-radius: 16px;
        background: linear-gradient(115deg, rgba(40,90,150,.18), rgba(30,160,130,.10));
        border: 1px solid rgba(128,128,128,.2); margin-bottom: 1rem;
      }
      .muted {opacity: .72; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

RENAME_MAP = {
    "purchase_amount_(usd)": "purchase_amount",
    "purchase_amount_usd": "purchase_amount",
    "subscription_status": "subscription_status",
    "review_rating": "review_rating",
    "item_purchased": "item_purchased",
    "previous_purchases": "previous_purchases",
    "frequency_of_purchases": "frequency_of_purchases",
}

EXPECTED_COLUMNS = [
    "customer_id", "age", "gender", "item_purchased", "category",
    "purchase_amount", "location", "size", "color", "season",
    "review_rating", "subscription_status", "shipping_type",
    "discount_applied", "promo_code_used", "previous_purchases",
    "payment_method", "frequency_of_purchases",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower())).strip("_")
        for c in df.columns
    ]
    df = df.rename(columns=RENAME_MAP)
    return df


def parse_bool_series(s: pd.Series) -> pd.Series:
    true_values = {"yes", "true", "1", "y"}
    return s.astype(str).str.strip().str.lower().isin(true_values)


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean data and create transparent, defensible analytical features."""
    df = normalize_columns(raw)
    notes = []

    if "purchase_amount" not in df.columns:
        raise ValueError(
            "Could not find a purchase amount column. Expected "
            "'purchase_amount_(usd)' or 'purchase_amount'."
        )

    # Convert numeric fields safely.
    for col in ["age", "purchase_amount", "review_rating", "previous_purchases"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Median imputation by category, then overall median as fallback.
    if "review_rating" in df.columns:
        missing_before = int(df["review_rating"].isna().sum())
        if missing_before:
            if "category" in df.columns:
                df["review_rating"] = df["review_rating"].fillna(
                    df.groupby("category")["review_rating"].transform("median")
                )
            df["review_rating"] = df["review_rating"].fillna(df["review_rating"].median())
            notes.append(f"Imputed {missing_before:,} missing review ratings using category medians, then the overall median.")

    # Remove exact duplicate records, but do not assume customer_id is unique.
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        df = df.drop_duplicates()
        notes.append(f"Removed {duplicate_count:,} exact duplicate rows.")

    # Create age groups only when age is available.
    if "age" in df.columns and "age_group" not in df.columns:
        labels = ["Young Adult", "Adult", "Middle-Aged", "Senior"]
        try:
            df["age_group"] = pd.qcut(df["age"], q=4, labels=labels, duplicates="drop")
        except ValueError:
            df["age_group"] = pd.cut(
                df["age"], bins=[0, 25, 40, 55, np.inf],
                labels=labels, include_lowest=True
            )

    # Consistent yes/no flags.
    for col in ["discount_applied", "promo_code_used", "subscription_status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Frequency is an approximate cadence, not observed transaction timing.
    if "frequency_of_purchases" in df.columns:
        cadence = {
            "weekly": 7, "fortnightly": 14, "bi-weekly": 14,
            "monthly": 30, "quarterly": 90, "every 3 months": 90,
            "annually": 365,
        }
        df["estimated_purchase_interval_days"] = (
            df["frequency_of_purchases"].astype(str).str.strip().str.lower().map(cadence)
        )

    # Dataset-level quality flags.
    if "discount_applied" in df.columns and "promo_code_used" in df.columns:
        same = (df["discount_applied"].astype(str).str.lower()
                == df["promo_code_used"].astype(str).str.lower()).all()
        if same:
            notes.append(
                "Discount Applied and Promo Code Used are identical in this dataset; "
                "interpret them as one signal rather than independent drivers."
            )

    return df, notes


@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def money(value: float) -> str:
    return f"${value:,.2f}"


def safe_pct(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def section_heading(title: str, subtitle: str = ""):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def require_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        st.info("This analysis needs these columns: " + ", ".join(missing))
        return False
    return True


# ----------------------------- SIDEBAR / DATA ------------------------------

st.sidebar.title("🛍️ RetailPulse")
st.sidebar.caption("Customer Shopping Intelligence")
uploaded = st.sidebar.file_uploader(
    "Upload customer shopping CSV",
    type=["csv"],
    help="Upload your customer_shopping_behavior.csv file or another compatible CSV.",
)

if uploaded is not None:
    try:
        raw_df = load_csv_bytes(uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()
else:
    st.info("Upload your customer shopping CSV file in the sidebar to launch the dashboard.")
    st.stop()

try:
    df, cleaning_notes = prepare_data(raw_df)
except Exception as exc:
    st.error(f"Data preparation failed: {exc}")
    st.stop()

if df.empty:
    st.error("The dataset is empty after loading.")
    st.stop()

st.sidebar.divider()
st.sidebar.caption(f"Loaded rows: {len(df):,}")
st.sidebar.caption(f"Available fields: {len(df.columns)}")
st.sidebar.download_button(
    "Download cleaned dataset",
    data=csv_download(df),
    file_name="retailpulse_cleaned_data.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0;">RetailPulse</h1>
      <p style="margin:.35rem 0 0 0;font-size:1.05rem;">
        Turn customer shopping data into clear, actionable business decisions.
      </p>
      <p class="muted" style="margin:.45rem 0 0 0;">
        Customer intelligence · Product performance · Marketing analytics · Behavioral segmentation
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if cleaning_notes:
    with st.expander("Data quality and preparation log", expanded=False):
        for note in cleaning_notes:
            st.write("• " + note)

# ----------------------------- GLOBAL FILTERS ------------------------------

st.sidebar.subheader("Dashboard filters")
filtered = df.copy()
filter_columns = ["gender", "category", "season", "location", "subscription_status", "payment_method"]
for col in filter_columns:
    if col in filtered.columns:
        choices = sorted(filtered[col].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(col.replace("_", " ").title(), choices, default=choices)
        filtered = filtered[filtered[col].astype(str).isin(selected)]

if "age_group" in filtered.columns:
    age_choices = [str(x) for x in filtered["age_group"].dropna().unique()]
    age_choices = sorted(age_choices)
    selected_ages = st.sidebar.multiselect("Age group", age_choices, default=age_choices)
    filtered = filtered[filtered["age_group"].astype(str).isin(selected_ages)]

st.sidebar.caption(f"Filtered records: {len(filtered):,} / {len(df):,}")
if filtered.empty:
    st.warning("No records match the selected filters. Adjust the sidebar filters.")
    st.stop()

# ----------------------------- NAVIGATION ----------------------------------

pages = [
    "Executive Overview",
    "Customer Intelligence",
    "Product & Sales",
    "Marketing Intelligence",
    "Customer Segmentation",
    "Data Quality & SQL",
]
page = st.sidebar.radio("Explore", pages)

# ----------------------------- EXECUTIVE OVERVIEW --------------------------

if page == "Executive Overview":
    section_heading("Executive Overview", "A decision-focused snapshot of the selected customer records.")

    total_records = len(filtered)
    avg_purchase = filtered["purchase_amount"].mean()
    median_purchase = filtered["purchase_amount"].median()
    total_purchase_amount = filtered["purchase_amount"].sum()
    avg_rating = filtered["review_rating"].mean() if "review_rating" in filtered else np.nan
    subscription_rate = (
        safe_pct((filtered["subscription_status"].astype(str).str.lower() == "yes").sum(), total_records)
        if "subscription_status" in filtered else np.nan
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Customer records", f"{total_records:,}")
    m2.metric("Avg. purchase amount", money(avg_purchase))
    m3.metric("Total purchase amount*", money(total_purchase_amount))
    m4.metric("Average rating", f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "N/A")
    m5.metric("Subscription rate", f"{subscription_rate:.1f}%" if pd.notna(subscription_rate) else "N/A")

    st.caption("*Sum of purchase_amount across dataset records. It is not a time-based revenue measure.")

    left, right = st.columns(2)
    with left:
        if "category" in filtered:
            cat = filtered.groupby("category", dropna=False)["purchase_amount"].agg(
                records="count", avg_purchase="mean"
            ).reset_index().sort_values("records", ascending=False)
            fig = px.bar(cat, x="category", y="records", color="avg_purchase",
                         title="Customer records by product category",
                         labels={"records": "Records", "category": "Category", "avg_purchase": "Average purchase ($)"})
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if "age_group" in filtered:
            age = filtered.groupby("age_group", observed=False)["purchase_amount"].mean().reset_index()
            fig = px.bar(age, x="age_group", y="purchase_amount",
                         title="Average purchase amount by age group",
                         labels={"age_group": "Age group", "purchase_amount": "Average purchase ($)"})
            st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        if "gender" in filtered:
            gender = filtered["gender"].value_counts().rename_axis("gender").reset_index(name="records")
            fig = px.pie(gender, names="gender", values="records", hole=.48, title="Customer-record mix by gender")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if "season" in filtered:
            season = filtered.groupby("season")["purchase_amount"].mean().reset_index()
            fig = px.bar(season, x="season", y="purchase_amount", title="Average purchase amount by season",
                         labels={"purchase_amount": "Average purchase ($)"})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Executive takeaways")
    insights = []
    if "category" in filtered:
        top_cat = filtered["category"].value_counts().idxmax()
        top_count = int(filtered["category"].value_counts().max())
        insights.append(f"**{top_cat}** is the largest category by record count ({top_count:,} records).")
    if "purchase_amount" in filtered:
        insights.append(f"The average purchase amount is **{money(avg_purchase)}**, compared with a median of **{money(median_purchase)}**.")
    if "review_rating" in filtered and filtered["review_rating"].notna().any():
        insights.append(f"Average customer review rating is **{avg_rating:.2f}/5**.")
    if "subscription_status" in filtered:
        insights.append(f"**{subscription_rate:.1f}%** of records are marked as subscribed.")
    for item in insights:
        st.markdown(f"- {item}")

# ----------------------------- CUSTOMER INTELLIGENCE -----------------------

elif page == "Customer Intelligence":
    section_heading("Customer Intelligence", "Understand who appears in the dataset and how purchase behavior differs across groups.")
    if require_columns(filtered, ["age", "gender", "purchase_amount"]):
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(filtered, x="age", color="gender", nbins=20,
                               title="Age distribution", marginal="box")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(filtered, x="gender", y="purchase_amount", color="gender",
                         title="Purchase amount distribution by gender")
            st.plotly_chart(fig, use_container_width=True)

    if "age_group" in filtered and "purchase_amount" in filtered:
        age_summary = filtered.groupby("age_group", observed=False).agg(
            records=("purchase_amount", "size"),
            avg_purchase=("purchase_amount", "mean"),
            median_purchase=("purchase_amount", "median"),
            avg_rating=("review_rating", "mean") if "review_rating" in filtered else ("purchase_amount", "mean"),
        ).reset_index()
        st.markdown("#### Age-group comparison")
        st.dataframe(age_summary, use_container_width=True, hide_index=True)

    if "location" in filtered and "purchase_amount" in filtered:
        location_summary = filtered.groupby("location").agg(
            records=("purchase_amount", "size"),
            avg_purchase=("purchase_amount", "mean"),
        ).reset_index().sort_values("records", ascending=False).head(15)
        fig = px.bar(location_summary, x="records", y="location", orientation="h",
                     color="avg_purchase", title="Top locations by customer records")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------- PRODUCT & SALES -----------------------------

elif page == "Product & Sales":
    section_heading("Product & Sales Analytics", "Compare categories and items using purchase amount and customer-record volume.")
    if "category" in filtered:
        category_summary = filtered.groupby("category").agg(
            records=("purchase_amount", "size"),
            total_purchase_amount=("purchase_amount", "sum"),
            avg_purchase=("purchase_amount", "mean"),
            median_purchase=("purchase_amount", "median"),
        ).reset_index().sort_values("total_purchase_amount", ascending=False)
        st.dataframe(category_summary, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(category_summary, x="category", y="total_purchase_amount",
                         title="Sum of purchase amount by category",
                         labels={"total_purchase_amount": "Purchase amount ($)"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(category_summary, x="records", y="avg_purchase", size="total_purchase_amount",
                             color="category", hover_name="category",
                             title="Category volume vs average purchase")
            st.plotly_chart(fig, use_container_width=True)

    if "item_purchased" in filtered:
        item_summary = filtered.groupby("item_purchased").agg(
            records=("purchase_amount", "size"),
            avg_purchase=("purchase_amount", "mean"),
        ).reset_index().sort_values("records", ascending=False).head(20)
        fig = px.bar(item_summary, x="records", y="item_purchased", orientation="h",
                     color="avg_purchase", title="Top 20 items by record count")
        st.plotly_chart(fig, use_container_width=True)

    if "size" in filtered:
        size_summary = filtered["size"].value_counts().rename_axis("size").reset_index(name="records")
        fig = px.pie(size_summary, names="size", values="records", hole=.4, title="Product size mix")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------- MARKETING INTELLIGENCE ----------------------

elif page == "Marketing Intelligence":
    section_heading("Marketing Intelligence", "Explore promotion flags, subscription status, payment methods and shipping preferences.")
    c1, c2 = st.columns(2)

    if "discount_applied" in filtered:
        disc = filtered.groupby("discount_applied").agg(
            records=("purchase_amount", "size"),
            avg_purchase=("purchase_amount", "mean"),
        ).reset_index()
        with c1:
            fig = px.bar(disc, x="discount_applied", y="avg_purchase", color="discount_applied",
                         title="Average purchase amount by discount flag",
                         labels={"discount_applied": "Discount applied", "avg_purchase": "Average purchase ($)"})
            st.plotly_chart(fig, use_container_width=True)

    if "subscription_status" in filtered:
        sub = filtered.groupby("subscription_status").agg(
            records=("purchase_amount", "size"),
            avg_purchase=("purchase_amount", "mean"),
        ).reset_index()
        with c2:
            fig = px.bar(sub, x="subscription_status", y="records", color="subscription_status",
                         title="Records by subscription status")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Subscription comparison**")
        st.dataframe(sub, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    if "payment_method" in filtered:
        payment = filtered["payment_method"].value_counts().rename_axis("payment_method").reset_index(name="records")
        with c1:
            fig = px.pie(payment, names="payment_method", values="records", hole=.4, title="Payment method mix")
            st.plotly_chart(fig, use_container_width=True)
    if "shipping_type" in filtered:
        shipping = filtered["shipping_type"].value_counts().rename_axis("shipping_type").reset_index(name="records")
        with c2:
            fig = px.bar(shipping, x="shipping_type", y="records", title="Shipping preference")
            st.plotly_chart(fig, use_container_width=True)

    if "discount_applied" in filtered and "review_rating" in filtered:
        st.caption("These are observational comparisons, not proof that discounts cause higher purchase amounts or ratings.")

# ----------------------------- SEGMENTATION --------------------------------

elif page == "Customer Segmentation":
    section_heading("Behavioral Customer Segmentation", "K-Means clustering on available customer attributes. Segments are exploratory, not customer lifetime-value or churn labels.")
    numeric_candidates = [c for c in ["age", "purchase_amount", "review_rating", "previous_purchases",
                                       "estimated_purchase_interval_days"] if c in filtered.columns]
    categorical_candidates = [c for c in ["gender", "category", "season", "subscription_status",
                                           "payment_method", "shipping_type"] if c in filtered.columns]

    selected_numeric = st.multiselect("Numeric features", numeric_candidates, default=numeric_candidates[:4])
    selected_categorical = st.multiselect("Categorical features", categorical_candidates, default=categorical_candidates[:3])
    k = st.slider("Number of clusters (K)", min_value=2, max_value=8, value=4)

    if not selected_numeric and not selected_categorical:
        st.warning("Choose at least one feature to create segments.")
    elif len(filtered) <= k:
        st.warning(f"At least {k + 1} filtered records are needed for {k} clusters.")
    else:
        feature_df = filtered[selected_numeric + selected_categorical].copy()
        for col in selected_numeric:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        for col in selected_categorical:
            feature_df[col] = feature_df[col].fillna("Unknown").astype(str)

        numeric_pipe = Pipeline([("scale", StandardScaler())])
        categorical_pipe = Pipeline([("encode", OneHotEncoder(handle_unknown="ignore"))])
        transformers = []
        if selected_numeric:
            transformers.append(("num", numeric_pipe, selected_numeric))
        if selected_categorical:
            transformers.append(("cat", categorical_pipe, selected_categorical))
        preprocessor = ColumnTransformer(transformers, remainder="drop")
        try:
            matrix = preprocessor.fit_transform(feature_df)
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(matrix)
            segment_df = filtered.copy()
            segment_df["segment"] = labels.astype(str)
            st.success(f"Created {k} exploratory behavioral segments using {len(segment_df):,} records.")

            segment_summary = segment_df.groupby("segment").agg(
                records=("purchase_amount", "size"),
                avg_purchase=("purchase_amount", "mean"),
                median_purchase=("purchase_amount", "median"),
                total_purchase_amount=("purchase_amount", "sum"),
            ).reset_index()
            segment_summary["share_pct"] = 100 * segment_summary["records"] / len(segment_df)
            st.dataframe(segment_summary.round(2), use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(segment_df, x="age" if "age" in segment_df else selected_numeric[0],
                                 y="purchase_amount", color="segment",
                                 title="Behavioral segments",
                                 hover_data=[c for c in ["category", "gender", "subscription_status"] if c in segment_df])
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(segment_summary, x="segment", y="records", color="segment",
                             title="Records in each segment")
                st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download customer segments",
                data=csv_download(segment_df),
                file_name="retailpulse_customer_segments.csv",
                mime="text/csv",
            )
            st.caption("For a production segmentation project, compare K values using silhouette score and stability checks before selecting a final model.")
        except Exception as exc:
            st.error(f"Segmentation could not be completed: {exc}")

# ----------------------------- DATA QUALITY & SQL -------------------------

elif page == "Data Quality & SQL":
    section_heading("Data Quality & SQL", "Inspect the schema, missing values, distributions and cleaned records.")
    quality = pd.DataFrame({
        "column": df.columns,
        "data_type": [str(df[c].dtype) for c in df.columns],
        "missing_values": [int(df[c].isna().sum()) for c in df.columns],
        "missing_pct": [round(100 * df[c].isna().mean(), 2) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Exact duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.markdown("#### Dataset inspection")
    st.caption(
        "The uploaded CSV is cleaned and analyzed in memory. Use the tables, charts, "
        "filters, and download options on this page to inspect the dataset."
    )

    st.markdown("#### Preview cleaned records")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download current filtered records",
        data=csv_download(filtered),
        file_name="retailpulse_filtered_records.csv",
        mime="text/csv",
    )

# ----------------------------- FOOTER --------------------------------------

st.divider()
st.caption(
    "RetailPulse portfolio project | Python · Pandas · Streamlit · Plotly · scikit-learn"
)
st.caption(
    "Analytical note: the source dataset is cross-sectional. Findings describe patterns in the available records "
    "and should not be interpreted as causal effects or longitudinal customer behavior."
)

