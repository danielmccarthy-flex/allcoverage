import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from rapidfuzz import process, fuzz

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(layout="wide")
st.title("Agency Coverage & Rate Card Intelligence")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COVERAGE_FILE = os.path.join(BASE_DIR, "coverage.csv")
RATE_FILE = os.path.join(BASE_DIR, "rate_cards.csv")

FUZZY_THRESHOLD = 85

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def clean_name(val):
    if pd.isna(val):
        return ""
    val = val.lower()
    val = re.sub(r"[^a-z0-9 ]", "", val)
    val = re.sub(r"\s+", "", val)
    return val

def fuzzy_match(name, choices):
    if not name or not choices:
        return None
    match = process.extractOne(name, choices, scorer=fuzz.ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        return match[0]
    return None

# ------------------------------------------------
# Load data
# ------------------------------------------------
@st.cache_data
def load_data():
    coverage = pd.read_csv(COVERAGE_FILE)
    rate = pd.read_csv(RATE_FILE)

    coverage["agency_clean"] = coverage["agency_name"].apply(clean_name)
    coverage["city_clean"] = coverage["city"].apply(clean_name)

    rate["agency_clean"] = rate["agency_name"].apply(clean_name)
    rate["city_clean"] = rate["venue_city"].apply(clean_name)

    return coverage, rate

coverage, rate_cards = load_data()

# ------------------------------------------------
# Fuzzy agency alignment (rate → coverage)
# ------------------------------------------------
coverage_agencies = coverage["agency_clean"].dropna().unique().tolist()
rate_cards["coverage_agency_clean"] = rate_cards["agency_clean"].apply(
    lambda x: fuzzy_match(x, coverage_agencies)
)

# ------------------------------------------------
# Build agency × city spine (union)
# ------------------------------------------------
rate_pairs = (
    rate_cards[["agency_name", "agency_clean", "city_clean", "venue_city"]]
    .dropna(subset=["agency_name", "venue_city"])
    .rename(columns={"venue_city": "city"})
)
coverage_pairs = (
    coverage[["agency_name", "agency_clean", "city_clean", "city"]]
    .dropna(subset=["agency_name", "city"])
)
agency_city_spine = pd.concat([rate_pairs, coverage_pairs], ignore_index=True)
agency_city_spine = agency_city_spine.drop_duplicates(subset=["agency_clean", "city_clean"])

# ------------------------------------------------
# Attach rate cards and coverage
# ------------------------------------------------
spine_rates = agency_city_spine.merge(
    rate_cards,
    how="left",
    left_on=["agency_clean", "city_clean"],
    right_on=["agency_clean", "city_clean"],
    suffixes=("", "_rate")
)
final = spine_rates.merge(
    coverage,
    how="left",
    left_on=["agency_clean", "city_clean"],
    right_on=["agency_clean", "city_clean"],
    suffixes=("", "_coverage")
)

# ------------------------------------------------
# Presence type
# ------------------------------------------------
final["presence_type"] = np.select(
    [
        final["agency_margin"].notna() & final["role_category"].notna(),
        final["agency_margin"].notna(),
        final["role_category"].notna(),
    ],
    [
        "Rate Card + Coverage",
        "Rate Card Only",
        "Coverage Only",
    ],
    default="Unknown"
)

# ------------------------------------------------
# City averages
# ------------------------------------------------
city_avg = final.groupby("city", as_index=False)["agency_margin"].mean().rename(
    columns={"agency_margin": "city_avg_margin"}
)
final = final.merge(city_avg, on="city", how="left")
final["margin_vs_city_avg"] = final["agency_margin"] - final["city_avg_margin"]

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.header("View")
view = st.sidebar.radio(
    "Mode",
    ["Coverage View", "Agency View", "City View"]
)

st.sidebar.header("Filters")
cities = sorted(final["city"].dropna().unique())
agencies = sorted(final["agency_name"].dropna().unique())

selected_city = st.sidebar.selectbox("City", ["All"] + cities)
selected_agency = st.sidebar.selectbox("Agency", ["All"] + agencies)

df = final.copy()
if selected_city != "All":
    df = df[df["city"] == selected_city]
if selected_agency != "All":
    df = df[df["agency_name"] == selected_agency]

# =================================================
# COVERAGE VIEW
# =================================================
if view == "Coverage View":
    st.subheader("Market Coverage Overview")
    cols = [
        "agency_name",
        "city",
        "presence_type",
        "agency_margin",
        "city_avg_margin",
        "margin_vs_city_avg",
        "role_category",
        "supply_capability"
    ]
    st.dataframe(
        df[cols].sort_values(
            ["city", "presence_type", "agency_margin"],
            ascending=[True, True, False]
        ),
        use_container_width=True
    )

# =================================================
# AGENCY VIEW
# =================================================
elif view == "Agency View":
    st.subheader("Agency Performance")
    if selected_agency == "All":
        st.warning("Select a specific agency.")
    else:
        a = df[df["agency_name"] == selected_agency]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cities (Any Presence)", a["city"].nunique())
        c2.metric("Cities w/ Rate Cards", a[a["agency_margin"].notna()]["city"].nunique())
        c3.metric("Avg Margin", round(a["agency_margin"].mean(), 2))
        c4.metric("Avg vs City", round(a["margin_vs_city_avg"].mean(), 2))

        st.markdown("### City Breakdown")
        city_tbl = a.groupby("city", as_index=False).agg(
            presence=("presence_type", lambda x: ", ".join(sorted(set(x)))),
            avg_margin=("agency_margin", "mean"),
            city_avg=("city_avg_margin", "mean")
        )
        city_tbl["delta"] = city_tbl["avg_margin"] - city_tbl["city_avg"]
        st.dataframe(city_tbl.sort_values("delta", ascending=False), use_container_width=True)

# =================================================
# CITY VIEW
# =================================================
elif view == "City View":
    st.subheader("City Market View")
    if selected_city == "All":
        st.warning("Select a city.")
    else:
        c = df[df["city"] == selected_city]
        c1, c2, c3 = st.columns(3)
        c1.metric("Agencies", c["agency_name"].nunique())
        c2.metric("Coverage Only", (c["presence_type"] == "Coverage Only").sum())
        c3.metric("City Avg Margin", round(c["agency_margin"].mean(), 2))

        st.markdown("### Agencies in City")
        rank = c.groupby("agency_name", as_index=False).agg(
            presence=("presence_type", lambda x: ", ".join(sorted(set(x)))),
            avg_margin=("agency_margin", "mean"),
            venues=("venue_name", "nunique")
        )
        rank["rank"] = rank["avg_margin"].rank(method="dense", ascending=False)
        st.dataframe(rank.sort_values("rank"), use_container_width=True)

# ------------------------------------------------
# Export
# ------------------------------------------------
st.download_button(
    "Export Current View",
    df.to_csv(index=False),
    file_name="agency_city_spine_export.csv",
    mime="text/csv"
)
