import streamlit as st
import pandas as pd
import numpy as np
import re
from rapidfuzz import process, fuzz

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(layout="wide")
st.title("Agency Coverage & Rate Card Intelligence")

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
    if not name or len(choices) == 0:
        return None
    match = process.extractOne(name, choices, scorer=fuzz.ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        return match[0]
    return None

# ------------------------------------------------
# Cached CSV loader
# ------------------------------------------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# ------------------------------------------------
# Sidebar Uploads
# ------------------------------------------------
st.sidebar.header("📂 Upload Data Files")

agency_file = st.sidebar.file_uploader("Upload Agency Coverage CSV", type=["csv"], key="agency_csv")
ratecard_file = st.sidebar.file_uploader("Upload Rate Card CSV", type=["csv"], key="ratecard_csv")

if agency_file is None or ratecard_file is None:
    st.info("Please upload both CSV files to begin.")
    st.stop()

# ------------------------------------------------
# Load Data
# ------------------------------------------------
agency_df = load_csv(agency_file)
ratecard_df = load_csv(ratecard_file)

# ------------------------------------------------
# Required Columns Check
# ------------------------------------------------
required_agency_cols = ["agency_name", "city"]
missing_agency = [c for c in required_agency_cols if c not in agency_df.columns]

required_ratecard_cols = ["agency_name", "venue_city", "agency_margin"]
missing_ratecard = [c for c in required_ratecard_cols if c not in ratecard_df.columns]

if missing_agency:
    st.error(f"Missing columns in Agency Coverage CSV: {missing_agency}")
    st.stop()
if missing_ratecard:
    st.error(f"Missing columns in Rate Card CSV: {missing_ratecard}")
    st.stop()

# ------------------------------------------------
# Handle optional columns
# ------------------------------------------------
if "role_category" not in agency_df.columns:
    agency_df["role_category"] = np.nan
if "supply_capability" not in agency_df.columns:
    agency_df["supply_capability"] = np.nan

# Map employer_id to platforms.employer_id if needed
if "platforms.employer_id" not in ratecard_df.columns and "employer_id" in ratecard_df.columns:
    ratecard_df["platforms.employer_id"] = ratecard_df["employer_id"]
elif "platforms.employer_id" not in ratecard_df.columns:
    ratecard_df["platforms.employer_id"] = np.nan

# Ensure margins are numeric and drop empty
ratecard_df["agency_margin"] = pd.to_numeric(ratecard_df.get("agency_margin", np.nan), errors="coerce")
ratecard_df = ratecard_df.dropna(subset=["agency_margin"])

# ------------------------------------------------
# Clean & prepare names for fuzzy matching
# ------------------------------------------------
agency_df["agency_clean"] = agency_df["agency_name"].apply(clean_name)
agency_df["city_clean"] = agency_df["city"].apply(clean_name)

ratecard_df["agency_clean"] = ratecard_df["agency_name"].apply(clean_name)
ratecard_df["city_clean"] = ratecard_df["venue_city"].apply(clean_name)

# ------------------------------------------------
# Fuzzy matching: agency names
# ------------------------------------------------
coverage_agencies = agency_df["agency_clean"].dropna().unique().tolist()
ratecard_df["coverage_agency_clean"] = ratecard_df["agency_clean"].apply(
    lambda x: fuzzy_match(x, coverage_agencies)
)

# ------------------------------------------------
# Fuzzy matching: city names
# ------------------------------------------------
coverage_cities = agency_df["city_clean"].dropna().unique().tolist()
ratecard_df["coverage_city_clean"] = ratecard_df["city_clean"].apply(
    lambda x: fuzzy_match(x, coverage_cities)
)

# ------------------------------------------------
# Build agency × city spine (union)
# ------------------------------------------------
rate_pairs = (
    ratecard_df[["agency_name", "agency_clean", "coverage_city_clean", "venue_city", "platforms.employer_id"]]
    .dropna(subset=["agency_name", "coverage_city_clean"])
    .rename(columns={"coverage_city_clean": "city_clean", "venue_city": "city"})
)

coverage_pairs = (
    agency_df[["agency_name", "agency_clean", "city_clean", "city"]]
    .dropna(subset=["agency_name", "city"])
)

agency_city_spine = (
    pd.concat([rate_pairs, coverage_pairs], ignore_index=True)
    .drop_duplicates(subset=["agency_clean", "city_clean"])
)

# ------------------------------------------------
# Attach rate cards
# ------------------------------------------------
spine_rates = agency_city_spine.merge(
    ratecard_df,
    how="left",
    left_on=["agency_clean", "city_clean"],
    right_on=["agency_clean", "city_clean"],
    suffixes=("", "_rate")
)

# ------------------------------------------------
# Attach coverage
# ------------------------------------------------
final = spine_rates.merge(
    agency_df,
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
        final["agency_margin"].notna() & final.get("role_category").notna(),
        final["agency_margin"].notna(),
        final.get("role_category").notna(),
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
city_avg = (
    final.groupby("city", as_index=False)["agency_margin"]
    .mean()
    .rename(columns={"agency_margin": "city_avg_margin"})
)

final = final.merge(city_avg, on="city", how="left")
final["margin_vs_city_avg"] = final["agency_margin"] - final["city_avg_margin"]

# ------------------------------------------------
# Sidebar Filters
# ------------------------------------------------
st.sidebar.header("View")
view = st.sidebar.radio("Mode", ["Coverage View", "Agency View", "City View", "Client View"])

st.sidebar.header("Filters")
cities = sorted(final["city"].dropna().unique())
agencies = sorted(final["agency_name"].dropna().unique())
clients = sorted(final["platforms.employer_id"].dropna().unique())

selected_city = st.sidebar.multiselect("City", ["All"] + cities, default=["All"])
selected_agency = st.sidebar.multiselect("Agency", ["All"] + agencies, default=["All"])
selected_client = st.sidebar.selectbox("Client", ["All"] + clients)

df = final.copy()
if "All" not in selected_city:
    df = df[df["city"].isin(selected_city)]
if "All" not in selected_agency:
    df = df[df["agency_name"].isin(selected_agency)]
if selected_client != "All":
    df = df[df["platforms.employer_id"] == selected_client]

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
    if len(selected_agency) != 1 or selected_agency[0] == "All":
        st.warning("Select a single specific agency to view Agency View.")
    else:
        agency_name = selected_agency[0]
        a = df[df["agency_name"] == agency_name]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cities (Any Presence)", a["city"].nunique())
        c2.metric("Cities w/ Rate Cards", a[a["agency_margin"].notna()]["city"].nunique())
        c3.metric("Avg Margin", round(a["agency_margin"].mean(), 2) if not a["agency_margin"].empty else "N/A")
        c4.metric("Avg vs City", round(a["margin_vs_city_avg"].mean(), 2) if not a["margin_vs_city_avg"].empty else "N/A")

        st.markdown("### City Breakdown")
        city_tbl = (
            a.groupby("city", as_index=False)
            .agg(
                presence=("presence_type", lambda x: ", ".join(sorted(set(x)))),
                avg_margin=("agency_margin", "mean"),
                city_avg=("city_avg_margin", "mean")
            )
        )
        city_tbl["delta"] = city_tbl["avg_margin"] - city_tbl["city_avg"]
        st.dataframe(
            city_tbl.sort_values("delta", ascending=False),
            use_container_width=True
        )

# =================================================
# CITY VIEW
# =================================================
elif view == "City View":
    st.subheader("City Market View")
    if len(selected_city) != 1 or selected_city[0] == "All":
        st.warning("Select a single city to view City View.")
    else:
        city_name = selected_city[0]
        c = df[df["city"] == city_name]

        c1, c2, c3 = st.columns(3)
        c1.metric("Agencies", c["agency_name"].nunique())
        c2.metric("Coverage Only", (c["presence_type"] == "Coverage Only").sum())
        city_avg_val = c["agency_margin"].dropna()
        c3.metric("City Avg Margin", round(city_avg_val.mean(), 2) if not city_avg_val.empty else "N/A")

        st.markdown("### Agencies in City")
        rank = (
            c.groupby("agency_name", as_index=False)
            .agg(
                presence=("presence_type", lambda x: ", ".join(sorted(set(x)))),
                avg_margin=("agency_margin", "mean"),
                venues=("venue_name", "nunique") if "venue_name" in c.columns else pd.Series(np.nan)
            )
        )
        if "rank" not in rank.columns:
            rank["rank"] = rank["avg_margin"].rank(method="dense", ascending=False)
        st.dataframe(
            rank.sort_values("rank"),
            use_container_width=True
        )

# =================================================
# CLIENT VIEW
# =================================================
elif view == "Client View":
    st.subheader("Client View")
    if selected_client == "All":
        st.warning("Select a specific client to view Client View.")
    else:
        client_df = df[df["platforms.employer_id"] == selected_client]
        if client_df.empty:
            st.info("No data for this client with current filters.")
        else:
            pivot = client_df.pivot_table(
                index="city",
                columns="agency_name",
                values="agency_margin",
                aggfunc="mean"
            )
            st.dataframe(pivot.fillna("-"), use_container_width=True)

# ------------------------------------------------
# Export
# ------------------------------------------------
st.download_button(
    "Export Current View",
    df.to_csv(index=False),
    file_name="agency_city_spine_export.csv",
    mime="text/csv"
)
