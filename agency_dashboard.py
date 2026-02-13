import streamlit as st
import pandas as pd
import numpy as np
import re
from rapidfuzz import process, fuzz

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Agency Intelligence")
st.title("Agency Coverage & Rate Card Intelligence")

# ------------------------------------------------
# 1. Manual Overrides (Universal Brand Sync)
# ------------------------------------------------
NAME_OVERRIDES = {
    "allegiencestaffing": "Allegiance Staffing",
    "hwstaffing": "HW Staffing Solutions",
    "manpower": "Manpower Group",
    "manpowergroup": "Manpower Group",
    "malonesolutions": "Management Registry Inc. DBA Malone Workforce Solutions",
    "managementregistry": "Management Registry Inc. DBA Malone Workforce Solutions",
    "selectsource": "Select Source International",
    "epicpersonnelpartners": "Epic Personnel Partners",
    "peopleready": "PeopleReady"
}

# ------------------------------------------------
# 2. Helpers
# ------------------------------------------------
def get_clean_key(s):
    """Standardizes strings for matching by removing non-alphanumerics."""
    s = str(s).lower()
    return re.sub(r"[^a-z0-9]", "", s).strip()

def unify_agency_names(names_list):
    """Standardizes variants while picking the most formal version."""
    unique_raw = [n for n in names_list if pd.notna(n) and str(n).strip() != ""]
    master_map = {}
    
    groups = {}
    for name in unique_raw:
        key = get_clean_key(name)
        if key not in groups: groups[key] = []
        groups[key].append(name)
    
    for key, variants in groups.items():
        if key in NAME_OVERRIDES:
            master = NAME_OVERRIDES[key]
        else:
            master = sorted(variants, key=lambda x: (x.count(' '), sum(1 for c in x if c.isupper())), reverse=True)[0]
        for v in variants:
            master_map[v] = master
                
    return master_map

def clean_numeric(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('%', '').replace(',', '').strip()
    try:
        return float(s)
    except:
        return np.nan

@st.cache_data
def load_csv_safe(file):
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, sep='\t', encoding='utf-16')
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=None, engine='python')

# ------------------------------------------------
# 3. Multi-File Upload & Identification
# ------------------------------------------------
st.sidebar.header("📂 Data Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload Coverage, Rate Card, and Scorecard files together", 
    type=["csv", "txt"], 
    accept_multiple_files=True
)

a_df, r_df, s_df = None, None, None

if uploaded_files:
    for file in uploaded_files:
        temp_df = load_csv_safe(file)
        cols = [c.lower() for c in temp_df.columns]
        
        if "supply_capability" in cols or "role_category" in cols:
            a_df = temp_df
        elif "agency_margin" in cols:
            r_df = temp_df
        elif "fulfilled%" in cols or "agency_worker_requested" in cols:
            s_df = temp_df

if a_df is None or r_df is None or s_df is None:
    st.info("Please upload all three required files to begin.")
    st.stop()

def standardize_columns(df):
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    mapping = {
        'brand': 'agency_name', 'vendor': 'agency_name', 'agency': 'agency_name',
        'venue_city': 'city', 'location': 'city', 'market': 'city',
        'employer_id': 'platforms.employer_id', 'venue': 'venue_name',
        'employer_name': 'client_name', 'fulfilled%': 'fulfilled_val',
        'agency_worker_requested': 'shifts_requested', 
        'actual_agency_worker_provided': 'shifts_filled'
    }
    return df.rename(columns=mapping)

a_df = standardize_columns(a_df)
r_df = standardize_columns(r_df)
s_df = standardize_columns(s_df)

# ------------------------------------------------
# 4. Global Data Integration
# ------------------------------------------------
all_names = pd.concat([a_df["agency_name"], r_df["agency_name"], s_df["agency_name"]]).unique()
master_map = unify_agency_names(all_names)

for df in [a_df, r_df, s_df]:
    df["agency_name"] = df["agency_name"].map(master_map)
    df["city"] = df["city"].fillna("Unknown").str.strip().str.title()

s_df["fulfilled_val"] = s_df["fulfilled_val"].apply(clean_numeric)
s_df["shifts_requested"] = s_df["shifts_requested"].apply(clean_numeric)
s_df["shifts_filled"] = s_df["shifts_filled"].apply(clean_numeric)

scorecard_summary = s_df.groupby(['agency_name', 'city'], as_index=False).agg({
    'fulfilled_val': 'mean',
    'shifts_requested': 'sum',
    'shifts_filled': 'sum'
}).rename(columns={'fulfilled_val': 'avg_fulfillment'})

combined = pd.concat([a_df, r_df], ignore_index=True)
combined["agency_margin"] = pd.to_numeric(combined.get("agency_margin", np.nan), errors="coerce")

agg_logic = {
    "agency_margin": "mean",
    "venue_name": lambda x: x.nunique() if 'venue_name' in combined.columns else 0
}
for col in ["role_category", "supply_capability", "platforms.employer_id"]:
    if col in combined.columns:
        agg_logic[col] = lambda x: ", ".join(sorted(set([str(i).strip() for i in x if pd.notna(i) and str(i).strip() != ""])))

master_display_df = combined.groupby(["agency_name", "city"], as_index=False).agg(agg_logic)
master_display_df = master_display_df.merge(scorecard_summary, on=['agency_name', 'city'], how='left')

r_df["is_2554"] = r_df["platforms.employer_id"].astype(str).str.contains("2554")
agency_rank_metrics = r_df.groupby(['agency_name', 'city']).agg(
    avg_margin=('agency_margin', 'mean'),
    count_2554=('is_2554', 'sum')
).reset_index()

# ------------------------------------------------
# 5. Dashboard Navigation
# ------------------------------------------------
view = st.sidebar.radio("View Mode", ["Coverage View", "Agency View", "City View", "Cascade Builder"])
cities = sorted(master_display_df["city"].unique())
selected_city = st.sidebar.multiselect("Filter City", ["All"] + cities, default=["All"])

df_filt = master_display_df.copy()
if "All" not in selected_city:
    df_filt = df_filt[df_filt["city"].isin(selected_city)]

# ------------------------------------------------
# 6. View Implementations
# ------------------------------------------------

if view == "Coverage View":
    st.subheader("Market Coverage Overview")
    st.dataframe(df_filt.sort_values(["city", "agency_name"]), use_container_width=True)

elif view == "Agency View":
    agencies = sorted(df_filt["agency_name"].unique())
    sel_agency = st.sidebar.selectbox("Select Agency", agencies)
    a_data = df_filt[df_filt["agency_name"] == sel_agency]
    st.subheader(f"Agency Deep-Dive: {sel_agency}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Margin", f"{round(a_data['agency_margin'].mean(), 2)}")
    c2.metric("Avg Fulfillment", f"{round(a_data['avg_fulfillment'].mean(), 1)}%")
    c3.metric("Total Requested", f"{int(a_data['shifts_requested'].sum()):,}")
    c4.metric("Total Filled", f"{int(a_data['shifts_filled'].sum()):,}")
    st.dataframe(a_data[["city", "agency_margin", "avg_fulfillment", "shifts_requested", "shifts_filled"]], use_container_width=True)

elif view == "City View":
    if len(selected_city) != 1 or selected_city[0] == "All":
        st.warning("Please filter for exactly one city in the sidebar to see market-wide metrics.")
    else:
        c_name = selected_city[0]
        c_data = df_filt[df_filt["city"] == c_name]
        st.subheader(f"Market Snapshot: {c_name}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("City Avg Margin", f"{round(c_data['agency_margin'].mean(), 2)}")
        m2.metric("Market Fulfillment", f"{round(c_data['avg_fulfillment'].mean(), 1)}%")
        m3.metric("Total Market Volume", f"{int(c_data['shifts_requested'].sum()):,}")
        m4.metric("Active Agencies", c_data["agency_name"].nunique())
        st.dataframe(c_data[["agency_name", "agency_margin", "avg_fulfillment", "shifts_requested", "shifts_filled"]].sort_values("avg_fulfillment", ascending=False), use_container_width=True)

elif view == "Cascade Builder":
    st.subheader("Agency Cascade Builder")
    st.write("Upload a cascade template to automatically rank and fill agency tiers.")
    
    # Exclusion tool: List unique unified agencies for selection
    all_final_agencies = sorted(master_display_df["agency_name"].unique())
    excluded_agencies = st.multiselect("Exclude Specific Agencies from Cascade", all_final_agencies)

    cascade_file = st.file_uploader("Upload Cascade Template CSV", type=["csv"])
    
    if cascade_file:
        template_df = pd.read_csv(cascade_file)
        
        if st.button("Generate Filled Cascade"):
            # Step 1: Gather coverage pool
            coverage_pool = a_df[['agency_name', 'city']].drop_duplicates()
            
            # Step 2: Apply exclusion filter
            if excluded_agencies:
                # Filter against the MASTER unified name
                coverage_pool = coverage_pool[~coverage_pool['agency_name'].isin(excluded_agencies)]
            
            # Step 3: Enrich with performance metrics
            pool = coverage_pool.merge(agency_rank_metrics, on=['agency_name', 'city'], how='left')
            pool['avg_margin'] = pool['avg_margin'].fillna(999) 
            pool['count_2554'] = pool['count_2554'].fillna(0)
            
            # Step 4: Rank - 2554 preference first, then lowest margin
            pool = pool.sort_values(by=['city', 'count_2554', 'avg_margin'], ascending=[True, False, True])
            
            # Step 5: Map to template tiers
            rank_cols = [c for c in template_df.columns if "RANK_" in c]
            
            for idx, row in template_df.iterrows():
                row_city = str(row['CITY']).strip().title()
                city_agencies = pool[pool['city'] == row_city]['agency_name'].tolist()
                
                for i, col in enumerate(rank_cols):
                    if pd.isna(row[col]) and i < len(city_agencies):
                        template_df.at[idx, col] = city_agencies[i]
            
            st.success("Cascade filled successfully.")
            st.dataframe(template_df, use_container_width=True)
            
            st.download_button(
                "Download Completed Cascade",
                template_df.to_csv(index=False),
                "filled_agency_cascade.csv",
                "text/csv"
            )

st.sidebar.markdown("---")
st.sidebar.download_button("Export Master Dataset", master_display_df.to_csv(index=False), "agency_intelligence.csv")