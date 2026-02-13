import streamlit as st
import pandas as pd
import numpy as np
import re

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(layout="wide", page_title="Cascade Strategy Tool")
st.title("🎯 Agency Cascade Builder (v2.9.4)")

# ------------------------------------------------
# 1. Helpers & Standard Loaders
# ------------------------------------------------
def get_zip3(z):
    """Extracts the first 3 digits of a ZIP code."""
    z = re.sub(r"[^0-9]", "", str(z))
    return z[:3] if len(z) >= 3 else None

def get_clean_key(s):
    """Standardizes strings for matching by removing spaces and special characters."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower()).strip()

@st.cache_data
def load_data_safe(file):
    """Robustly loads files with multiple encoding fallbacks."""
    try:
        file.seek(0)
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding='ISO-8859-1') 
        except:
            file.seek(0)
            df = pd.read_csv(file, sep='\t', encoding='utf-16')
    except Exception:
        df = pd.read_excel(file)
    
    # Clean headers immediately to prevent KeyErrors
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    return df

# ------------------------------------------------
# 2. Sidebar - Data Ingestion
# ------------------------------------------------
st.sidebar.header("📂 1. Data Ingestion")
data_files = st.sidebar.file_uploader("Upload Data (Coverage, Rate, Score)", accept_multiple_files=True)
template_file = st.sidebar.file_uploader("Upload Cascade Template", type=["csv", "xlsx"])
key_file = st.sidebar.file_uploader("Upload Agency-to-ACP ID Key (Optional)", type=["csv", "xlsx"])

if not data_files or not template_file:
    st.info("Please upload your Data Files and Cascade Template in the sidebar to begin.")
    st.stop()

# Identify Data Files
a_df, r_df, s_df, k_df = None, None, None, None
for f in data_files:
    df = load_data_safe(f)
    cols = df.columns.tolist()
    if "supply_capability" in cols or "role_category" in cols: a_df = df
    elif "agency_margin" in cols: r_df = df
    elif "fulfilled%" in cols or "agency_worker_requested" in cols: s_df = df

if key_file: k_df = load_data_safe(key_file)

# Standardize Columns
def standardize(df):
    m = {
        'brand':'agency_name', 'vendor':'agency_name', 'agency':'agency_name',
        'venue_city':'city', 'fulfilled%':'fulfillment', 'actual_agency_worker_provided':'filled',
        'employer_name': 'client_name', 'employer_id': 'platforms.employer_id'
    }
    df = df.rename(columns=m)
    if 'city' in df.columns:
        df['city_match'] = df['city'].apply(get_clean_key)
        df['city'] = df['city'].fillna("").str.strip().str.title()
    return df

if a_df is not None: a_df = standardize(a_df)
if r_df is not None: r_df = standardize(r_df)
if s_df is not None: s_df = standardize(s_df)

# ------------------------------------------------
# 3. Strategy & Performance Logic
# ------------------------------------------------
s_df['fulfillment'] = pd.to_numeric(s_df['fulfillment'].astype(str).str.replace('%',''), errors='coerce')
c_col = 'client_name' if 'client_name' in s_df.columns else s_df.columns[1]
target_clients = s_df[s_df[c_col].str.contains("Stord|CORT", na=False, case=False)][c_col].unique().tolist()

# Hierarchical Performance scores
cc_perf = s_df[s_df[c_col].isin(target_clients)].groupby(['agency_name','city_match'])['fulfillment'].mean().reset_index().rename(columns={'fulfillment':'score_city'})
cg_perf = s_df[s_df[c_col].isin(target_clients)].groupby(['agency_name'])['fulfillment'].mean().reset_index().rename(columns={'fulfillment':'score_client_global'})
g_perf = s_df.groupby(['agency_name'])['fulfillment'].mean().reset_index().rename(columns={'fulfillment':'score_global'})

r_df['agency_margin'] = pd.to_numeric(r_df['agency_margin'], errors='coerce')
margins = r_df.groupby(['agency_name','city_match'])['agency_margin'].mean().reset_index()

# Build Pool from BOTH Coverage and Rate Cards (LaVergne fix)
pool_a = a_df[['agency_name','city_match']].drop_duplicates() if a_df is not None else pd.DataFrame()
pool_r = r_df[['agency_name','city_match']].drop_duplicates() if r_df is not None else pd.DataFrame()
pool = pd.concat([pool_a, pool_r], ignore_index=True).drop_duplicates()

# Merge scores and margins
pool = pool.merge(cc_perf, on=['agency_name','city_match'], how='left').merge(cg_perf, on=['agency_name'], how='left').merge(g_perf, on=['agency_name'], how='left').merge(margins, on=['agency_name','city_match'], how='left')
pool['eff_fulfillment'] = pool['score_city'].fillna(pool['score_client_global']).fillna(pool['score_global'])

# ID Mapping
id_map = {}
if k_df is not None:
    n_col = next((c for c in k_df.columns if 'name' in c or 'agency' in c), None)
    i_col = next((c for c in k_df.columns if 'id' in c or 'acp' in c), None)
    if n_col and i_col: id_map = dict(zip(k_df[n_col], k_df[i_col]))

# ------------------------------------------------
# 4. Strategy Settings
# ------------------------------------------------
st.sidebar.header("⚙️ 2. Strategy Settings")
threshold = st.sidebar.slider("Fulfillment Threshold (%)", 0, 100, 50)
export_mode = st.sidebar.radio("Export Format", ["Agency Name", "ACP ID"])

# FIXED: Handle NaN values in agency_name for sorting/multiselect
valid_agencies = sorted([str(x) for x in pool['agency_name'].unique() if pd.notna(x)])
exclude = st.sidebar.multiselect("Exclude Agencies", valid_agencies)

if exclude: 
    pool = pool[~pool['agency_name'].astype(str).isin(exclude)]

def get_ranked_agencies(city_pool, thresh):
    prioritized = city_pool[city_pool['eff_fulfillment'] >= thresh].sort_values(['eff_fulfillment','agency_margin'], ascending=[False, True])
    deprioritized = city_pool[(city_pool['eff_fulfillment'] < thresh) | (city_pool['eff_fulfillment'].isna())].sort_values(['agency_margin','eff_fulfillment'], ascending=[True, False])
    return pd.concat([prioritized, deprioritized])['agency_name'].tolist()

# ------------------------------------------------
# 5. Main Generation Logic
# ------------------------------------------------
template = load_data_safe(template_file)
city_col = next((c for c in template.columns if 'city' in c), 'city')
zip_col = next((c for c in template.columns if 'post' in c or 'zip' in c), 'post_code')
venue_col = next((c for c in template.columns if 'venue' in c), 'venue_name')
rank_cols = [c for c in template.columns if "rank_" in c]

if st.button("🚀 Generate Final Cascade & Run Analysis"):
    # Phase 1: Direct City Match (using clean keys for LaVergne fix)
    for idx, row in template.iterrows():
        clean_city = get_clean_key(row[city_col])
        city_pool = pool[pool['city_match'] == clean_city]
        if not city_pool.empty:
            ranked = get_ranked_agencies(city_pool, threshold)
            for i, col in enumerate(rank_cols):
                if i < len(ranked):
                    val = ranked[i]
                    template.at[idx, col] = id_map.get(val, val) if export_mode == "ACP ID" else val

    # Phase 2: ZIP3 Cloning
    template['zip3'] = template[zip_col].apply(get_zip3)
    zip3_cascades = {r['zip3']: [r[c] for c in rank_cols] for _, r in template.iterrows() if pd.notna(r[rank_cols[0]]) and r['zip3']}
    for idx, row in template.iterrows():
        if pd.isna(row[rank_cols[0]]) and row['zip3'] in zip3_cascades:
            for i, col in enumerate(rank_cols): template.at[idx, col] = zip3_cascades[row['zip3']][i]

    # Phase 3: Coverage Gap Analysis
    template['filled_count'] = template[rank_cols].count(axis=1)
    critical = template[template['filled_count'] == 0]
    insufficient = template[(template['filled_count'] > 0) & (template['filled_count'] < 3)]

    st.subheader("🚩 Coverage Gap Report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Venues", len(template))
    c2.metric("Critical (0 Coverage)", len(critical))
    c3.metric("Insufficient (< 3 Agencies)", len(insufficient))

    if not critical.empty or not insufficient.empty:
        with st.expander("View Venue Gaps"):
            gaps = pd.concat([critical, insufficient]).sort_values('filled_count')
            st.dataframe(gaps[[venue_col, city_col, zip_col, 'filled_count']], use_container_width=True)

    st.divider()
    st.subheader("✅ Ranked Cascade Preview")
    st.dataframe(template.drop(columns=['zip3', 'filled_count']), use_container_width=True)
    st.download_button("Download Final CSV", template.to_csv(index=False), "final_cascade.csv")