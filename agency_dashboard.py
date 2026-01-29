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
# 2. Unification Logic
# ------------------------------------------------
def unify_agency_names(names_list):
    def get_clean_key(s):
        s = str(s).lower()
        return re.sub(r"[^a-z0-9]", "", s).strip()

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

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# ------------------------------------------------
# 3. Data Loading & Standardization
# ------------------------------------------------
st.sidebar.header("📂 Upload Data Files")
agency_file = st.sidebar.file_uploader("Upload Agency Coverage CSV", type=["csv"], key="agency_csv")
ratecard_file = st.sidebar.file_uploader("Upload Rate Card CSV", type=["csv"], key="ratecard_csv")

if agency_file is None or ratecard_file is None:
    st.info("Please upload both CSV files to begin.")
    st.stop()

a_df = load_csv(agency_file)
r_df = load_csv(ratecard_file)

def standardize_columns(df):
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    mapping = {
        'brand': 'agency_name', 'vendor': 'agency_name', 'agency': 'agency_name',
        'venue_city': 'city', 'location': 'city', 'market': 'city',
        'employer_id': 'platforms.employer_id', 'venue': 'venue_name'
    }
    return df.rename(columns=mapping)

a_df = standardize_columns(a_df)
r_df = standardize_columns(r_df)

# ------------------------------------------------
# 4. Processing
# ------------------------------------------------
all_names = pd.concat([a_df["agency_name"], r_df["agency_name"]]).unique()
master_map = unify_agency_names(all_names)

a_df["agency_name"] = a_df["agency_name"].map(master_map)
r_df["agency_name"] = r_df["agency_name"].map(master_map)

combined = pd.concat([a_df, r_df], ignore_index=True)
combined["city"] = combined["city"].fillna("Unknown").str.strip().str.title()
combined["agency_margin"] = pd.to_numeric(combined.get("agency_margin", np.nan), errors="coerce")

# Calculate city metrics
city_avg_map = combined.groupby("city")["agency_margin"].mean().to_dict()
combined["city_avg_margin"] = combined["city"].map(city_avg_map)
combined["margin_vs_city_avg"] = combined["agency_margin"] - combined["city_avg_margin"]

# Global aggregation for clean display
agg_logic = {
    "agency_margin": "mean",
    "city_avg_margin": "mean",
    "margin_vs_city_avg": "mean"
}
optional_cols = ["role_category", "supply_capability", "platforms.employer_id", "venue_name"]
for col in optional_cols:
    if col in combined.columns:
        agg_logic[col] = lambda x: ", ".join(sorted(set([str(i).strip() for i in x if pd.notna(i) and str(i).strip() != ""])))

master_display_df = combined.groupby(["agency_name", "city"], as_index=False).agg(agg_logic)

# Presence Type Calculation
has_role = "role_category" in master_display_df.columns
master_display_df["presence_type"] = np.where(
    master_display_df["agency_margin"].notna() & (master_display_df["role_category"] != "" if has_role else False), 
    "Rate Card + Coverage",
    np.where(master_display_df["agency_margin"].notna(), "Rate Card Only", "Coverage Only")
)

# ------------------------------------------------
# 5. Navigation & Filters
# ------------------------------------------------
st.sidebar.header("View Control")
view = st.sidebar.radio("Mode", ["Coverage View", "Agency View", "City View", "Client View"])

all_cities = sorted(master_display_df["city"].unique())
all_agencies = sorted(master_display_df["agency_name"].unique())
all_clients = sorted(combined["platforms.employer_id"].dropna().unique()) if "platforms.employer_id" in combined.columns else []

selected_city = st.sidebar.multiselect("Filter City", ["All"] + all_cities, default=["All"])
selected_agency = st.sidebar.multiselect("Filter Agency", ["All"] + all_agencies, default=["All"])
selected_client = st.sidebar.selectbox("Select Client (for Client View)", ["All"] + all_clients)

df_filt = master_display_df.copy()
if "All" not in selected_city: df_filt = df_filt[df_filt["city"].isin(selected_city)]
if "All" not in selected_agency: df_filt = df_filt[df_filt["agency_name"].isin(selected_agency)]

# ------------------------------------------------
# 6. Views
# ------------------------------------------------
if view == "Coverage View":
    st.subheader("Market Coverage Overview")
    cols = ["agency_name", "city", "presence_type", "agency_margin", "city_avg_margin", "margin_vs_city_avg"]
    if "role_category" in df_filt.columns: cols.append("role_category")
    if "supply_capability" in df_filt.columns: cols.append("supply_capability")
    st.dataframe(df_filt[cols].sort_values(["city", "agency_name"]), use_container_width=True)

elif view == "Agency View":
    st.subheader("Agency Performance")
    if len(selected_agency) != 1 or selected_agency[0] == "All":
        st.warning("Please select one specific agency.")
    else:
        a_data = df_filt[df_filt["agency_name"] == selected_agency[0]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Cities", a_data["city"].nunique())
        c2.metric("Avg Margin", round(a_data["agency_margin"].mean(), 2) if not a_data["agency_margin"].dropna().empty else "N/A")
        c3.metric("Vs Market", round(a_data["margin_vs_city_avg"].mean(), 2) if not a_data["margin_vs_city_avg"].dropna().empty else "N/A")
        st.dataframe(a_data.drop(columns=["agency_name"]), use_container_width=True)

elif view == "City View":
    if len(selected_city) != 1 or selected_city[0] == "All":
        st.subheader("City Market View")
        st.warning("Please select a single city to see detailed metrics.")
    else:
        city_name = selected_city[0]
        st.subheader(f"Market Snapshot: {city_name}")
        city_data = df_filt[df_filt["city"] == city_name]
        
        # At-a-glance metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg City Margin", round(city_data["agency_margin"].mean(), 2) if not city_data["agency_margin"].dropna().empty else "N/A")
        
        # Calculate unique venues and clients specifically for this city
        raw_city_data = combined[combined["city"] == city_name]
        num_venues = raw_city_data["venue_name"].nunique() if "venue_name" in raw_city_data.columns else 0
        clients_in_city = sorted(raw_city_data["platforms.employer_id"].dropna().unique())
        
        c2.metric("Number of Venues", num_venues)
        c3.markdown(f"**Clients in Market:**\n {', '.join([str(c) for c in clients_in_city]) if clients_in_city else 'None Found'}")
        
        st.divider()
        st.dataframe(city_data[["agency_name", "presence_type", "agency_margin", "role_category"]].sort_values("agency_margin"), use_container_width=True)

elif view == "Client View":
    if selected_client == "All":
        st.subheader("Client Portfolio View")
        st.warning("Please select a specific Client ID from the sidebar.")
        client_pivot = master_display_df.pivot_table(index="city", columns="agency_name", values="agency_margin", aggfunc="mean").fillna("-")
        st.dataframe(client_pivot, use_container_width=True)
    else:
        st.subheader(f"Client Venue Breakdown: {selected_client}")
        # Filter combined data for specific client
        client_data = combined[combined["platforms.employer_id"].astype(str) == str(selected_client)]
        
        if client_data.empty:
            st.info("No venue or agency data found for this client ID.")
        else:
            # Group by venue to show supporting agencies and their margins
            venue_breakdown = client_data.groupby(["venue_name", "city", "agency_name"], as_index=False).agg({
                "agency_margin": "mean"
            }).rename(columns={"agency_margin": "Avg Margin at Venue"})
            
            st.dataframe(venue_breakdown.sort_values(["city", "venue_name"]), use_container_width=True)

# Export
st.sidebar.markdown("---")
st.sidebar.download_button("Export Results", master_display_df.to_csv(index=False), "agency_intelligence.csv")