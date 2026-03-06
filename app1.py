import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="miRNA Evolution Lab", layout="wide", page_icon="🧬")

# --- 2. HELPER FUNCTIONS ---
def strip_prefix(name):
    """Used for Code 9 and 10 to prevent AI cheating."""
    return re.sub(r'^[a-z]{3}-', '', str(name).lower().strip())

def get_time_bin(hours):
    """Used for Code 6 and 7."""
    if hours <= 7: return 'early'
    if hours <= 13: return 'mid'
    return 'late'

# --- 3. SIDEBAR: MODEL SELECTOR ---
st.sidebar.title("🚀 Project History")
version_choice = st.sidebar.selectbox(
    "Select Model Version",
    [f"Code {i}" for i in range(1, 11)],
    index=9  # Default to Code 10 (Final Dog Model)
)

# Descriptions for the report
code_desc = {
    "Code 1": "Baseline Random Forest",
    "Code 2": "Pipeline Implementation",
    "Code 3": "GridSearchCV Optimized",
    "Code 4": "Hierarchical RF (Leaf Limit)",
    "Code 5": "Target Leakage Mirage (XGBoost/RF)",
    "Code 6": "Time Binning Strategy",
    "Code 7": "Interaction Constraints",
    "Code 8": "Scenario Merging (Parasite+Cell)",
    "Code 9": "Biological Blinding (No Prefixes)",
    "Code 10": "Final Super-Scenario (3 Organisms)"
}
st.sidebar.info(f"**Method:** {code_desc[version_choice]}")

# --- 4. MAIN INTERFACE ---
st.title("🧬 miRNA Upregulation Predictor")
st.write(f"Currently testing: **{version_choice}** logic.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        mirna_raw = st.text_input("miRNA Name", "hsa-mir-21-5p")
        organism = st.selectbox("Organism", ["human", "mouse", "dog"])
        parasite = st.selectbox("Parasite", ["l.donovani", "l.major", "l.infantum", "l.amazonensis"])
    
    with col2:
        cell_type = st.selectbox("Cell Type", ["pbmc", "thp-1", "bmdm", "raw264.7", "hmdm"])
        time_hours = st.number_input("Time (Hours)", min_value=1, value=12)
        infection = st.selectbox("Infection Type", ["in vitro", "natural"])

    submit = st.form_submit_button("Run Prediction")

# --- 5. PREDICTION ENGINE ---
if submit:
    try:
        # Load the specific model file
        v_num = int(version_choice.split(" ")[1])
        model_path = f"model_code_{v_num}.pkl"
        loaded_obj = joblib.load(model_path)
        
        # PRE-PROCESSING
        p_name = mirna_raw.lower().strip()
        # Only strip prefixes for Code 9 and 10
        if v_num >= 9:
            p_name = strip_prefix(p_name)
            
        para_clean = parasite.lower().replace(" ", "")
        cell_clean = cell_type.lower().replace(" ", "")
        org_clean = organism.lower()
        
        # DYNAMIC DATAFRAME CONSTRUCTION
        # This creates the exact columns each specific code expects
        if v_num == 10:
            super_scenario = f"{para_clean}_{cell_clean}_{org_clean}"
            input_df = pd.DataFrame({
                'microrna': [p_name],
                'microrna_group_simplified': [p_name],
                'super_scenario': [super_scenario],
                'infection': [infection.lower()],
                'time': [time_hours]
            })
        elif v_num in [8, 9]:
            scenario = f"{para_clean}_{cell_clean}"
            input_df = pd.DataFrame({
                'microrna': [p_name],
                'microrna_group_simplified': [p_name],
                'scenario': [scenario],
                'organism': [org_clean],
                'time': [time_hours]
            })
        elif v_num in [6, 7]:
            scenario = f"{para_clean}_{cell_clean}"
            input_df = pd.DataFrame({
                'microrna': [p_name],
                'microrna_group_simplified': [p_name],
                'scenario': [scenario],
                'organism': [org_clean],
                'time_bin': [get_time_bin(time_hours)]
            })
        else:
            # Codes 1-5 expect raw columns
            input_df = pd.DataFrame({
                'microrna': [p_name],
                'microrna_group_simplified': [p_name],
                'parasite': [para_clean],
                'cell type': [cell_clean],
                'organism': [org_clean],
                'time': [time_hours]
            })

        # PREDICTION LOGIC
        if isinstance(loaded_obj, dict):
            # For Manual Pair models (1, 5, 6, 7)
            encoder = loaded_obj['encoder']
            model = loaded_obj['model']
            X_encoded = encoder.transform(input_df)
            prediction = model.predict(X_encoded)[0]
            probability = model.predict_proba(X_encoded)[0][1]
        else:
            # For Pipeline models (2, 3, 4, 8, 9, 10)
            prediction = loaded_obj.predict(input_df)[0]
            probability = loaded_obj.predict_proba(input_df)[0][1]

        # RESULTS DISPLAY
        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == 1:
                st.success("### Result: UPREGULATED (1)")
            else:
                st.error("### Result: DOWNREGULATED (0)")
        
        with res_col2:
            st.metric("Confidence Score", f"{probability*100:.1f}%")
        
        st.info(f"Model Processing Log: Name interpreted as '{p_name}'")

    except FileNotFoundError:
        st.error(f"Missing File: {model_path} not found.")
    except Exception as e:
        st.error(f"Logic Error: {e}")