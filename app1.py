import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
import category_encoders
import xgboost as xgb

st.set_page_config(page_title="miRNA Prediction Evolution", layout="wide", page_icon="🧬")

# --- HELPERS ---
def strip_prefix(name):
    return re.sub(r'^[a-z]{3}-', '', str(name).lower().strip())

def get_time_bin(hours):
    if hours <= 7: return 'early'
    if hours <= 13: return 'mid'
    return 'late'

def clean_text(text):
    return str(text).lower().replace(" ", "").strip()

# --- SIDEBAR ---
v_choice = st.sidebar.selectbox("Select Model Version", [f"Code {i}" for i in range(1, 11)], index=9)
v_num = int(v_choice.split(" ")[1])

# --- INPUT FORM ---
st.title(f"🧬 miRNA Upregulation Predictor")
with st.form("mirna_form"):
    c1, c2 = st.columns(2)
    with c1:
        mirna_raw = st.text_input("miRNA Name", "hsa-mir-21-5p")
        organism = st.selectbox("Organism", ["Human", "Mouse", "Dog"])
        parasite = st.selectbox("Parasite", ["L. donovani", "L. major", "L. infantum", "L. amazonensis"])
    with c2:
        cell_type = st.selectbox("Cell Type", ["PBMC", "THP-1", "BMDM", "RAW 264.7", "HMDM"])
        time_hours = st.number_input("Time (Hours)", min_value=1, value=12)
        inf_display = st.selectbox("Infection Type", ["In Vitro", "Naturally Infected"])
    submit = st.form_submit_button("Run Prediction")

if submit:
    try:
        model_path = f"model_code_{v_num}.pkl"
        loaded_obj = joblib.load(model_path)
        
        # 1. STANDARDIZE INPUTS
        p_clean = mirna_raw.lower().strip()
        p_blind = strip_prefix(p_clean)
        para = clean_text(parasite)
        cell = clean_text(cell_type)
        org = organism.lower()
        org_num = 1 if org == "human" else 0
        inf = "naturallyinfected" if inf_display == "Naturally Infected" else "invitro"
        
        # 2. BUILD VERSION-SPECIFIC DATAFRAMES
        if v_num == 10:
            input_df = pd.DataFrame({
                'microrna': [p_blind], 'microrna_group_simplified': [p_blind],
                'super_scenario': [f"{para}_{cell}_{org}"], 'infection': [inf], 'time': [time_hours]
            })
        
        elif v_num in [8, 9]:
            # FIX for Codes 8 & 9: Column must be named 'organism_num'
            name = p_blind if v_num == 9 else p_clean
            input_df = pd.DataFrame({
                'microrna': [name], 'microrna_group_simplified': [name],
                'scenario': [f"{para}_{cell}"], 'organism_num': [org_num], 'time': [time_hours]
            })

        elif v_num in [6, 7]:
            # FIX for Codes 6 & 7: Column name varies, using 'organism' as numeric
            input_df = pd.DataFrame({
                'microrna': [p_clean], 'microrna_group_simplified': [p_clean],
                'scenario': [f"{para}_{cell}"], 'organism': [float(org_num)], 'time_bin': [get_time_bin(time_hours)]
            })

        elif v_num == 5:
            # FIX for Code 5: Model only expected 3 features (Encoded ones)
            # We create the full DF, but the dictionary logic will handle the encoding
            input_df = pd.DataFrame({
                'microrna': [p_clean], 'microrna_group_simplified': [p_clean],
                'scenario': [f"{para}_{cell}"], 'organism': [float(org_num)], 'time': [time_hours]
            })
            # Code 5 logic specifically requires only these 5 columns in this order:
            input_df = input_df[['microrna', 'microrna_group_simplified', 'scenario', 'organism', 'time']]

        elif v_num in [1, 2, 3, 4]:
            # FIX for Codes 1-4: Standardize to Numeric Organism to prevent 'human' string error
            input_df = pd.DataFrame({
                'microrna': [p_clean], 'microrna_group_simplified': [p_clean],
                'parasite': [para], 'cell type': [cell], 'organism': [float(org_num)], 'time': [time_hours]
            })

        # 3. PREDICTION ENGINE
        if isinstance(loaded_obj, dict):
            # For Codes 1, 5, 6
            enc = loaded_obj['encoder']
            mdl = loaded_obj['model']
            
            # Select only columns the specific encoder was trained on
            X_encoded = enc.transform(input_df)
            
            # DIMENSION FIX: If model expects 2 columns but gets 6 (Error in Code 1/5)
            # We force it to use only the columns it was trained on
            if hasattr(mdl, "feature_names_in_"):
                X_encoded = X_encoded[mdl.feature_names_in_]
                
            prediction = mdl.predict(X_encoded)[0]
            probability = mdl.predict_proba(X_encoded)[0][1]
        else:
            # For Pipelines 2, 3, 4, 7, 8, 9, 10
            prediction = loaded_obj.predict(input_df)[0]
            probability = loaded_obj.predict_proba(input_df)[0][1]

        # 4. RESULTS
        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == 1: st.success("### RESULT: UPREGULATED")
            else: st.error("### RESULT: DOWNREGULATED")
        with res_col2:
            st.metric("Confidence Level", f"{probability*100:.1f}%")
            st.progress(float(probability))

    except Exception as e:
        st.error(f"Logic Error: {e}")
        with st.expander("Show Detailed Debug Info"):
            st.write("Columns Sent:", input_df.columns.tolist())
            st.write("Data Types:", input_df.dtypes)
            st.write("Full Error:", e)
