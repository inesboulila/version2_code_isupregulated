import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
import category_encoders
import xgboost as xgb

# --- 1. PAGE CONFIG & HELPERS ---
st.set_page_config(page_title="miRNA Evolution Lab", layout="wide", page_icon="🧬")

def strip_prefix(name):
    return re.sub(r'^[a-z]{3}-', '', str(name).lower().strip())

def get_time_bin(hours):
    if hours <= 7: return 'early'
    if hours <= 13: return 'mid'
    return 'late'

def clean_text(text):
    return str(text).lower().replace(" ", "").strip()

# --- 2. SIDEBAR ---
st.sidebar.title("🚀 Model Selector")
v_choice = st.sidebar.selectbox(
    "Select Research Version", 
    [f"Code {i}" for i in range(1, 11)], 
    index=9
)
v_num = int(v_choice.split(" ")[1])

# --- 3. DYNAMIC INPUT UI ---
st.title(f"🧬 miRNA Prediction Interface")
st.write(f"Current Model: **{v_choice}**")

with st.form("mirna_form"):
    c1, c2 = st.columns(2)
    
    with c1:
        mirna_raw = st.text_input("miRNA Name", "hsa-mir-21-5p")
        organism = st.selectbox("Organism", ["Human", "Mouse", "Dog"])
        parasite = st.selectbox("Parasite", ["L. donovani", "L. major", "L. infantum", "L. amazonensis"])
    
    with c2:
        cell_type = st.selectbox("Cell Type", ["PBMC", "THP-1", "BMDM", "RAW 264.7", "HMDM"])
        time_hours = st.number_input("Time (Hours)", min_value=1, value=12)
        
        # HIDE INFECTION for Codes 1-9
        if v_num == 10:
            inf_display = st.selectbox("Infection Type", ["In Vitro", "Naturally Infected"])
        else:
            inf_display = "In Vitro" # Default hidden value

    submit = st.form_submit_button("Predict Result")

# --- 4. UNIVERSAL LOGIC ENGINE ---
if submit:
    try:
        model_path = f"model_code_{v_num}.pkl"
        loaded_obj = joblib.load(model_path)
        
        # Normalize inputs
        p_clean = mirna_raw.lower().strip()
        p_blind = strip_prefix(p_clean)
        para = clean_text(parasite)
        cell = clean_text(cell_type)
        org = organism.lower()
        org_num = 1 if org == "human" else 0
        inf = "naturallyinfected" if inf_display == "Naturally Infected" else "invitro"

        # BUILD THE DATAFRAME
        if v_num == 10:
            input_df = pd.DataFrame({
                'microrna': [p_blind], 'microrna_group_simplified': [p_blind],
                'super_scenario': [f"{para}_{cell}_{org}"], 'infection': [inf], 'time': [time_hours]
            })
        
        elif v_num in [8, 9]:
            name = p_blind if v_num == 9 else p_clean
            input_df = pd.DataFrame({
                'microrna': [name], 'microrna_group_simplified': [name],
                'scenario': [f"{para}_{cell}"], 'organism_num': [org_num], 'time': [time_hours]
            })

        elif v_num in [6, 7]:
            input_df = pd.DataFrame({
                'microrna': [p_clean], 'microrna_group_simplified': [p_clean],
                'scenario': [f"{para}_{cell}"], 'organism': [float(org_num)], 'time_bin': [get_time_bin(time_hours)]
            })

        elif v_num == 5:
            # Code 5 trained on ONLY 3 columns (Encoded Name, Group, Scenario)
            # Organism and Time were NOT part of the model input
            input_df = pd.DataFrame({
                'microrna': [p_clean], 'microrna_group_simplified': [p_clean],
                'scenario': [f"{para}_{cell}"]
            })

        elif v_num == 1:
            # Code 1 trained on ONLY 2 columns (Encoded Name and Group)
            input_df = pd.DataFrame({
                'microrna': [p_clean], 'microrna_group_simplified': [p_clean]
            })

        elif v_num in [2, 3, 4]:
            input_df = pd.DataFrame({
                'microrna': [p_clean], 'microrna_group_simplified': [p_clean],
                'parasite': [para], 'cell type': [cell], 'organism': [float(org_num)], 'time': [time_hours]
            })

        # EXECUTION
        if isinstance(loaded_obj, dict):
            # For Codes 1, 5, 6
            enc = loaded_obj['encoder']
            mdl = loaded_obj['model']
            
            # Encoder needs the columns it was trained on
            # We add dummy columns just for the encoder to run, then drop them
            X_encoded = enc.transform(input_df)
            
            # Ensure we only send the features the model expects
            if hasattr(mdl, "feature_names_in_"):
                X_encoded = X_encoded[mdl.feature_names_in_]
            
            prediction = mdl.predict(X_encoded)[0]
            probability = mdl.predict_proba(X_encoded)[0][1]
        else:
            # For Pipelines 2, 3, 4, 7, 8, 9, 10
            prediction = loaded_obj.predict(input_df)[0]
            probability = loaded_obj.predict_proba(input_df)[0][1]

        # --- 5. RESULTS DISPLAY ---
        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == 1: st.success(f"### RESULT: UPREGULATED")
            else: st.error(f"### RESULT: DOWNREGULATED")
        with res_col2:
            st.metric("AI Confidence", f"{probability*100:.1f}%")
            st.progress(float(probability))

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        with st.expander("Technical Log"):
            st.write("Columns Sent:", input_df.columns.tolist())
            st.write("Expected by model:", loaded_obj['model'].feature_names_in_ if isinstance(loaded_obj, dict) else "Pipeline handles logic")
