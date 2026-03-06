import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
import category_encoders
import xgboost as xgb

st.set_page_config(page_title="miRNA Predictor", layout="wide", page_icon="🧬")

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
v_choice = st.sidebar.selectbox("Select Research Version", [f"Code {i}" for i in range(1, 11)], index=9)
v_num = int(v_choice.split(" ")[1])

# --- DYNAMIC UI ---
st.title(f"🧬 miRNA Prediction")
with st.form("mirna_form"):
    c1, c2 = st.columns(2)
    with c1:
        mirna_raw = st.text_input("miRNA Name", "hsa-mir-21-5p")
        organism = st.selectbox("Organism", ["Human", "Mouse", "Dog"])
        parasite = st.selectbox("Parasite", ["L. donovani", "L. major", "L. infantum", "L. amazonensis"])
    with c2:
        cell_type = st.selectbox("Cell Type", ["PBMC", "THP-1", "BMDM", "RAW 264.7", "HMDM"])
        time_hours = st.number_input("Time (Hours)", min_value=1, value=12)
        if v_num == 10:
            inf_display = st.selectbox("Infection Type", ["In Vitro", "Naturally Infected"])
        else:
            inf_display = "In Vitro"
    submit = st.form_submit_button("Predict")

if submit:
    try:
        model_path = f"model_code_{v_num}.pkl"
        loaded_obj = joblib.load(model_path)
        
        # 1. PREPARE RAW DATA
        p_clean = mirna_raw.lower().strip()
        p_blind = strip_prefix(p_clean)
        para = clean_text(parasite)
        cell = clean_text(cell_type)
        org = organism.lower()
        org_num = 1 if org == "human" else 0
        inf = "naturallyinfected" if inf_display == "Naturally Infected" else "invitro"
        
        # 2. CREATE EVERY POSSIBLE COLUMN NAME COMBINATION
        # We create a huge dictionary so no matter what the model wants, we have it
        data_map = {
            'microrna': p_blind if v_num >= 9 else p_clean,
            'microrna_group_simplified': p_blind if v_num >= 9 else p_clean,
            'parasite': para,
            'cell type': cell,
            'organism': float(org_num),
            'organism_num': float(org_num),
            'time': float(time_hours),
            'infection': inf,
            'scenario': f"{para}_{cell}",
            'super_scenario': f"{para}_{cell}_{org}",
            'time_bin': get_time_bin(time_hours)
        }
        
        input_df = pd.DataFrame([data_map])

        # 3. PREDICTION ENGINE
        if isinstance(loaded_obj, dict):
            # For Manual Models (1, 5, 6)
            enc = loaded_obj['encoder']
            mdl = loaded_obj['model']
            
            # Step A: Transform using encoder
            # (Encoder will only transform columns it recognizes)
            X_encoded = enc.transform(input_df)
            
            # Step B: HANDLE ONE-HOT DUMMIES (The fix for your Code 1 screenshot)
            # If the model wants things like 'parasite_l.major', we must create them
            if v_num == 1:
                # Re-create dummies for parasite and cell type
                dummies = pd.get_dummies(input_df, columns=['parasite', 'cell type'])
                # Merge dummies with the encoded columns
                X_encoded = pd.concat([X_encoded, dummies.drop(columns=enc.cols, errors='ignore')], axis=1)

            # Step C: CRITICAL DIMENSION FIX
            # We force X_encoded to have ONLY the columns the model was trained on
            # This fixes "Expected 2, got 6" and "Expected 3, got 5"
            if hasattr(mdl, "feature_names_in_"):
                expected = list(mdl.feature_names_in_)
                # Fill missing columns with 0 (like rare cell types)
                for col in expected:
                    if col not in X_encoded.columns:
                        X_encoded[col] = 0.0
                # Filter and reorder exactly
                X_encoded = X_encoded[expected]
                
            prediction = mdl.predict(X_encoded)[0]
            probability = mdl.predict_proba(X_encoded)[0][1]
        else:
            # For Pipelines (2, 3, 4, 7, 8, 9, 10)
            # We must still ensure column names match the Pipeline's feature_names_in_
            expected = list(loaded_obj.feature_names_in_)
            # Sub-select only the columns this pipeline needs
            pipeline_input = input_df[expected]
            
            prediction = loaded_obj.predict(pipeline_input)[0]
            probability = loaded_obj.predict_proba(pipeline_input)[0][1]

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
        st.error(f"Prediction Error: {e}")
        with st.expander("Technical Debugging Info"):
            st.write("Model Type:", "Dictionary" if isinstance(loaded_obj, dict) else "Pipeline")
            if isinstance(loaded_obj, dict):
                st.write("Model Expected Columns:", list(loaded_obj['model'].feature_names_in_))
            else:
                st.write("Pipeline Expected Columns:", list(loaded_obj.feature_names_in_))
            st.write("Columns App Sent:", X_encoded.columns.tolist() if 'X_encoded' in locals() else input_df.columns.tolist())
