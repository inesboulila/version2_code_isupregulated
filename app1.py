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
            # Code 5: encoder was fitted on ONLY 3 cols (microrna, microrna_group_simplified, scenario)
            # organism and time were already numeric in X_c5 and added separately AFTER encoding.
            # The XGBoost model expects exactly 3 features: the 3 target-encoded columns.
            enc = loaded_obj['encoder']
            mdl = loaded_obj['model']

            # Step 1: Build the 3-column categorical DataFrame for the encoder
            cat_df = pd.DataFrame({
                'microrna': [p_clean],
                'microrna_group_simplified': [p_clean],
                'scenario': [f"{para}_{cell}"]
            })

            # Step 2: Transform through encoder (outputs 3 float columns)
            X_encoded = enc.transform(cat_df)

            # Step 3: Check how many features the model actually expects
            # If model expects 3, use only encoded cols; if 5, append organism + time
            n_expected = mdl.n_features_in_ if hasattr(mdl, 'n_features_in_') else X_encoded.shape[1]

            if n_expected == 5:
                X_encoded['organism'] = float(org_num)
                X_encoded['time'] = float(time_hours)
            # else n_expected == 3, use X_encoded as-is

            prediction = mdl.predict(X_encoded)[0]
            probability = mdl.predict_proba(X_encoded)[0][1]

            # Skip the generic execution block below
            input_df = None

        elif v_num == 1:
            # FIX: Code 1 used manual one-hot encoding on parasite + cell type,
            # then target-encoded microrna + microrna_group_simplified.
            # We must reconstruct the full encoded row manually.
            enc = loaded_obj['encoder']
            mdl = loaded_obj['model']

            # Get the exact column list the model was trained on
            trained_cols = list(mdl.feature_names_in_)

            # Build a zero-row matching all trained columns
            input_row = {col: [0] for col in trained_cols}

            # Fill target-encoded columns via the encoder
            mirna_df = pd.DataFrame({
                'microrna': [p_clean],
                'microrna_group_simplified': [p_clean]
            })
            mirna_encoded = enc.transform(mirna_df)
            input_row['microrna'] = [mirna_encoded['microrna'].iloc[0]]
            input_row['microrna_group_simplified'] = [mirna_encoded['microrna_group_simplified'].iloc[0]]

            # Fill organism and time directly
            if 'organism' in input_row:
                input_row['organism'] = [float(org_num)]
            if 'time' in input_row:
                input_row['time'] = [float(time_hours)]

            # Fill one-hot parasite column (e.g. parasite_l.donovani)
            para_col = f"parasite_{para}"
            if para_col in input_row:
                input_row[para_col] = [1]

            # Fill one-hot cell type column (e.g. cell type_pbmc)
            cell_col = f"cell type_{cell}"
            if cell_col in input_row:
                input_row[cell_col] = [1]

            X_encoded = pd.DataFrame(input_row)[trained_cols]

            prediction = mdl.predict(X_encoded)[0]
            probability = mdl.predict_proba(X_encoded)[0][1]

            # Skip the generic execution block below
            input_df = None

        elif v_num in [2, 3, 4]:
            input_df = pd.DataFrame({
                'microrna': [p_clean], 'microrna_group_simplified': [p_clean],
                'parasite': [para], 'cell type': [cell], 'organism': [float(org_num)], 'time': [time_hours]
            })

        # GENERIC EXECUTION (for pipeline-based models: 2, 3, 4, 6, 7, 8, 9, 10)
        if input_df is not None:
            if isinstance(loaded_obj, dict):
                enc = loaded_obj['encoder']
                mdl = loaded_obj['model']
                X_encoded = enc.transform(input_df)
                if hasattr(mdl, "feature_names_in_"):
                    X_encoded = X_encoded[mdl.feature_names_in_]
                prediction = mdl.predict(X_encoded)[0]
                probability = mdl.predict_proba(X_encoded)[0][1]
            else:
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
            import traceback
            st.code(traceback.format_exc())
