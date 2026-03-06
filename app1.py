import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
import category_encoders
import xgboost as xgb

st.set_page_config(page_title="miRNA Evolution Lab", layout="wide", page_icon="🧬")

# --- 1. HELPERS ---
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
v_choice = st.sidebar.selectbox("Select Version", [f"Code {i}" for i in range(1, 11)], index=9)
v_num = int(v_choice.split(" ")[1])

# --- 3. INPUT UI ---
st.title(f"🧬 miRNA Prediction interface")
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
    submit = st.form_submit_button("Predict Result")

# --- 4. THE SELF-HEALING ENGINE ---
if submit:
    try:
        model_path = f"model_code_{v_num}.pkl"
        loaded_obj = joblib.load(model_path)
        
        # A. PREPARE RAW PIECES
        p_clean = mirna_raw.lower().strip()
        p_blind = strip_prefix(p_clean)
        para = clean_text(parasite)
        cell = clean_text(cell_type)
        org = organism.lower()
        org_num = 1.0 if org == "human" else 0.0
        if org == "dog": org_num = 2.0
        inf = "naturallyinfected" if inf_display == "Naturally Infected" else "invitro"

        # B. THE "SUPER-DATAFRAME"
        # We include EVERY variation of every column name we used in the 10 codes.
        # If Code 1 wants 'parasite' and Code 9 wants 'scenario', we provide BOTH.
        master_dict = {
            # MicroRNA variations
            'microrna': p_blind if v_num >= 9 else p_clean,
            'microrna_group_simplified': p_blind if v_num >= 9 else p_clean,
            'miRNA': p_blind if v_num >= 9 else p_clean, # In case you capitalized it
            
            # Context variations
            'parasite': para,
            'cell type': cell,
            'cell_type': cell,
            'organism': org if v_num <= 4 else org_num, # Text for early, Numeric for late
            'organism_num': org_num,
            'infection': inf,
            'time': float(time_hours),
            'time_bin': get_time_bin(time_hours),
            
            # Scenario variations
            'scenario': f"{para}_{cell}",
            'super_scenario': f"{para}_{cell}_{org}"
        }
        
        full_input_df = pd.DataFrame([master_dict])

        # C. THE AUTOMATIC FILTER
        # This part asks the model: "What columns do you want?" and gives it ONLY those.
        if isinstance(loaded_obj, dict):
            # Manual Codes (1, 5, 6, 7)
            encoder = loaded_obj['encoder']
            model = loaded_obj['model']
            
            # 1. Let the encoder transform the super-dataframe
            # (Encoder will ignore columns it wasn't fitted on)
            X_encoded = encoder.transform(full_input_df)
            
            # 2. Re-add numeric columns that encoder might have skipped
            X_encoded['organism'] = org_num
            X_encoded['organism_num'] = org_num
            X_encoded['time'] = float(time_hours)

            # 3. IF Code 1 expects One-Hot columns (parasite_l.major), we generate them
            if v_num == 1:
                one_hot_parts = pd.get_dummies(full_input_df[['parasite', 'cell type']])
                X_encoded = pd.concat([X_encoded, one_hot_parts], axis=1)

            # 4. FILTER: Look at 'feature_names_in_' of the model
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
                # Fill any missing one-hot columns with 0
                for col in expected_features:
                    if col not in X_encoded.columns:
                        X_encoded[col] = 0.0
                # Select and Reorder
                X_final = X_encoded[expected_features]
            else:
                X_final = X_encoded

            prediction = model.predict(X_final)[0]
            probability = model.predict_proba(X_final)[0][1]
            
        else:
            # Pipeline Codes (2, 3, 4, 8, 9, 10)
            # Pipelines are smarter. We just filter to what it expects.
            if hasattr(loaded_obj, "feature_names_in_"):
                expected_features = list(loaded_obj.feature_names_in_)
                X_final = full_input_df[expected_features]
            else:
                X_final = full_input_df
                
            prediction = loaded_obj.predict(X_final)[0]
            probability = loaded_obj.predict_proba(X_final)[0][1]

        # D. SHOW RESULTS
        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == 1: st.success("### Prediction: UPREGULATED")
            else: st.error("### Prediction: DOWNREGULATED")
        with res_col2:
            st.metric("Confidence Score", f"{probability*100:.1f}%")
            st.progress(float(probability))

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        with st.expander("Show Logic Breakdown (Debug)"):
            st.write("Model type:", "Manual Dict" if isinstance(loaded_obj, dict) else "Pipeline")
            try:
                st.write("Columns the model is looking for:", list(model.feature_names_in_) if isinstance(loaded_obj, dict) else list(loaded_obj.feature_names_in_))
            except:
                st.write("Could not detect model feature names.")
            st.write("Data sent to predictor:", X_final if 'X_final' in locals() else "None")
