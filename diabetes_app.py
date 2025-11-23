import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. Load Artifacts
# ==========================================
@st.cache_resource
def load_model_data():
    try:
        model = joblib.load('diabetes_model_v2.pkl')
        features = joblib.load('model_features_v2.pkl')
        explainer = joblib.load('shap_explainer_v2.pkl')
        return model, features, explainer
    except FileNotFoundError:
        return None, None, None

model, model_features, explainer = load_model_data()

# ==========================================
# 2. App Interface
# ==========================================
st.set_page_config(page_title="Diabetes Risk AI (Extended)", layout="wide")

if model is None:
    st.error("Error: Model files not found. Please run 'train_model.py' first.")
    st.stop()

st.title(" Comprehensive Diabetes Risk Predictor")
st.markdown("This tool analyzes **26 different health markers** to provide a personalized risk assessment.")
st.info("Please fill out all the sections below for the most accurate prediction.")

# Dictionary to store inputs
user_input = {}

# --- SECTION 1: PERSONAL DETAILS ---
st.header("1. Personal Details")
c1, c2, c3 = st.columns(3)
with c1:
    user_input['Age'] = st.number_input("Age", 0, 120, 30)
    user_input['Gender'] = st.selectbox("Gender", ["Male", "Female", "Other"])
with c2:
    user_input['BMI'] = st.number_input("BMI", 10.0, 60.0, 25.0)
    user_input['Waist_Hip_Ratio'] = st.number_input("Waist/Hip Ratio", 0.5, 2.0, 0.90)
with c3:
    user_input['Urban_Rural'] = st.selectbox("Residence", ["Urban", "Rural"])
    user_input['Health_Insurance'] = st.selectbox("Has Insurance?", ["No", "Yes"])

st.divider()

# --- SECTION 2: MEDICAL HISTORY ---
st.header("2. Medical History")
c1, c2, c3 = st.columns(3)
with c1:
    user_input['Family_History'] = st.selectbox("Family History of Diabetes", ["No", "Yes"])
    user_input['Hypertension'] = st.selectbox("Has Hypertension?", ["No", "Yes"])
    user_input['Heart_Rate'] = st.number_input("Heart Rate (bpm)", 40, 200, 72)
with c2:
    user_input['Pregnancies'] = st.number_input("Pregnancies", 0, 20, 0)
    user_input['Polycystic_Ovary_Syndrome'] = st.selectbox("PCOS?", ["No", "Yes"])
with c3:
    user_input['Thyroid_Condition'] = st.selectbox("Thyroid Condition?", ["No", "Yes"])
    user_input['Medication_For_Chronic_Conditions'] = st.selectbox("Taking Chronic Meds?", ["No", "Yes"])

st.divider()

# --- SECTION 3: BLOOD & LAB RESULTS ---
st.header("3. Blood & Lab Results")
c1, c2, c3 = st.columns(3)
with c1:
    user_input['Fasting_Blood_Sugar'] = st.number_input("Fasting Sugar (mg/dL)", 50.0, 300.0, 100.0)
    user_input['Postprandial_Blood_Sugar'] = st.number_input("Post-Meal Sugar (mg/dL)", 50.0, 400.0, 140.0)
    user_input['HBA1C'] = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
with c2:
    user_input['Cholesterol_Level'] = st.number_input("Cholesterol", 100.0, 400.0, 200.0)
    user_input['Glucose_Tolerance_Test_Result'] = st.number_input("Glucose Tolerance Test", 50.0, 300.0, 120.0)
    user_input['C_Protein_Level'] = st.number_input("C-Reactive Protein", 0.0, 50.0, 3.0)
with c3:
    user_input['Vitamin_D_Level'] = st.number_input("Vitamin D", 5.0, 100.0, 30.0)

st.divider()

# --- SECTION 4: LIFESTYLE ---
st.header("4. Lifestyle Factors")
c1, c2, c3 = st.columns(3)
with c1:
    user_input['Physical_Activity'] = st.selectbox("Activity Level", ["Low", "Medium", "High"])
    user_input['Diet_Type'] = st.selectbox("Diet", ["Vegetarian", "Non-Vegetarian", "Vegan"])
with c2:
    user_input['Smoking_Status'] = st.selectbox("Smoking", ["Never", "Former", "Current"])
    user_input['Alcohol_Intake'] = st.selectbox("Alcohol Intake", ["None", "Moderate", "High"])
with c3:
    user_input['Stress_Level'] = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    user_input['Regular_Checkups'] = st.selectbox("Regular Checkups?", ["No", "Yes"])

st.divider()

# ==========================================
# 3. Logic & Prediction
# ==========================================
# Center the button for better visibility
_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    predict_btn = st.button("📊 Analyze Complete Health Profile", type="primary", use_container_width=True)

if predict_btn:
    # 1. Create DataFrame
    input_df = pd.DataFrame([user_input])
    
    # 2. One-Hot Encoding (Same as training)
    input_encoded = pd.get_dummies(input_df)
    
    # 3. Align Columns
    final_df = input_encoded.reindex(columns=model_features, fill_value=0)
    
    # 4. Predict
    prediction = model.predict(final_df)[0]
    probability = model.predict_proba(final_df)[0][1]
    
    # 5. Display Results
    st.markdown("---")
    st.header(" Analysis Results")
    
    col_result, col_shap = st.columns([1, 2])
    
    with col_result:
        st.subheader("Risk Assessment")
        if prediction == 1:
            st.error(f"⚠️ High Risk ({probability*100:.1f}%)")
            st.markdown("""
            **Interpretation:**
            Your health profile shares significant characteristics with diabetic patients. 
            
            **Next Steps:**
            - Consult a healthcare provider immediately.
            - Consider confirmatory testing (e.g., OGTT).
            """)
        else:
            st.success(f"✅ Low Risk ({probability*100:.1f}%)")
            st.markdown("""
            **Interpretation:**
            Your markers are currently within a lower-risk range.
            
            **Next Steps:**
            - Maintain your healthy habits.
            - Continue regular screening as recommended by your doctor.
            """)
            
    with col_shap:
        st.subheader("Key Driving Factors")
        with st.spinner("Identifying your specific risk factors..."):
            
            # 1. Get SHAP values
            shap_values = explainer.shap_values(final_df)
            
            # 2. Robust Extraction logic to handle different SHAP output formats
            # Goal: Get a 1D array of shape (n_features,)
            
            if isinstance(shap_values, list):
                # Case A: List of arrays [ (1, n_features), (1, n_features) ]
                # We want the second array (Positive Class / Risk)
                vals = shap_values[1]
            elif len(shap_values.shape) == 3:
                # Case B: 3D Array (1, n_features, 2)
                # We want all features for the first sample, for class 1
                vals = shap_values[0, :, 1]
            else:
                # Case C: 2D Array (1, n_features)
                vals = shap_values

            # 3. Flatten to ensure 1D array
            vals = vals.flatten()
            
            # Safety Check: Ensure lengths match before creating DataFrame
            if len(vals) != len(model_features):
                st.error(f"Shape Mismatch: Model expects {len(model_features)} features, but SHAP calculated {len(vals)} values.")
                st.stop()

            # 4. Create DataFrame
            shap_df = pd.DataFrame({
                'Feature': model_features,
                'Contribution': vals
            })
            
            # Filter & Sort
            shap_df = shap_df[shap_df['Contribution'].abs() > 0.01]
            shap_df = shap_df.sort_values(by='Contribution', key=abs, ascending=False).head(7)
            
            if not shap_df.empty:
                top_factor = shap_df.iloc[0]
                impact_type = "raised" if top_factor['Contribution'] > 0 else "lowered"
                
                st.info(f"The most significant factor was **{top_factor['Feature']}**, which **{impact_type}** your risk score.")
                
                # Plot
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#ff4b4b' if x > 0 else '#21c354' for x in shap_df['Contribution']]
                ax.barh(shap_df['Feature'], shap_df['Contribution'], color=colors)
                ax.set_xlabel("Impact on Risk Score (Right/Red = Higher Risk)")
                ax.invert_yaxis() # Put the biggest factor at the top
                st.pyplot(fig)
            else:
                st.warning("No single factor had a strong enough impact to highlight.")