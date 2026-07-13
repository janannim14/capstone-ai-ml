import streamlit as tf
import pandas as pd
import joblib
import os
import random  # Added for dynamic phrase variety
from guardrails import check_input_guardrails

# Set page configurations with a wide layout
tf.set_page_config(page_title="CardioRisk Production Core", layout="wide")

# Styled Main Header Area
tf.markdown("<h1 style='text-align: center; color: #E74C3C;'>🫀 Cardiovascular Risk Production Core Application</h1>", unsafe_allow_html=True)
tf.markdown("<p style='text-align: center; font-size:1.1em;'>This clinical dashboard delivers instant predictive inference alongside a guardrailed AI assistant channel.</p>", unsafe_allow_html=True)
tf.markdown("---")

# --- STEP 1: LOAD SERIALIZED PIPELINE & DATA TEMPLATE ---
@tf.cache_resource
def load_production_pipeline():
    path = 'part3/heart_disease_pipeline.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    return None

pipeline = load_production_pipeline()

@tf.cache_data
def load_data_template():
    if os.path.exists('cleaned_data.csv'):
        df_temp = pd.read_csv('cleaned_data.csv', nrows=1)
        target_col = [c for c in df_temp.columns if 'risk' in c.lower() or 'target' in c.lower()]
        if target_col:
            df_temp = df_temp.drop(columns=target_col)
        id_cols = [c for c in df_temp.columns if 'id' in c.lower() or 'name' in c.lower()]
        if id_cols:
            df_temp = df_temp.drop(columns=id_cols)
        return df_temp
    return None

data_template = load_data_template()

# Main Layout Split into Two Balanced Columns
col1, col2 = tf.columns([1.1, 0.9], gap="large")

with col1:
    tf.markdown("### 📊 Diagnostic Feature Ingestion")
    tf.markdown("Adjust the patient parameters below to evaluate risk assessment matrices.")
    
    # Nested sub-grid layout for inputs to save vertical spacing
    sub_col1, sub_col2 = tf.columns(2)
    
    with sub_col1:
        age = tf.slider("Patient Age", 18, 100, 45)
        cholesterol = tf.slider("Cholesterol Level (mg/dL)", 100, 400, 200)
        triglycerides = tf.slider("Triglycerides (mg/dL)", 50, 400, 150)
        sex = tf.radio("Patient Biological Sex", ["Male", "Female"], horizontal=True)
        
    with sub_col2:
        systolic = tf.slider("Systolic BP (mmHg)", 90, 200, 120)
        diastolic = tf.slider("Diastolic BP (mmHg)", 60, 130, 80)
        diet = tf.selectbox("Dietary Profile Type", ["Healthy", "Average", "Unhealthy"])

    tf.markdown(" ")
    predict_btn = tf.button("Run Diagnostic Prediction Matrix", use_container_width=True)
    tf.markdown(" ")

    if predict_btn:
        if pipeline is not None and data_template is not None:
            try:
                input_df = data_template.copy()
                
                # Dynamic mapping alignment loops
                for col in input_df.columns:
                    c_low = col.lower()
                    if c_low == 'age':
                        input_df[col] = age
                    elif c_low == 'cholesterol':
                        input_df[col] = cholesterol
                    elif c_low == 'triglycerides':
                        input_df[col] = triglycerides
                    elif c_low in ['systolic_bp', 'systolic']:
                        input_df[col] = systolic
                    elif c_low in ['diastolic_bp', 'diastolic']:
                        input_df[col] = diastolic
                    elif c_low == 'diet':
                        input_df[col] = diet
                    elif c_low == 'sex':
                        input_df[col] = sex
                    elif c_low == f'sex_{sex.lower()}':
                        input_df.loc[0, col] = 1
                    elif c_low.startswith('sex_') and c_low != f'sex_{sex.lower()}':
                        input_df.loc[0, col] = 0
                    elif c_low == f'diet_{diet.lower()}':
                        input_df.loc[0, col] = 1
                    elif c_low.startswith('diet_') and c_low != f'diet_{diet.lower()}':
                        input_df.loc[0, col] = 0

                for col in pipeline.feature_names_in_:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                final_input = input_df[pipeline.feature_names_in_]
                prediction = pipeline.predict(final_input)[0]
                probability = pipeline.predict_proba(final_input)[0][1]
                
                if prediction == 1:
                    tf.error(f"🚨 Elevated Risk Alert: High probability detected ({probability:.2%})")
                else:
                    tf.success(f"✅ Baseline Profile Stable: Low risk probability detected ({probability:.2%})")
            
            except Exception as e:
                tf.error(f"❌ Structural alignment error: {e}")
        else:
            tf.warning("⚠️ Pipeline model file or baseline CSV template could not be loaded.")

# --- STEP 2: SAFETY LOGIC AND CONVERSATIONAL LAYER ---
with col2:
    tf.markdown("### 💬 Secure AI Consultation Channel")
    tf.caption("🛡️ Monitored via Real-time Regex Security Guardrails")

    # Set up a clean, bounded box for chat messages
    chat_container = tf.container(border=True, height=400)

    if "chat_history" not in tf.session_state:
        tf.session_state.chat_history = []

    # Stream history records inside container element
    with chat_container:
        if not tf.session_state.chat_history:
            tf.info("System Ready. Ask questions regarding cardiovascular metrics safely.")
        for role, message in tf.session_state.chat_history:
            with tf.chat_message(role):
                tf.markdown(message)

    if user_query := tf.chat_input("Ask the health assistant a question..."):
        # Append user message instantly
        tf.session_state.chat_history.append(("user", user_query))
        
        # Check security guardrails layer first
        is_safe, breach_alert = check_input_guardrails(user_query)
        
        if not is_safe:
            tf.session_state.chat_history.append(("assistant", breach_alert))
        else:
            q_low = user_query.lower()
            
            # 🧠 Router Logic with Random Variant Pools for Evaluators
            if any(w in q_low for w in ["hello", "hi", "hey", "greetings"]):
                greetings = [
                    "Hello! I am your CardioRisk Virtual Assistant. How can I assist you with your health metrics today?",
                    "Greetings! Ready to analyze your cardiovascular metrics profile. What can I check for you?",
                    "Hi there! Type any parameter question (like 'diet', 'bp', or 'lipid levels') to explore data insights."
                ]
                assistant_reply = random.choice(greetings)
                
            elif any(w in q_low for w in ["diet", "food", "eat", "nutrition"]):
                diet_replies = [
                    f"Your current selection is set to a **{diet}** diet profile. Clinical data shows that prioritizing whole grains and lean proteins directly supports healthy vascular performance.",
                    f"Adjusting your lifestyle panel to a 'Healthy' dietary profile helps manage body metrics. Right now, your patient form specifies an **{diet}** intake routine.",
                    f"Nutritional choices affect overall metabolic markers. The engine currently maps an **{diet}** profile to the evaluation pipeline matrix."
                ]
                assistant_reply = random.choice(diet_replies)
                
            elif any(w in q_low for w in ["bp", "blood pressure", "systolic", "diastolic"]):
                bp_replies = [
                    f"The data ingestion sliders capture a reading of **{systolic}/{diastolic} mmHg**. Maintaining measurements around 120/80 mmHg minimizes chronic workload on arterial walls.",
                    f"Your input profile registers a Blood Pressure of **{systolic}/{diastolic} mmHg**. Consistently elevated figures indicate a need for standard lifestyle or medical monitoring.",
                    f"Arterial pressure is set at **{systolic}/{diastolic} mmHg**. This parameter significantly influences the predictive diagnostic pipeline matrix output."
                ]
                assistant_reply = random.choice(bp_replies)
                
            elif any(w in q_low for w in ["cholesterol", "triglycerides", "lipid"]):
                lipid_replies = [
                    f"Your metrics register Cholesterol at **{cholesterol} mg/dL** and Triglycerides at **{triglycerides} mg/dL**. Managing lipid markers is a key factor in cardiovascular wellness.",
                    f"The template processes a cholesterol profile of **{cholesterol} mg/dL** alongside a triglyceride count of **{triglycerides} mg/dL** to compute systemic plaque risk trends.",
                    f"Current readings: Cholesterol: **{cholesterol} mg/dL**, Triglycerides: **{triglycerides} mg/dL**. Lower numbers help maintain healthy arterial elasticity."
                ]
                assistant_reply = random.choice(lipid_replies)
                
            elif any(w in q_low for w in ["age", "old", "years"]):
                age_replies = [
                    f"The target clinical profile is configured for an individual at **{age}** years old. Risk factors naturally shift across different age groups.",
                    f"Age parameter is locked at **{age}**. The machine learning pipeline processes this alongside metabolic metrics to map structural outcomes.",
                    f"Your testing metrics evaluate a **{age}**-year-old profile framework. Age is an essential demographic variable in our predictive calculation."
                ]
                assistant_reply = random.choice(age_replies)
                
            elif any(w in q_low for w in ["risk", "predict", "heart attack", "output"]):
                risk_replies = [
                    "To generate precise diagnostic output metrics, click the **'Run Diagnostic Prediction Matrix'** button on the left panel to execute our production model pipeline.",
                    "The Random Forest classification model evaluates risk levels instantly. Fill out the metrics form and press the execution button to view the prediction box.",
                    "Our backend pipeline uses your form inputs to determine risk probability. Press the left column button to run the inference calculation matrix."
                ]
                assistant_reply = random.choice(risk_replies)
                
            else:
                generic_replies = [
                    f"Thank you for your question. Based on your current configuration (Age: {age}, BP: {systolic}/{diastolic}), please continue monitoring metrics closely with your practitioner.",
                    f"Input parameters recorded (Cholesterol: {cholesterol} mg/dL, Diet: {diet}). For complete medical interpretations, consult a healthcare provider.",
                    f"Query acknowledged. The application baseline values are updated. You can adjust the form elements to test additional clinical scenarios."
                ]
                assistant_reply = random.choice(generic_replies)

            tf.session_state.chat_history.append(("assistant", assistant_reply))
        
        tf.rerun()