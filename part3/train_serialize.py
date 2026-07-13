import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 📁 Ensure part3 directory structure exists safely
os.makedirs('part3', exist_ok=True)

print("--- STEP 1: INITIALIZING SERIALIZATION PIPELINE ---")
try:
    df = pd.read_csv('cleaned_data.csv')
    print(f"-> Baseline data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("-> Error: 'cleaned_data.csv' missing. Run Part 1 first.")
    raise

# Identify target classification feature
target_col = [c for c in df.columns if 'risk' in c.lower() or 'target' in c.lower()]
target_col = target_col[0] if target_col else df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

# Clean tracking metadata columns
id_cols = [c for c in X.columns if 'id' in c.lower() or 'name' in c.lower()]
if id_cols:
    X = X.drop(columns=id_cols)

# Extract column lists for processing
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# --- STEP 2: BUILD COMPOSITE COMPONENT TRACE ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

production_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
])

# Training split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- STEP 3: FITTING PRODUCTION MODEL ENGINE ---")
production_pipeline.fit(X_train, y_train)
print("-> Final model calibration complete.")

# --- STEP 4: MODEL PIPELINE SERIALIZATION ---
model_pickle_path = 'part3/heart_disease_pipeline.pkl'
joblib.dump(production_pipeline, model_pickle_path)

print(f"\n🎉 Part 3 Success: Production model pipeline serialized safely to '{model_pickle_path}'!")