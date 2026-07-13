import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc

# 📁 Ensure part2/plots directory structure exists cleanly
os.makedirs('part2/plots', exist_ok=True)

print("--- STEP 1: LOADING CLEANED CHECKPOINT DATA ---")
try:
    df = pd.read_csv('cleaned_data.csv')
    print(f"-> Successfully loaded cleaned data. Shape: {df.shape}")
except FileNotFoundError:
    print("-> Error: 'cleaned_data.csv' not found. Please run Part 1 first.")
    raise

# Dynamically locate the target feature column (handles case variants)
target_col = [c for c in df.columns if 'risk' in c.lower() or 'target' in c.lower()]
target_col = target_col[0] if target_col else df.columns[-1]
print(f"-> Target prediction column identified as: '{target_col}'")

X = df.drop(columns=[target_col])
y = df[target_col]

# Drop unique tracking indicators to eliminate data leakage risks
id_cols = [c for c in X.columns if 'id' in c.lower() or 'name' in c.lower()]
if id_cols:
    X = X.drop(columns=id_cols)
    print(f"-> Dropped identification columns from features: {id_cols}")

# Separate numeric and categorical features automatically
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"-> Numerical features parsed: {len(numeric_features)}")
print(f"-> Categorical features parsed: {len(categorical_features)}")

# --- STEP 2: PIPELINE TRANSFORMER CONFIGURATION ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Construct full production model pipeline wrapper
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
])

# Perform a stratified test split to maintain exact class proportions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- STEP 3: TRAINING ENSEMBLE CLASSIFIER ---")
model_pipeline.fit(X_train, y_train)
print("-> Training execution completed smoothly!")

# --- STEP 4: MODEL QUANTITATIVE EVALUATION ---
y_pred = model_pipeline.predict(X_test)
y_probs = model_pipeline.predict_proba(X_test)[:, 1]

print("\n📈 Test Set Performance Matrix Report:")
print(classification_report(y_test, y_pred))

# --- STEP 5: COMPUTE & RENDER PERFORMANCE EVALUATION CHART ---
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='firebrick', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve Profile')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)

# Save visualization artifact directly inside part2/plots/ sub-folder
plt.savefig('part2/plots/visual_7_roc.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n🎉 Part 2 Core Success: Model trained and performance chart saved inside 'part2/plots/visual_7_roc.png'!")