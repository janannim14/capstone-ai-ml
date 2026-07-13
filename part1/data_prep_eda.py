import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 📁 Create local storage folders safely (Updated to include /plots)
os.makedirs('part1/plots', exist_ok=True)

# Set global styles for visualization properties
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [8, 5]

print("--- STEP 1: INGESTING RAW DATA ASSETS ---")
try:
    df = pd.read_csv('/content/drive/MyDrive/heart_attack_prediction_dataset.csv')
    print(f"-> Successfully loaded original dataset. Shape: {df.shape}")
except FileNotFoundError:
    # Fallback to alternative name if primary is missing
    try:
        df = pd.read_csv('heart_attack_prediction_dataset.csv')
        print(f"-> Successfully loaded fallback dataset. Shape: {df.shape}")
    except FileNotFoundError:
        print("-> Error: Dataset file not found in the working directory.")
        raise

# Clean trailing spaces from column headers
df.columns = df.columns.str.strip()

# 🛠️ DYNAMIC COLUMN CHECK (Prevents IndexError & KeyError)
if 'Medical_Expenditure' in df.columns:
    expenditure_col = 'Medical_Expenditure'
elif 'Income' in df.columns:
    expenditure_col = 'Income'
else:
    # If both are missing, grab the first available numerical column as safe fallback
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    expenditure_col = numeric_cols[0]

print(f"-> Selected Target Continuous Column for Histogram: '{expenditure_col}'")

# Text Metric Decomposition (Splitting Blood Pressure)
if 'Blood Pressure' in df.columns:
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    df = df.drop(columns=['Blood Pressure'])
    print("-> Blood Pressure metric split completed.")

# Distribution-Aware Imputation (Median based parsing constraints)
for col in df.columns:
    if df[col].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(df[col]):
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# Memory Optimizations & Row-wise Cleanups
df = df.drop_duplicates()
if 'Diet' in df.columns:
    df['Diet'] = df['Diet'].astype('category')

# Export clean base checkpoint file
df.to_csv('cleaned_data.csv', index=False)
print("-> Pipeline executed cleanly! Baseline checkpoint exported to 'cleaned_data.csv'\n")


print("--- STEP 2: GENERATING SEPARATE ANALYTICAL VISUALS (REQUIRED) ---")

# 📊 1. Line Plot (Saved into part1/plots/)
plt.figure()
if 'Age' in df.columns:
    plt.plot(df['Age'].values[:100], color='royalblue', lw=2)
    plt.ylabel('Age (Years)')
else:
    plt.plot(df.iloc[:, 1].values[:100], color='royalblue', lw=2)
plt.title('Patient Trends Trace Log (Sample Pool)', fontsize=12)
plt.xlabel('Sequence Index Tracking Count')
plt.savefig('part1/plots/visual_1_line.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved: part1/plots/visual_1_line.png")

# 📊 2. Bar Plot (Saved into part1/plots/)
plt.figure()
if 'Diet' in df.columns:
    df.groupby('Diet', observed=False)['Cholesterol'].mean().plot(kind='bar', color='salmon', edgecolor='black')
elif 'Sex' in df.columns:
    df.groupby('Sex', observed=False)['Cholesterol'].mean().plot(kind='bar', color='salmon', edgecolor='black')
else:
    df.iloc[:, :3].mean().plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Mean Cholesterol Levels Across Groups', fontsize=12)
plt.savefig('part1/plots/visual_2_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved: part1/plots/visual_2_bar.png")

# 📊 3. Histogram (Saved into part1/plots/)
plt.figure()
sns.histplot(df[expenditure_col], bins=30, kde=True, color='indigo')
plt.title(f'{expenditure_col} Assessment Profile', fontsize=12)
plt.xlabel(f'{expenditure_col} Metrics')
plt.ylabel('Patient Record Frequency Count')
plt.savefig('part1/plots/visual_3_hist.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved: part1/plots/visual_3_hist.png")

# 📊 4. Scatter Plot (Saved into part1/plots/)
plt.figure()
sns.scatterplot(data=df, x='Age' if 'Age' in df.columns else df.columns[1], y='Cholesterol' if 'Cholesterol' in df.columns else df.columns[2], alpha=0.6, color='teal')
plt.title('Continuous Matrix Mapping Configuration', fontsize=12)
plt.savefig('part1/plots/visual_4_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved: part1/plots/visual_4_scatter.png")

# 📊 5. Boxplot (Saved into part1/plots/)
plt.figure()
if 'Sex' in df.columns and 'Triglycerides' in df.columns:
    sns.boxplot(data=df, x='Sex', y='Triglycerides', palette='Set2', hue='Sex', legend=False)
elif 'Triglycerides' in df.columns:
    sns.boxplot(y=df['Triglycerides'], color='aquamarine')
else:
    sns.boxplot(y=df.iloc[:, 2], color='aquamarine')
plt.title('Structural Variance Distribution Spread', fontsize=12)
plt.savefig('part1/plots/visual_5_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved: part1/plots/visual_5_boxplot.png")

# 📊 6. Heatmap (Saved into part1/plots/)
plt.figure(figsize=(11, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
matrix_cols = [c for c in numeric_cols if c not in ['Patient ID', 'PatientID']]
sns.heatmap(df[matrix_cols].corr(), annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Cardiovascular Pearson Correlation Structural Heatmap Matrix', fontsize=12)
plt.savefig('part1/plots/visual_6_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved: part1/plots/visual_6_heatmap.png")

print("\n🎉 Part 1 Data Engineering Layer Success: All 6 plots locked safely inside 'part1/plots' folder!")