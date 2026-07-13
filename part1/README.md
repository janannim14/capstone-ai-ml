# Part 1: Pipeline Setup & Exploratory Data Analysis

This directory handles the raw data ingestion pipeline, clean up scripts, and diagnostic plots required to understand our baseline patient metrics before model training.

---

## 🛠️ What the Code Does

The `data_prep_eda.py` script runs a complete clean-up on the dataset to handle messy real-world formats. It walks through these stages:

1. **Smart File Loading:** Looks for the heart attack dataset locally first. If it's missing, it defaults to a clean error breakout instead of crashing silently.
2. **Header Normalization:** Strips out annoying trailing whitespaces from column names (`df.columns.str.strip()`) to prevent broken key lookups later.
3. **Dynamic Target Selection:** Checks if `Medical_Expenditure` or `Income` exists in the schema. If neither is found, it automatically grabs the first numeric column available so the visualization code doesn't fail.
4. **Feature Splitting:** Parses compound strings like the `Blood Pressure` column (e.g., "135/85") into two clean numeric columns: `Systolic_BP` and `Diastolic_BP`, then discards the raw text feature.
5. **Null Patching:** Targets empty numeric cells and fills them using the series median to keep distributions safe from skewed mean values.
6. **Deduplication:** Drops exact redundant row clones and converts raw text columns like `Diet` into optimized categorical indicators to save RAM footprint.

---

## 📊 Understanding the Generated Plots

Running the script automatically dumps 6 high-res plots directly into `part1/plots/`:

### 1. Line Plot (`visual_1_line.png`)
* **What it shows:** A sequential trend log tracking patient age variations across the first 100 sample records.
* **Why it matters:** It lets us scan for abnormal raw spikes, tracking sequence glitches, or unexpected step-patterns in the sample order.

### 2. Bar Plot (`visual_2_bar.png`)
* **What it shows:** An aggregated group comparison showing mean cholesterol levels sliced by patient diet categories.
* **Why it matters:** Shows us at a glance whether categorical slices (like diet types) exhibit distinct variance shifts in cholesterol profiles.

### 3. Histogram (`visual_3_hist.png`)
* **What it shows:** A distribution curve tracking patient record frequencies alongside a smooth KDE (Kernel Density Estimate) overlay line.
* **Why it matters:** It helps diagnose data skewness, modalities, and outlines whether our target financial/continuous vectors require tail-end transformation.

### 4. Scatter Plot (`visual_4_scatter.png`)
* **What it shows:** A standard continuous cross-plot tracking the relationship between Patient Age and Cholesterol.
* **Why it matters:** Used to spot initial structural groupings, density clouds, or check if there's an obvious linear relationship between the two features.

### 5. Boxplot (`visual_5_boxplot.png`)
* **What it shows:** A full five-number statistical summary (min, Q1, median, Q3, max) comparing patient sex classifications against triglycerides spread.
* **Why it matters:** Perfect for identifying hidden statistical outliers and observing directional variance shifts across category distributions.

### 6. Heatmap (`visual_6_heatmap.png`)
* **What it shows:** A Pearson Correlation matrix calculating linear dependencies across all numerical features.
* **Why it matters:** Crucial for feature selection. It lets us spot high multicollinearity factors early on so we don't feed highly redundant data features to our classifiers down the line.