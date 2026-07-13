# Part 2: Supervised Machine Learning Pipeline

This directory handles the end-to-end model training workflow, automated pipeline preprocessing, and quantitative evaluation charts.

---

## 🛠️ Machine Learning Workflow

The `train_model.py` script automatically structures raw features into a reliable ensemble classifier using the following pipeline setup:

1. **Checkpoint Recovery:** Pulls the baseline `cleaned_data.csv` exported from Part 1 to maintain data consistency.
2. **Schema Separation:** Automatically scans the incoming feature space to separate raw numeric series from multi-class categorical arrays.
3. **Target Guardrails:** Dynamically isolates target flag markers (like `Heart Attack Risk`) and drops indexing trackers to ensure zero target data leakage during evaluation.
4. **Scikit-Learn Pipeline Wrapper:** Combines a `ColumnTransformer` (scaling numerical features with `StandardScaler` and encoding text slices with `OneHotEncoder`) directly with a `RandomForestClassifier` engine.
5. **Stratified Split Partitioning:** Splices features into 80/20 train/test segments while ensuring class proportions remain identical across both datasets.

---

## 📈 Performance Assessment Results

Running the script outputs core classification metrics into the console and exports the performance chart directly to `part2/plots/`:

### Receiver Operating Characteristic (`visual_7_roc.png`)
* **What it shows:** A diagnostic curve mapping the True Positive Rate (Sensitivity) directly against the False Positive Rate (1 - Specificity) across different probability thresholds.
* **Why it matters:** It displays our model's capacity to discriminate between high-risk cardiac events and baseline indicators. The calculated Area Under the Curve (AUC) provides a single metric to track how well our model performs compared to random guessing (the diagonal line).