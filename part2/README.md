# Part 2: Supervised Machine Learning Foundation

This directory documents the core blueprint for our supervised machine learning workflow, automated pipeline preprocessing steps, and quantitative evaluation planning.

---

## ⚙️ Baseline Machine Learning Workflow

The pipeline foundation structures raw features into a reliable, scalable classification engine using the following architecture:

1. **Checkpoint Ingestion:** Pulls the baseline `cleaned_data.csv` exported from Part 1 to maintain structural data consistency.
2. **Schema Separation:** Automatically scans the incoming feature space to separate raw numeric series from multi-class categorical arrays.
3. **Target Guardrails:** Isolates target flag markers (like `Heart Attack Risk`) and drops arbitrary indexing trackers to ensure zero target data leakage during evaluation phases.
4. **Scikit-Learn Pipeline Wrapper:** Prepares a `ColumnTransformer` layout to seamlessly pair feature scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) directly with an ensemble classifier engine.
5. **Stratified Split Partitioning:** Splices features into 80/20 train/test segments while ensuring class proportions remain identical across both datasets for balanced validation.

---

## 📊 Performance Assessment Results

The execution loops calculate core classification metrics into the console and track historical performance configurations:

### Receiver Operating Characteristic (`visual_7_roc.png`)
* **What it shows:** A diagnostic curve mapping the True Positive Rate (Sensitivity) directly against the False Positive Rate (1 - Specificity) across different probability thresholds.
* **Why it matters:** It displays our model's capacity to discriminate between high-risk cardiac events and baseline indicators. The calculated Area Under the Curve (AUC) provides a single metric to track how well our model performs compared to random guessing.
