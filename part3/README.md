# Part 3: Production Pipeline Training & Model Serialization

This directory contains the production-grade machine learning model training loops, serialized pipeline objects, and environmental configurations.

---

## ⚙️ Pipeline Tracking Configuration

The pipeline behavior is managed using localized environment configurations to keep infrastructure variable parameters separate from code logic:

* **Training Environment:** `development`
* **Model Serialization Artifact Name:** `heart_disease_pipeline.pkl`
* **Local Workspace Export Path:** `./part3`
* **Tracking Subsystem Registry Target:** `http://localhost:5000`

---

## 🏗️ Core Training Architecture (`train_serialize.py`)

The orchestration module executes an end-to-end classification workflow using a **Random Forest Classifier** ensemble layout built on top of `scikit-learn`:

1. **Feature Ingestion:** Loads the structured matrices exported from the Part 1 staging phase.
2. **Preprocessing Integration:** Combines column-wise processing steps and missing value imputations into a single unit.
3. **Pipeline Fitting:** Trains the underlying tree matrices across all clinical parameters to optimize validation accuracy.
4. **Serialization:** Exports the complete, fitted pipeline pipeline object straight to a standalone file (`heart_disease_pipeline.pkl`) for immediate deployment to production interfaces.

---

## 📊 Model Evaluation Metrics

During evaluation runs, the model outputs standard tracking diagnostics to confirm high-confidence generalization behavior:

* **Classification Diagnostics:** Full validation matrix outputting Precision, Recall, and F1-scores for binary risk categories.
* **Graphical Artifacts:** A receiver operating characteristic curve visualization (`visual_7_roc.png`) is compiled and exported inside the project asset directory to track true positive rates against false positive thresholds.
