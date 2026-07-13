# 🩺 End-to-End Cardiovascular Risk Assessment & Guardrailed AI Simulation Pipeline

Welcome to the comprehensive repository for the AI/ML Capstone Project. This production-grade system spans the entire data lifecycle—from raw feature engineering and automated pipeline training to interactive web-application serving with integrated string safety guardrails.

---

## 🗺️ Project Architecture & Directory Layout

The repository is modularly structured into dedicated pipeline spaces following strict production separation standards:

* **📂 `part1/` (Data Engineering Foundation):** Handles structural type casting, manages anomalies/nulls, engineers the custom binary classification target risk label, and materializes `cleaned_data.csv`.
* **📂 `part2/` (Supervised ML Blueprint):** Blueprints the data splits ($80/20$ train/validation partitions), manages schema separation configurations, and hosts exploratory data analysis visualization artifacts.
* **📂 `part3/` (Model Training & Serialization):** Features production training workflows utilizing an ensemble Random Forest algorithm, tracks training pipeline environment metrics, and serializes the final fitted model object (`heart_disease_pipeline.pkl`).
* **📂 `part4/` (Interactive Application Layer):** Powers the core user interface using Streamlit (`app.py`), runs deterministic JSON extraction schema workflows (Track C), and operates regex string safety sanitization layers (`guardrails.py`) to block PII.

---

## 🛠️ Installation & Local Execution

To spin up the entire production application interface locally, clone this workspace and execute the following dependency setup sequences:

```bash
# 1. Navigate to the workspace root
cd capstone-ai-ml

# 2. Fire up the interactive frontend application
streamlit run part4/app.py
