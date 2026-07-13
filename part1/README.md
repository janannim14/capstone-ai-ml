# Part 1: Data Engineering & Preprocessing Pipeline

This directory documents the data extraction, type conversion engineering, and cleaning pipelines used to construct the clean base framework for our modeling stages.

---

## ⚙️ Core Engineering Workflows

The preprocessing module ingestion layer sanitizes raw clinical records using the following distinct processing metrics:

1. **Target Feature Engineering:** Analyzes health records to map out a clear binary target label (Normal vs. High Risk Profile) to serve as our explicit supervised classification flag.
2. **Structural Type Casting:** Explicitly scans the raw dataset column arrays to separate continuous numeric markers from discrete categorical groups, converting all features into reliable data types.
3. **Data Quality Interceptions:** Identifies and cleans any missing, null, or out-of-bounds anomaly rows to ensure statistical stability during model training.
4. **Clean Baseline Export:** Compiles all processed steps and materializes the final outputs into a separate, tracking-ready dataset saved directly to the root workspace folder as `cleaned_data.csv`.

---

## 📁 Repository Directory Separation

To preserve production architectural cleanliness, all logic frameworks, data cleanings, and raw attribute configurations are completely isolated right here inside the `part1/` environment branch. This guarantees a clean handoff to the downstream exploratory analysis stages.
