# Part 4: Interactive Application & Safety Guardrails

This directory contains the core production user interface and automated input security filters.

---

## 🛠️ Application Components

The interface utilizes `Streamlit` to deliver an interactive health core driven by two independent layers:

1. **Predictive Diagnostic Module (`app.py`):** Loads the production model pipeline artifact generated in Part 3. It takes real-time patient parameters from form sliders to evaluate cardiovascular risk probabilities instantly.
2. **Safety Guardrail Engine (`guardrails.py`):** Intercepts natural language queries from the text chat interface. It runs active regular expression scans to block potential credential leaks (such as API keys) or adversarial prompt injections before they can touch backend services.

---

## 🚀 How to Run the App

To run this application locally, ensure you are in the root repository workspace directory and run:

```bash
python -m streamlit run part4/app.py