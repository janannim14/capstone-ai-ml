# Part 4: Interactive Application & Safety Guardrails

This directory contains the core production user interface, structured LLM-powered prediction explanations, and automated input security filters.

---

## 🎯 Chosen Feature Track
**Track C: Model Prediction Explanation Pipeline**

---

## 🛠️ Application Components

The interface utilizes `streamlit` to deliver an interactive health core driven by two independent layers:

1. **Predictive Diagnostic Module (`app.py`):** Loads the production model pipeline artifact generated in Part 3. It takes real-time patient parameters from form sliders to evaluate cardiovascular risk probabilities instantly.
2. **Safety Guardrail Engine (`guardrails.py`):** Intercepts natural language queries from the text chat interface. It runs active regular expression scans to block potential credential leaks or personal identifiable information (PII) before they can touch backend services.

---

## 🤖 Prompt Engineering Design

### System Prompt
```text
You are a structured clinical data analyzer. Your job is to provide a clear explanation for a heart disease prediction model's output. You must return your response in a strict JSON format matching the requested schema. Output only valid JSON.
