# Federated ML for IoT Security – Intrusion Detection System

This repository contains the implementation of a **Machine Learning-based Intrusion Detection System (IDS)** for IoT networks.  
The project supports **binary classification** (attack vs. benign) and **multiclass classification** (specific attack families), along with extensions for:

- Semi-supervised learning (label efficiency)
- Hybrid supervised–unsupervised models (zero-day detection)
- Concept drift detection (ADWIN)
- Visualization of evaluation metrics and performance curves

# Project Structure

PROJECT_IOT_IDS/
-.venv #Virtual environment
-data
    -processed
    -raw
-notebooks
    -visualize_results.ipynb
-results
    -figures
    -metrics
    -models
-src
    -pycache
    -utils
    -_init_.py
    -compare_label_efficiency.py
    -prepare_iot23.py
    -train_binary.py
    -train_concept_drift.py
    -train_hybrid.py
    -train_multiclass.py
    -train_semi_sumpervised.py
    -train_unsupervised.py
-README.md
-requirements.txt

# Create a virtual environment:

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Requirement Installation 

pip install -r requirements.txt

# Dataset Download

About the dataset - 


# sequence to run the codes





