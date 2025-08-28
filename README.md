# Machine Learning-Based Intrusion Detection for IoT Networks: Lightweight, Semi-Supervised, and Drift-Aware 

This repository contains the implementation of a **Machine Learning-based Intrusion Detection System (IDS)** for IoT networks.  
The project supports **binary classification** (attack vs. benign) and **multiclass classification** (specific attack families), along with extensions for:

- Semi-supervised learning (label efficiency)
- Hybrid supervised–unsupervised models (zero-day detection)
- Concept drift detection (ADWIN)
- Streaming evaluation
- Visualization of evaluation metrics and performance curves

# Introduction

Intrusion detection in IoT networks faces challenges such as class imbalance, evolving traffic (concept drift), and label scarcity. This project provides a reproducible implementation of lightweight ML-based IDS methods tailored for IoT-23, with focus on efficiency and adaptability.

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
    -metadata
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

Using a virtual environment (venv) ensures that the project dependencies remain isolated from the system Python installation. This avoids version conflicts, makes the setup reproducible across different machines, and allows the environment to be easily recreated using the provided requirements.txt file.

# Requirement Installation 

pip install -r requirements.txt

# Dataset Download

Download link - https://www.kaggle.com/datasets/engraqeel/iot23preprocesseddata

After downloading place the file here:
PROJECT_IOT_IDS/
-.venv #Virtual environment
-data
    -processed
    -raw
        iot23_combined_new.csv

About the dataset - The IoT-23 dataset is a widely used benchmark for intrusion detection in IoT environments. It was developed by the Stratosphere Laboratory at CTU University, Czech Republic, and first released in January 2020 with support from Avast Software. The dataset contains 23 network captures (pcap files), including 20 malware scenarios where real IoT devices were infected with different malware samples, and 3 benign scenarios captured from everyday smart devices such as a Philips HUE smart lamp, an Amazon Echo assistant, and a Somfy smart doorlock. Unlike simulated datasets, IoT-23 records real hardware traffic in a controlled network with live internet access, making it closer to real-world conditions. The malware scenarios were executed on a Raspberry Pi, producing diverse protocols and behaviors, while the benign scenarios provide normal background traffic. Together, these captures offer a rich mix of labeled IoT traffic that enables researchers to test and develop machine learning-based intrusion detection systems with a balance of realism and reproducibility.


# Sequence to run the codes

Start running the code from src folder (.py files)
1. Preparation_iot23.py
2. train_binary.py
3. train_multiclass.py
4. train_semi_supervised.py
5. compare_label_efficiency.py
6. train_unsupervised.py
7. train_hybrid.py
8. train_concept_drift.py (supervised)
9. train_concept_drift_semi.py
10. train_concept_drift_hybrid.py
11. realtime_stream_sim.py

# Running codes

For all the .py files, run on terminal, except reatime_stream_sim.py.
For reatime_stream_sim.py run this code on powershell: 

cd C:\PROJECT_IOT_IDS
.\.venv\Scripts\activate

python .\src\realtime_stream_sim.py `
  --parquet .\data\processed\dataset_cleaned.parquet `
  --model   .\results\models\supervised_binary_baseline.joblib `
  --chunk_rows 400 `
  --sleep 0.25 `
  --score_threshold 0.5 `
  --out_csv .\results\metrics\realtime_stream_results.csv `
  --alerts_csv .\results\metrics\realtime_stream_alerts.csv `
  --top_alerts_per_chunk 10

This part takes 15 minutes to run.

After running these codes, run the notebooks folder (consists of all the figures and tables).
 

# What is saved where

All the figures and csv files are saved in figures\vis and metrics folders respectively.

# Acknowledgements

Dataset courtesy of Stratosphere Laboratory, CTU University, Prague.




