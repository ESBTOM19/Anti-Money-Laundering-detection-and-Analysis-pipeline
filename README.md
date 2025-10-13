ANTI-MONEY LAUNDERING DETECTION AND ANALYSIS PIPELINE
Author

Caxton Henry

Focus

AI Engineering, Financial Crime Analytics, Quantitative Modeling

Vision

Building transparent, intelligent systems that merge financial insight with data-driven precision, with a keen interest in financial crime analytics.

Overview

This project implements a full end-to-end Financial Crime Detection Pipeline built using real-world principles of anti-money laundering (AML), transaction risk modeling, and supervised machine learning.

It leverages the combined power of data preprocessing, feature engineering, model training, evaluation, and visual analytics to replicate the intelligence of an early-stage fraud detection system used by financial institutions and compliance units.

The project simulates how banks flag and evaluate suspicious transactions based on historical alerts and transactional data, then trains predictive models to classify risk levels, providing data-driven insights into fraud dynamics.

Pipeline Workflow:

Data Preprocessing → Feature Engineering → Modeling → Evaluation → Visualization

Project Motivation

Financial crime remains one of the most critical global challenges for both regulators and institutions. The FATF (Financial Action Task Force) and Basel Committee consistently emphasize the need for AI-driven detection systems that go beyond rule-based filters.

Traditional systems struggle with:

High false positive rates

Poor adaptability to new laundering typologies

Latency in cross-border transaction monitoring

This project addresses those issues by introducing machine learning interpretability and statistical robustness into a detection framework that can adaptively improve with new data.

Financial Organization Context

In the financial domain, institutions must comply with AML directives, KYC regulations, and suspicious activity monitoring requirements. Every transaction carries a probabilistic risk of being part of fraud, money laundering, or insider trading.

This system models that risk using structured data such as:

Transaction Size

Account Age

Alert Frequency

Historical Disposition of Alerts

Behavioral patterns (e.g., unusually high transfers or round-number deposits)

Each feature contributes to understanding the probability of suspicious activity, helping compliance teams to prioritize investigations effectively.

From a financial modeling standpoint, this approach mimics expected loss modeling:

Expected Loss
=
Probability of Default
×
Exposure
×
Loss Given Default
Expected Loss=Probability of Default×Exposure×Loss Given Default

Here, instead of default, we model probability of suspicion — allowing AML units to quantify investigative risk.

Raw Data

This project is built upon SynthAML, a Synthetic Data Set to Benchmark Anti-Money Laundering Methods, an open-access dataset designed to emulate realistic financial transaction and alert behaviors.

The data comprises:

Synthetic Alerts (simulated Suspicious Activity Reports - SARs)

Synthetic Transactions (underlying transactional details)

These datasets provide a realistic sandbox for AML research while maintaining full anonymization and privacy compliance.

In this project:

Raw CSV files were converted to Parquet format for optimized performance and scalability.

The alerts dataset captures outcomes such as Report or Dismiss.

The transactions dataset provides granular details on financial movements linked to each alert.

Together, they form a realistic base for financial crime detection research.

Technical Architecture
Data Preprocessing

Conversion from CSV to Parquet format for optimized read/write speeds and scalability.

Handling of missing values, inconsistent data types, and outlier mitigation.

Data normalization and categorical encoding for model readiness.

The preprocess.py module ensures all inputs are clean, consistent, and suitable for large-scale ML training.

Feature Engineering

Creation of transactional ratios, rolling averages, and aggregate behavioral features.

Transformation of domain-driven variables (e.g., transaction size relative to account median activity).

Encoding of binary labels:

Report (1) → suspicious

Dismiss (0) → normal

The result is a balanced and interpretable feature matrix capturing both numerical and behavioral signals of potential fraud.

Modeling

Trained Models:

Random Forest Classifier – for robust non-linear detection

XGBoost Classifier – for gradient-boosted interpretability and performance

Decision Tree Classifier – for transparent decision logic

Linear Regression (baseline) – to model linear risk relationships

Each model was trained on structured features derived from merged alert and transaction datasets. The target variable represents whether an alert was legitimate (“Report”) or dismissed (“False Positive”).

Key Metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Score

This mirrors how banks validate models under Basel’s SR 11-7 model risk management framework.

Evaluation and Model Governance

Model evaluation goes beyond raw accuracy. The system generates:

Confusion Matrices (false positives vs. false negatives)

ROC Curves and AUC metrics (class separability)

Precision-Recall tradeoffs (threshold tuning impact)

From a compliance perspective:

False negatives → regulatory risk

False positives → operational inefficiency

This ensures alignment with explainability and transparency requirements demanded by regulators.

Visualization and Insights

The visualization layer (visualize_results.py) delivers interpretable and investor-ready dashboards:

Distribution of transaction sizes by risk label

Heatmaps showing confusion matrices

ROC curves highlighting performance trade-offs

Feature importance charts from tree-based models

These visuals allow both data scientists and financial analysts to interpret model behavior — bridging technical and financial perspectives.

Interpretation of Results

Random Forest and XGBoost models outperform linear methods due to better fraud pattern recognition.

ROC-AUC > 0.8 indicates strong discriminatory power between suspicious and non-suspicious activity.

The priority is minimizing false negatives (missed suspicious cases) without overwhelming investigators with false positives.

Real-world reflection:

High-risk transactions → investigated first

Low-risk transactions → processed with minimal oversight

Thresholds adjusted per institutional risk appetite

Technologies Used
Category	Tools / Libraries
Data Handling	pandas, numpy, pyarrow
Machine Learning	scikit-learn, xgboost, imbalanced-learn
Statistical Modeling	statsmodels, scipy
Visualization	matplotlib, seaborn
Automation & Utilities	tqdm, joblib, requests
Development	Python 3.10+, Jupyter, VS Code
Intellectual Summary

This project merges financial domain expertise with technical modeling proficiency. It demonstrates capability in:

Translating financial risk theory into algorithmic detection systems

Designing scalable ML pipelines aligned with regulatory and audit principles

Understanding economic and compliance consequences of model outputs

Professional Applications:

AML transaction monitoring

Insurance claim fraud detection

Credit risk modeling

FinTech compliance analytics

Research Relevance:
A step toward explainable AI in finance, ensuring models remain transparent, auditable, and ethically grounded.

Future Improvements

Integration of graph-based anomaly detection for hidden entity relationships

Addition of streaming analytics (Kafka / Spark) for real-time monitoring

Incorporation of NLP for textual alert descriptions

Use of SHAP or LIME for advanced interpretability

Financial calibration using weighted misclassification costs

Data Ethics and Privacy

All data are synthetic and fully anonymized, ensuring zero exposure of real client information.

The project complies with:

GDPR principles

FATF Recommendation 15 (technology and privacy)

AI ethics guidelines on fairness and transparency

It advocates for human-in-the-loop oversight, bias detection, and ethical model auditing — promoting accountability across the machine learning lifecycle.