Anti-Money Laundering Detection And Analysis Pipeline

Author

Caxton Henry 
Focus: AI Engineering, Financial Crime Analytics, Quantitative Modeling
Vision: Building transparent, intelligent systems that merge financial insight with data-driven precision with keen interest in financial crime analytics

Overview

This project implements a full end-to-end Financial Crime Detection Pipeline built using real-world principles of anti-money laundering (AML), transaction risk modeling, and supervised machine learning. It leverages the combined power of data preprocessing, feature engineering, model training, evaluation, and visual analytics to replicate the intelligence of an early-stage fraud detection system used by financial institutions and compliance units.

The project simulates how banks flag and evaluate suspicious transactions based on historical alerts and transactional data, then trains predictive models to classify risk levels, providing data-driven insights into fraud dynamics.

The workflow follows the professional data science pipeline used in financial analytics:

Data Preprocessing → Feature Engineering → Modeling → Evaluation → Visualization

Project Motivation

Financial crime remains one of the most critical global challenges for both regulators and institutions. The FATF (Financial Action Task Force) and Basel Committee consistently emphasize the need for AI-driven detection systems that go beyond rule-based filters.

Traditional systems struggle with:

High false positive rates

Poor adaptability to new laundering typologies

Latency in cross-border transaction monitoring

This project attempts to address those issues by introducing machine learning interpretability and statistical robustness into a detection framework that can adaptively improve with new data.

Financial Organization Context

In the financial domain, institutions must comply with AML directives, KYC regulations, and suspicious activity monitoring requirements. Every transaction carries a probabilistic risk of being part of fraud, money laundering, or insider trading.

This system models that risk using structured data such as:

Transaction Size

Account Age

Alert Frequency

Historical Disposition of Alerts

Behavioral patterns like unusually high transfers or round-number deposits

Each of these features contributes to an understanding of the probability of suspicious activity — helping compliance teams to prioritize investigations effectively.

From a financial modeling standpoint, this approach mimics expected loss modeling:

𝐸
𝑥
𝑝
𝑒
𝑐
𝑡
𝑒
𝑑
 
𝐿
𝑜
𝑠
𝑠
=
𝑃
𝑟
𝑜
𝑏
𝑎
𝑏
𝑖
𝑙
𝑖
𝑡
𝑦
 
𝑜
𝑓
 
𝐷
𝑒
𝑓
𝑎
𝑢
𝑙
𝑡
×
𝐸
𝑥
𝑝
𝑜
𝑠
𝑢
𝑟
𝑒
×
𝐿
𝑜
𝑠
𝑠
 
𝐺
𝑖
𝑣
𝑒
𝑛
 
𝐷
𝑒
𝑓
𝑎
𝑢
𝑙
𝑡
Expected Loss=Probability of Default×Exposure×Loss Given Default

except here, instead of default, we model probability of suspicion — allowing AML units to quantify investigative risk.

Technical Architecture
1. Data Preprocessing

Conversion from CSV to Parquet format for optimized read/write speeds and scalability.

Handling of missing values, inconsistent data types, and outlier mitigation.

Data normalization and categorical encoding for model-readiness.

The preprocessing module (preprocess.py) ensures that all inputs are clean, consistent, and suitable for large-scale ML training.

2. Feature Engineering

Creation of transactional ratios, rolling averages, and aggregate behavioral features.

Transformation of domain-driven variables — e.g., transaction size relative to account median activity.

Encoding of binary labels:

Report (1) → suspicious

Dismiss (0) → normal

The result is a balanced and interpretable feature matrix that captures both numerical and behavioral signals of potential fraud.

3. Modeling

Trained models include:

Random Forest Classifier – for robust non-linear detection

XGBoost Classifier – for gradient-boosted interpretability and performance

Decision Tree Classifier – for transparent decision logic

Linear Regression (as a baseline) – to model linear risk relationships

Each model was trained on structured features derived from the merged alerts and transactions datasets. The target variable represents whether an alert was legitimate (“Report”) or dismissed as a false positive.

Key metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Score

This mirrors what banks use internally for model validation under Basel’s model risk management frameworks (SR 11-7).

4. Evaluation and Model Governance

Model evaluation goes beyond raw accuracy.
The system generates:

Confusion Matrices for understanding false positives vs. false negatives

ROC Curves and AUC Metrics to gauge separability of classes

Precision-Recall Tradeoffs to evaluate the impact of threshold tuning

From a compliance perspective, this aligns with the explainability and transparency requirements demanded by regulators. False negatives (missed suspicious activities) are treated as regulatory risks, while false positives are treated as operational inefficiencies.

5. Visualization and Insights

The visualization layer (visualize_results.py) delivers interpretable and investor-ready visual dashboards:

Distribution of transaction sizes by risk label

Heatmaps showing confusion matrices

ROC curves highlighting trade-offs in model performance

Feature importance charts from tree-based models

These visualizations allow both data scientists and financial analysts to interpret model behavior intuitively — a crucial step in bridging technical results with financial decision-making.

Interpretation of Results

The Random Forest and XGBoost models generally outperform linear methods due to their ability to model complex fraud patterns.

ROC-AUC scores above 0.8 indicate strong discriminatory power between suspicious and non-suspicious activity.

However, the goal is not just accuracy — it’s the minimization of missed suspicious cases (false negatives) without overwhelming investigators with false positives.

This mirrors the real-world risk-based approach where:

High-risk transactions are investigated first

Low-risk transactions are processed with minimal human oversight

Thresholds are adjusted based on institutional risk appetite

Technologies Used
Category	Tools / Libraries
Data Handling	pandas, numpy, pyarrow
Machine Learning	scikit-learn, xgboost, imbalanced-learn
Statistical Modeling	statsmodels, scipy
Visualization	matplotlib, seaborn
Automation & Utilities	tqdm, joblib, requests
Development	Python 3.10+, Jupyter, VS Code
Intellectual Summary

This project merges financial domain expertise with technical modeling proficiency.
It demonstrates capability in:

Translating financial risk theory into algorithmic detection systems

Designing scalable ML pipelines aligned with regulatory and audit principles

Understanding economic and compliance consequences of model outputs

In professional environments, this project could be adapted for:

AML transaction monitoring

Insurance claim fraud detection

Credit risk modeling

FinTech product compliance analytics

From a research standpoint, it’s a stepping stone toward explainable AI in financial systems — ensuring that predictive models remain transparent, auditable, and ethically grounded.

Future Improvements

Integration of graph-based anomaly detection to detect hidden relationships between entities.

Addition of streaming analytics (Kafka / Spark) for real-time monitoring.

Incorporation of NLP for textual alert descriptions.

Use of SHAP or LIME for advanced interpretability.

Financial calibration against expected monetary loss using weighted misclassification costs.
