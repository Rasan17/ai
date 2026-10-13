# Universal Medical ML/NeuralNet Predictor

An interactive, in-browser client-side Machine Learning and Deep Neural Network predictive modeling suite for **Clinical Medicine and Healthcare Research**, running 100% in the web browser with zero server dependencies and complete patient data privacy.

Conceived, designed and directed by **Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)** ([g_narenthiran@hotmail.com](mailto:g_narenthiran@hotmail.com)), Neurosurgery Research Listserv, United Kingdom (Implemented with Antigravity AI pair programming).

**&copy; 2026 Dr G Narenthiran. All rights reserved.**

> ⚠️ **Clinical & Research Disclaimer:** The Universal Medical ML/NeuralNet Predictor is designed as an interactive scientific computing and educational tool for clinical medical research across healthcare disciplines. It must not be utilized as the sole determinant for treatment or clinical interventions without independent medical evaluation and validation.

---

## 🔬 Universal Clinical Scope & Purpose

Translational medicine, clinical research, and electronic health record (EHR) analytics often involve heterogeneous multimodal datasets spanning:
- **Demographics & Physiological Vitals:** Age, BMI, Systolic/Diastolic BP, Heart Rate, Respiratory Rate.
- **Biomarkers & Laboratory Panels:** Serum Creatinine, HbA1c, Troponin, eGFR, Inflammatory Markers.
- **Clinical Severity Scores & Ordinal Scales:** NYHA Heart Failure Class, modified Rankin Scale (mRS), Glasgow Coma Scale (GCS), Charlson Comorbidity Index, Pre-op Gait Scores.
- **Diagnostic Classifications & Imaging Findings:** Binary treatment responsiveness (Responder vs Non-Responder), 1-year cardiovascular risk, mortality, length of stay, or continuous functional recovery scores.

The **Universal Medical ML/NeuralNet Predictor** provides clinical researchers, medical statisticians, and clinicians with a self-contained, client-side predictive suite capable of handling any tabular medical dataset.

---

## ⚙️ Core System Architecture & Workflow

### 1. Complete Client-Side Data Privacy (GDPR & HIPAA Compliant)
- Runs locally in modern browsers via HTML5, Vanilla JavaScript, and TensorFlow.js.
- Patient health records and uploaded CSVs are processed solely in client memory—**zero data is transmitted to external servers**.

### 2. Flexible Variable Role & Scale Configuration
- **Variable Roles:**
  - `Meta / ID`: Identifiers excluded from modeling (e.g. `Patient_MRN`, `Study_ID`).
  - `Feature`: Clinical predictors and biomarkers.
  - `Target`: The primary clinical outcome (Binary, Multi-class, or Continuous).
- **Measurement Scales:**
  - `Continuous`: Interval / ratio parameters (Blood pressure, lab titers, dimensions, scores).
  - `Ordinal`: Ordered discrete clinical rankings (e.g. NYHA class 0–4, severity grades).
  - `Nominal`: Categorical attributes (Sex, comorbidities present/absent, treatment arms).

### 3. Tailored Missing Data Imputation Engine
Context-aware imputation strategies dynamically selectable based on variable scale:
- **Continuous:** *Median* (robust against clinical outliers), *Mean*, *K-Nearest Neighbors (KNN Imputation)*, *Mode*, *Constant*, or *Drop Rows*.
- **Ordinal:** *Mode / Most Frequent Rank*, *Median Rank*, *Constant Rank*, or *Drop Rows*.
- **Nominal:** *Mode / Most Frequent Category*, *New Category ("Missing" / "Unknown")*, or *Drop Rows*.

### 4. Deep Learning & Machine Learning Algorithms with Advanced Fine-Tuning
Every model includes fine-tuning controls specifically designed to maximize predictive accuracy, adapt to patient cohort characteristics, and prevent overfitting:

- **Ensemble Super Learner (Meta-Ensemble):**
  - Clinical consensus ensemble combining Deep Neural Networks, Random Forests, Logistic/Ridge Regression, and k-NN.
  - **Fine-Tuning Controls:**
    - Combination Strategy: *Soft Voting* (probability blending), *Performance-Weighted* (dynamic validation weighting), or *Hard Majority Voting*.
    - Validation Weighting Power ($\gamma$): Linear, Quadratic ($\gamma=2$ sharply boosts best models), or Cubic.
    - Capacity Presets: *Balanced*, *Fast*, or *High-Capacity Deep Ensemble*.
    - Dynamic constituent model inclusion checkboxes.
- **Deep Neural Network (MLP / TensorFlow.js):**
  - **Fine-Tuning Controls:**
    - Optimizer selection: *Adam* (Adaptive Moments), *RMSprop*, or *SGD* (Stochastic Gradient Descent).
    - Batch Size: *8* (Fine mini-batch), *16*, *32*, *64*, or *Full Batch* (entire cohort).
    - Architecture: Hidden Layer 1, Hidden Layer 2, and optional Hidden Layer 3 units.
    - Regularization: Dropout rate ($0$ to $0.5$) and L2 Weight Decay ($\lambda = 0, 0.0001, 0.001, 0.01$) to prevent overfitting on clinical cohorts.
    - Activation: *ReLU*, *Tanh*, or *Sigmoid*.
- **Random Forest:**
  - **Fine-Tuning Controls:**
    - Number of Estimator Trees ($5$ to $100$) and Maximum Tree Depth ($2$ to $15$).
    - Minimum Samples to Split ($2, 4, 8, 16$) for leaf node regularization against clinical outliers.
    - Feature Subsampling Ratio ($\sqrt{p}$, $50\%$, $75\%$, $\log_2(p)$, or $100\%$ bagging mode).
- **Logistic / Ridge Regression:**
  - **Fine-Tuning Controls:**
    - Learning rate and iterations ($20$ to $500$).
    - L2 Ridge Penalty ($\lambda = 0, 0.001, 0.01, 0.05, 0.1, 0.5$) for coefficient shrinkage.
    - Optimization Solver: *Momentum GD* (accelerated gradient descent) or *Standard GD*.
- **k-Nearest Neighbors (k-NN):**
  - **Fine-Tuning Controls:**
    - Number of Neighbors ($k = 1$ to $25$).
    - Distance Metric: *Euclidean ($L_2$)*, *Manhattan ($L_1$ Cityblock - robust to outliers)*, or *Chebyshev ($L_\infty$)*.
    - Neighbor Weighting: *Uniform* or *Distance-Weighted ($1/\text{dist}$)* to give closer patient cases higher influence.
- **Gaussian Naive Bayes:**
  - **Fine-Tuning Controls:**
    - Variance Smoothing ($\epsilon = 10^{-9}, 10^{-4}, 10^{-2}, 0.1$) for numerical stability with sparse or noisy biomarkers.
    - Class Prior Calculation: *Empirical* (cohort frequency), *Balanced / Uniform* (equal $1/K$ priors to correct for class imbalance), or *Laplace-Smoothed*.

### 5. Configurable Train/Test Split & Two-Phase 100% Retraining Protocol
- **Configurable Validation Partition (Default: 80% Train / 20% Test):**
  - Clinicians and researchers can dynamically configure the training percentage (50% to 95%) via an interactive slider, numeric percentage inputs, or quick presets (`80/20`, `70/30`, `85/15`, `90/10`).
- **Two-Phase Clinical Retraining Pipeline:**
  - **Phase 1 (Validation Benchmark):** The algorithm trains on the chosen training partition (default 80%) and benchmarks true out-of-sample performance on the held-out test partition (default 20%), reporting Test Accuracy, Macro F1, Precision, Recall, ROC-AUC, and Confusion Matrix.
  - **Phase 2 (100% Production Retraining):** Once validation accuracy has been recorded and displayed, the system automatically retrains the final model on **100% of all cohort data** ($N_{total}$) so that deployed single-patient and batch inference incorporates every piece of patient evidence available.

### 6. Dual Inference Modes
- **Single Patient Predictor:**
  - Dynamically generated clinical form tailored to the features (sliders, rank selectors, categorical dropdowns).
  - Outputs predicted class/value, confidence score, and probability distribution bar meters.
- **Batch Prediction Facility:**
  - Upload an unseen cohort CSV with multiple patient records.
  - Computes predictions simultaneously for all rows.
  - Enriches the dataset with `Predicted_[Target]` and `Prediction_Confidence` and exports as `.csv`.

---

## 📁 Pre-Loaded Sample Medical Cohorts

The suite includes built-in clinical datasets for instant experimentation:
1. **🫀 Cardiometabolic Risk Cohort (`universal_medical_cohort.csv`):** Multimodal cardiovascular and metabolic risk prediction (Age, BMI, BP, Creatinine, HbA1c, NYHA class, ECG ischemia) with missing cells.
2. **🧠 Neurosurgical Shunt Response Cohort (`medical_shunt_cohort.csv`):** Hakim-Adams triad, Evans' Index, Callosal Angle, DESH, CSF dynamics ($R_{out}$, PVI), predicting surgical response.
3. **📈 Functional Recovery Score Cohort (`medical_recovery_score.csv`):** Continuous regression dataset predicting post-operative clinical improvement points.
4. **📂 Unseen Batch Cases (`medical_batch_unseen_cases.csv`):** 10 patient cases ready for immediate batch prediction.

---

## 🚀 How to Run

Because the application is built entirely with client-side web technologies, no server installation, node packages, or build steps are required:
1. Double-click `index.html` to open directly in any modern web browser (Safari, Chrome, Edge, Firefox).
2. Alternatively, serve via any local web server:
   ```bash
   cd "universal-medical-ml-predictor"
   python3 -m http.server 8080
   ```
   Then navigate to `http://localhost:8080`.

---

## 👨‍⚕️ Author & Copyright Attribution

- **Conception, Design & Direction:** **Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)**
- **Email:** [g_narenthiran@hotmail.com](mailto:g_narenthiran@hotmail.com)
- **Affiliation:** Neurosurgery Research Listserv, United Kingdom
- **Implementation:** Implemented with Antigravity AI pair programming
- **Copyright:** &copy; 2026 Dr G Narenthiran. All rights reserved.
