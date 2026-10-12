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

### 4. Deep Learning & Machine Learning Algorithms
- **Ensemble Super Learner (Meta-Ensemble):**
  - Clinical consensus ensemble combining Deep Neural Networks, Random Forests, Logistic/Ridge Regression, and k-NN.
  - Supports **Soft Voting** (calibrated class probability blending), **Performance-Weighted Blending** (inversely weighted by validation error), and **Hard Majority Voting**.
  - Provides clinician transparency with real-time constituent model weights and prediction breakdowns.
- **Deep Neural Network (MLP / TensorFlow.js):**
  - Configurable hidden layers (e.g., 64, 32 units), Dropout regularization, ReLU / Tanh / Sigmoid activations, Adam optimizer, with live epoch-by-epoch loss tracking.
- **Random Forest:**
  - Bootstrap ensemble of decision trees with Gini impurity / variance reduction splits and feature importance rankings.
- **Logistic / Ridge Regression:**
  - L2-regularized linear/softmax regression providing interpretable odds ratios and regression coefficients.
- **k-Nearest Neighbors (k-NN):**
  - Non-parametric distance-based instance classifier and regressor in standardized feature space.
- **Gaussian Naive Bayes:**
  - Fast probabilistic classifier computing posterior probabilities via Gaussian class-conditional likelihoods.

### 5. Dual Inference Modes
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
