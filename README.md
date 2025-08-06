# 🧠 Parkinson Disease Prediction using Machine Learning

Parkinson's disease is a chronic neurological condition that affects movement and coordination. This project uses machine learning models to predict whether a person has Parkinson’s disease based on biomedical voice measurements and other diagnostic data.

---

##  Dataset

- Source: [Parkinson Disease Dataset](https://www.kaggle.com/)
- Format: CSV
- Entries: 252 patients
- Features: Originally 755 features (after processing reduced to 30 features + 1 target column)
- Target: `class` (0 = Healthy, 1 = Parkinson's Disease)

---

## Objectives

- Explore and preprocess high-dimensional clinical data
- Handle multicollinearity and reduce dimensionality
- Balance class distribution using oversampling
- Train and compare ML models to predict disease
- Evaluate performance using AUC, confusion matrix, and classification report

---

## Libraries Used

- `Pandas`, `NumPy`: Data manipulation
- `Seaborn`, `Matplotlib`: Visualization
- `scikit-learn`: Preprocessing, feature selection, model training
- `XGBoost`: High-performance gradient boosting
- `imblearn`: Handling imbalanced datasets (RandomOverSampler)
- `tqdm`: Progress bar for iterations

---

## Workflow Summary

### 1. Data Exploration
- Verified structure, data types, and confirmed no missing values
- Dataset includes multiple rows per patient

### 2. Data Wrangling
- Grouped data by patient ID (mean of rows)
- Removed highly correlated features (threshold > 0.7)

### 3. Feature Selection
- Normalized using MinMaxScaler
- Selected top 30 features via Chi-square test (SelectKBest)

### 4. Class Balancing & Splitting
- Pie chart showed class imbalance
- Used `RandomOverSampler` to balance classes
- Split into 80% training, 20% validation

### 5. Model Training
Models used:
- Logistic Regression (with balanced class weights)
- Support Vector Classifier (RBF kernel)
- XGBoost Classifier

**Evaluation Metric**: ROC AUC Score

### 6. Best Model
Logistic Regression showed the best performance with consistent training and validation scores.

---

## Model Evaluation

- **Confusion Matrix**
  - TP = 35
  - TN = 10
  - FP = 4
  - FN = 2

- **Classification Report**
  - High precision and recall for class 1 (healthy)
  - Slightly lower recall for class 0 (unhealthy)



