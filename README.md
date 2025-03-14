# 🏥 Healthcare Prediction using Machine Learning

## 📌 Project Overview
This project aims to predict healthcare outcomes using machine learning models such as **Logistic Regression, Random Forest, and XGBoost**. The dataset contains various health indicators, including **smoking history, BMI, age, hypertension, heart disease, and glucose levels**. The goal is to analyze key factors influencing predictions and improve model accuracy through feature engineering and hyperparameter tuning.

## 📂 Project Structure
```
📁 healthcare_prediction
│── 📄 README.md
│── 📁 data                # Contains dataset files
│── 📁 notebooks           # Jupyter notebooks for EDA and modeling
│── 📁 models              # Trained models and performance reports
│── 📁 scripts             # Python scripts for preprocessing and training
│── 📄 requirements.txt    # Dependencies
```

## 📊 Data Preprocessing
- **Data Cleaning**: Handled duplicates, missing values and outliers.
- **Feature Engineering**: Encoded categorical variables and normalized numerical features.
- **Splitting Data**: **80% training, 20% testing**.

## 🏗️ Model Building
### 1️⃣ Logistic Regression
```python
lr_pipe = Pipeline([
    ("processor", processor),
    ("lr_model", LogisticRegression())
])
lr_model = lr_pipe.fit(x_train, y_train)
```
#### 📌 Key Insights
- **Feature Importance**:
  - **Smoking History** had the strongest positive coefficient.
  - **BMI and Age** significantly influenced predictions.

### 2️⃣ Random Forest Classifier
```python
rf_pipe = Pipeline([
    ("processor", processor),
    ("rf_model", RandomForestClassifier())
])
rf_model = rf_pipe.fit(x_train, y_train)
```
#### 📌 Model Performance
- Accuracy: **90.9%**
- F1-score: **91%**
- **Confusion Matrix**:
  - False Positives: **174**
  - False Negatives: **140**

### 3️⃣ XGBoost Classifier
```python
xgb_pipe = Pipeline([
    ("processor", processor),
    ("xgb_model", XGBClassifier())
])
xgb_model = xgb_pipe.fit(x_train, y_train)
```
#### 📌 Model Performance
- Accuracy: **90.75%**
- F1-score: **91%**
- **Feature Importance**:
  - Smoking History (**0.41**) and BMI (**0.26**) were the most influential features.

## 📈 Model Evaluation
### 📌 Logistic Regression Metrics
```python
lr_predict = lr_model.predict(x_test)
lr_f1 = f1_score(y_test, lr_predict)
lr_report = classification_report(y_test, lr_predict)
print(lr_f1, lr_report)
```
- Accuracy: **88.7%**
- F1-score: **88%**

### 📌 XGBoost Metrics
```python
xgb_predict = xgb_model.predict(x_test)
xgb_f1 = f1_score(y_test, xgb_predict)
xgb_report = classification_report(y_test, xgb_predict)
print(xgb_f1, xgb_report)
```
- Accuracy: **90.75%**
- F1-score: **91%**

## 🚀 Next Steps
- **Hyperparameter tuning** for further optimization.
- **Feature selection** to enhance model interpretability.
- **Deploying the best model** using **Streamlit**.


## 🎯 Usage
```python
python train_model.py
```

## 🤝 Contributing
Feel free to fork this repository, create a new branch, and submit a pull request
