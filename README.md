# Credit Scoring Model

## Project Overview

This project predicts the creditworthiness of customers using Machine Learning techniques. The model analyzes customer financial information and classifies applicants as either Good Credit Risk or Bad Credit Risk.

The project compares multiple classification algorithms and selects the best-performing model based on evaluation metrics.

This project was developed as part of the CodeAlpha Machine Learning Internship Program.

---

## Objectives

* Predict customer credit risk using historical financial data.
* Compare multiple machine learning algorithms.
* Evaluate model performance using standard classification metrics.
* Save the best-performing model for future predictions.

---

## Dataset

Dataset Used: German Credit Dataset (Statlog German Credit Data)

Characteristics:

* 1000 customer records
* 24 input features
* 1 target variable
* Binary classification problem

Target Variable:

* 1 = Good Credit
* 2 = Bad Credit

Features include:

* Credit History
* Credit Amount
* Duration
* Employment Status
* Savings Account Information
* Personal Information
* Existing Credits
* Housing Status
* Age
* Other Financial Attributes

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

---

## Machine Learning Algorithms

### Logistic Regression

A statistical classification algorithm used as a baseline model.

### Decision Tree Classifier

A tree-based model that learns decision rules from data.

### Random Forest Classifier

An ensemble learning method that combines multiple decision trees to improve prediction accuracy.

---

## Project Workflow

### 1. Data Loading

The dataset is loaded from the German Credit Data file.

### 2. Data Preprocessing

* Feature extraction
* Target variable preparation
* Train-test splitting

### 3. Model Training

The following models are trained:

* Logistic Regression
* Decision Tree
* Random Forest

### 4. Model Evaluation

Performance metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

### 5. Model Selection

The model with the highest accuracy is selected as the final model.

### 6. Model Saving

The best model is saved using Joblib.

---

## Project Structure

```text
Credit_Scoring_Model/
│
├── data/
│   └── german.data-numeric
│
├── credit_scoring.py
│
├── best_credit_model.pkl
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Credit_Scoring_Model.git
```

Move into the project directory:

```bash
cd Credit_Scoring_Model
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the following command:

```bash
python credit_scoring.py
```

---

## Output

The program will:

* Train three machine learning models.
* Display evaluation metrics.
* Generate confusion matrices.
* Compare model performance.
* Save the best model as:

```text
best_credit_model.pkl
```

---

## Sample Evaluation Metrics

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* ROC-AUC Score

---

## Future Enhancements

* Hyperparameter Tuning using GridSearchCV
* Cross Validation
* ROC Curve Visualization
* Precision-Recall Curve
* Feature Importance Analysis
* Streamlit Web Application
* Model Deployment using Flask
* Explainable AI using SHAP

---

## Learning Outcomes

Through this project, the following concepts are demonstrated:

* Supervised Machine Learning
* Classification Algorithms
* Data Preprocessing
* Model Evaluation
* Feature Analysis
* Model Persistence

---

## Conclusion

The Credit Scoring Model successfully predicts customer credit risk using machine learning techniques. By comparing Logistic Regression, Decision Tree, and Random Forest algorithms, the project identifies the most effective model for credit risk assessment and demonstrates the practical application of machine learning in financial decision-making.



Author 
Priyal Parmar 

