# Diabetes Risk Prediction Project

## Project Overview
This project aims to develop an intelligent system to predict the risk of diabetes in patients using clinical features such as Glucose levels, Blood Pressure, BMI, Diabetes Pedigree Function, Age, and others. The project involves data exploration, preprocessing, clustering, supervised classification, and model evaluation.

---

## Features

- **Data Loading and Exploration:**  
  Import and inspect the dataset, understand the structure, check for missing values and duplicates, and analyze numerical variable distributions.

- **Data Preprocessing:**  
  Handle missing and outlier values using statistical techniques such as Z-score, IQR, and boxplots. Normalize or standardize numerical features as required.

- **Clustering Analysis:**  
  Implement K-Means clustering, determine the optimal number of clusters using the elbow and silhouette methods, and assign cluster labels to each observation.

- **Cluster Analysis and Risk Categorization:**  
  Analyze cluster characteristics to identify high-risk groups based on critical thresholds for Glucose, BMI, and Diabetes Pedigree Function. Add a risk category for each patient.

- **Supervised Classification and Model Evaluation:**  
  Train multiple classification models including Random Forest, SVM, Gradient Boosting, Decision Tree, Logistic Regression, and XGBoost. Handle class imbalance with over- or under-sampling techniques. Evaluate models using metrics such as confusion matrix, accuracy, recall, and F1-score. Optimize hyperparameters with GridSearchCV or RandomizedSearchCV and select the best-performing model. Save the final model for future predictions.

---

## Technologies and Libraries

```bash
Python 3.x
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn
xgboost
joblib
streamlit
```


## Setup Instructions

git clone https://github.com/khalid-io-dev/Prediction_of_Diabetes_Risk_model

# Navigate to the project directory
cd diabetes-risk-prediction

# Run the Streamlit application
streamlit run app.py

## Usage

Load the dataset and perform exploratory data analysis.

Preprocess the data to handle missing values, outliers, and scaling.

Train K-Means clustering and visualize the cluster distributions.

Assign risk categories based on cluster analysis.

Train and evaluate supervised classification models.

Use the Streamlit interface to input patient data and visualize real-time diabetes risk predictions.