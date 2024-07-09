# Predictive Loan Status Analysis

This project aims to predict the loan status of applicants based on various features using different machine learning models. The analysis involves data preprocessing, visualization, and model evaluation to determine the best-performing model.

## Table of Contents
- [Dataset Information](#dataset-information)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Dataset Information

The dataset used for this analysis contains information about loan applicants, including:
- Gender
- Marital Status
- Dependents
- Education
- Self-Employment Status
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status

## Data Preprocessing

The preprocessing steps include:
1. **Handling Missing Values**: 
   - Categorical columns (Gender, Married, Dependents, Self_Employed, Credit_History) filled with the mode.
   - Numerical columns (LoanAmount, Loan_Amount_Term) filled with the mean and median respectively.
2. **Encoding Categorical Variables**: Using Label Encoding for categorical features.
3. **Transforming Numerical Features**: Applying square root transformation to normalize distributions.
4. **Dropping Irrelevant Columns**: Removing the Loan_ID column.

## Exploratory Data Analysis

Various visualizations were created to understand the distribution of features and their relationship with the loan status:
- Count plots for categorical features.
- Histograms and boxplots for numerical features before and after transformation.

## Feature Engineering

Label encoding was applied to categorical features:
- Gender, Married, Education, Self_Employed, Property_Area, and Loan_Status were encoded.

## Model Training and Evaluation

The dataset was split into training and testing sets with a 90-10 split. The following models were trained and evaluated:

1. **Logistic Regression**: 
   - Hyperparameter tuning using GridSearchCV.
   - Accuracy: 80.65%
2. **Support Vector Machine (SVM)**: 
   - Kernel: RBF, Gamma: Auto, C: 6
   - Accuracy: 78.05%
3. **Decision Tree**: 
   - Criterion: Gini, Splitter: Random
   - Accuracy: 78.05%
4. **Random Forest**: 
   - Criterion: Entropy, N_estimators: 120
   - Accuracy: 78.05%
5. **K-Nearest Neighbors (KNN)**: 
   - N_neighbors: 13
   - Accuracy: 78.05%
6. **Categorical Naive Bayes (NB)**: 
   - Accuracy: 69.51%

## Results

| Model                    | Accuracy Score (%) |
|--------------------------|--------------------|
| Logistic Regression      | 80.65              |
| Support Vector Machine   | 78.05              |
| Decision Tree            | 78.05              |
| Random Forest            | 78.05              |
| K-Nearest Neighbors      | 78.05              |
| Categorical Naive Bayes  | 69.51              |

Logistic Regression outperformed other models with an accuracy score of 80.65%.

## Conclusion

This project demonstrates the application of various machine learning models for predicting loan status. Logistic Regression emerged as the best model for this task, achieving the highest accuracy. Further improvements could be made by exploring more advanced feature engineering techniques and hyperparameter tuning.
