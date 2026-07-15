# 🤖 Machine Learning Algorithms & Projects

A collection of hands-on machine learning notebooks covering supervised learning algorithms — regression, classification, and ensemble methods — implemented using **scikit-learn**, along with a complete end-to-end capstone project (**ShopSmart E-Commerce Revenue Prediction**). Built while strengthening my ML fundamentals as part of my Data Analytics coursework and AI/ML Scholars journey.

## 📁 Project Structure

```
machine-learning-algorithms/
│
├── Regression/
│   ├── linear_regression.ipynb              # Linear Regression on insurance cost data
│   ├── project_inear_regression.ipynb       # Linear Regression on house price prediction
│   ├── Lasso_Regression.ipynb               # Lasso (L1) Regression
│   ├── Decision_random_regressor.ipynb      # Decision Tree Regressor (diabetes dataset)
│   └── SVR_implementation.ipynb             # Support Vector Regression (kernels + tuning)
│
├── Classification/
│   ├── Logistic_regression.ipynb            # Logistic Regression (heart disease dataset)
│   ├── logistic_project.ipynb               # Logistic Regression (employee turnover)
│   ├── KNN.ipynb                            # K-Nearest Neighbors + GridSearchCV + Pipelines
│   ├── Gussian_Naive.ipynb                  # Gaussian Naive Bayes
│   ├── SVC_implementation.ipynb             # Support Vector Classifier (Iris dataset, kernels)
│   ├── decision_tree_classifier.ipynb       # Decision Tree (pre/post-pruning, Titanic dataset)
│   ├── iris_flower.ipynb                    # Multi-class classification on Iris dataset
│   └── random_forest.ipynb                  # Random Forest Classifier (Titanic dataset)
│
├── Ensemble Methods/
│   ├── ensemble_heterogenous.ipynb          # Voting & Stacking classifiers
│   ├── adaboost.ipynb                       # AdaBoost (classifier + regressor)
│   ├── gradient_boosting.ipynb              # Gradient Boosting (classifier + regressor)
│   └── xgboost_classification.ipynb         # XGBoost Classifier
│
├── ShopSmart E-Commerce Project/
│   ├── shopsmart_E-commerce.ipynb           # Full EDA + ML pipeline (capstone)
│   ├── shop_smart.ipynb                     # Hyperparameter tuning variant
│   └── shop_smart_ecommerce.csv             # Dataset
│
├── data/
│   ├── insurance.csv
│   ├── Iris.csv
│   ├── 1-heart.csv
│   ├── HousePricePrediction.csv
│   └── employee_turnover.csv
│
└── README.md
```

## 🚀 What's Inside

### 📈 Regression
- **Linear Regression** — insurance cost prediction with interaction features, and house price prediction with feature engineering
- **Lasso Regression** — L1-regularized regression to reduce overfitting and perform feature selection
- **Decision Tree Regressor** — non-linear regression on the diabetes dataset
- **Support Vector Regression (SVR)** — linear, polynomial, and sigmoid kernels with hyperparameter tuning via GridSearchCV

### 🧩 Classification
- **Logistic Regression** — binary classification on heart disease and employee turnover datasets, with feature scaling and full evaluation (accuracy, precision, recall, F1, confusion matrix)
- **K-Nearest Neighbors (KNN)** — hyperparameter selection (K=5 vs K=7), cross-validation with GridSearchCV, and sklearn Pipelines
- **Gaussian Naive Bayes** — probabilistic classification on heart disease data
- **Support Vector Classifier (SVC)** — kernel comparison (linear, sigmoid) and the effect of the `C` parameter, with feature scaling
- **Decision Tree Classifier** — pre-pruning vs. post-pruning strategies on the Titanic dataset
- **Random Forest Classifier** — missing-value imputation, feature encoding, and comparison against a single decision tree
- **Iris Flower Classification** — multi-class classification with label encoding

### 🌲 Ensemble Methods
- **Voting & Stacking Classifiers** — combining heterogeneous base models (Logistic Regression, SVC, Decision Tree)
- **AdaBoost** — boosting with decision stump base estimators, for both classification and regression
- **Gradient Boosting** — sequential boosting for both classification and regression tasks
- **XGBoost** — gradient-boosted trees using the `xgboost` library

### 🛒 ShopSmart E-Commerce Project (Capstone)
An end-to-end applied ML project predicting online shopper purchase intent (`Revenue`: True/False):
- Exploratory Data Analysis — class balance, visitor type, weekend traffic, monthly trends
- Distribution analysis of key behavioral features (`ProductRelated`, `BounceRates`, `ExitRates`, `PageValues`, etc.)
- Feature relationships with the target via boxplots
- Feature encoding and preprocessing with `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`)
- Model pipeline using `Pipeline` + `DecisionTreeClassifier`
- Model evaluation with F1-score, classification report, and confusion matrix
- Hyperparameter tuning with `GridSearchCV`

## 🛠️ Tech Stack

- **Python 3**
- **scikit-learn** — models, pipelines, preprocessing, model selection & evaluation
- **XGBoost**
- **Pandas / NumPy** — data handling and numerical operations
- **Matplotlib / Seaborn** — visualization and EDA
- **Jupyter Notebook**

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/tal123ph/machine-learning-algorithms.git
   cd machine-learning-algorithms
   ```

2. Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn xgboost matplotlib seaborn jupyter
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open any notebook and run the cells. Notebooks that read a CSV expect it in the same folder (see the `data/` folder / dataset paths above).

## 🎯 Purpose

This repository documents my progression through core supervised machine learning algorithms — from simple linear models to ensemble and boosting methods — and demonstrates applying them end-to-end in a real dataset (ShopSmart E-Commerce). It serves as both a personal learning reference and a portfolio piece as I build toward a career in **AI Engineering and Data Science**.

## 👤 Author

**Muhammad Talha**
Data Analytics Student | Aspiring AI Engineer & Data Scientist
- GitHub: [@tal123ph](https://github.com/tal123ph)
- LinkedIn: [muhammad-talha12b](https://www.linkedin.com/in/muhammad-talha12b)

## 📄 License

This project is open source and available for learning purposes.
