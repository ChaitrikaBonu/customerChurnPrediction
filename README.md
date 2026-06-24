# Customer Churn Prediction System

## Overview

The Customer Churn Prediction System is a Machine Learning project designed to predict whether a customer is likely to leave a service based on historical customer information. The application uses a trained Random Forest Classifier and provides predictions through an interactive Streamlit dashboard.

This project helps businesses identify customers at risk of churn and take proactive measures to improve customer retention.

---

## Features

* Customer churn prediction using Machine Learning
* Interactive Streamlit web application
* Real-time prediction results
* Churn probability estimation
* Data preprocessing and feature encoding
* Easy-to-use user interface

---

## Dataset

Dataset Used: Telco Customer Churn Dataset

Features include:

* Gender
* Senior Citizen Status
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract Type
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

Target Variable:

* Churn (Yes/No)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Pickle
* Matplotlib

---

## Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Label Encoding of Categorical Features
4. Train-Test Split
5. Model Training using Random Forest Classifier
6. Model Evaluation
7. Model Serialization using Pickle
8. Streamlit Dashboard Development
9. Deployment

---

## Model Performance

Evaluation Metrics:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* ROC-AUC Score

The Random Forest model was selected due to its strong performance and ability to handle mixed feature types effectively.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── churn_model.pkl
├── requirements.txt
├── README.md
├── Customer_Churn_Prediction.ipynb
└── dataset.csv
```

---

## Future Enhancements

* XGBoost Model Integration
* SMOTE for Class Imbalance Handling
* Feature Importance Visualization
* Model Comparison Dashboard
* Cloud Deployment
* Advanced Business Analytics

---

## Results

The system predicts whether a customer is likely to churn and displays the probability score, enabling organizations to identify high-risk customers and implement targeted retention strategies.

---

## Author

Chaithu

Artificial Intelligence & Machine Learning Enthusiast
