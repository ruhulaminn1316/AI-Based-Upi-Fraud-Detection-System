# UPI Fraud Detection

## Overview

This project focuses on detecting fraudulent transactions within the Unified Payments Interface (UPI) system using machine learning techniques. Given the increasing prevalence of digital payments, ensuring the security of such platforms is paramount. This system aims to identify and prevent potential fraudulent activities by analyzing transaction patterns and anomalies.

## Features

- **Data Analysis**: Examination of transaction data to identify patterns indicative of fraud.
- **Machine Learning Models**: Implementation of algorithms to classify transactions as legitimate or fraudulent.
- **Interactive Dashboard**: A user-friendly interface for real-time monitoring and analysis of transaction data.

## Project Structure

- **`dataset/`**: Contains the UPI transaction data used for analysis and model training.
- **`src/`**: Includes the source code for data processing, model training, and evaluation.
- **`static/`**: Houses static files such as CSS and images for the web interface.
- **`templates/`**: Contains HTML templates for the web application's frontend.
- **`app.py`**: The main Flask application file that integrates all components and runs the web server.
- **`upi_fraud_dataset.csv`**: The dataset comprising UPI transaction records used for model training and testing.

## Installation

1. **Clone the Repository**:

2. **Create a Virtual Environment**:

3. **Install Dependencies**:

4. **Set Up the Database**:
   Ensure that the `upi_fraud_dataset.csv` file is placed in the `dataset/` directory. This dataset will be used for training and evaluating the machine learning models.

## Usage

1. **Run the Application**:

2. **Access the Web Interface**:
   Open your web browser and navigate to `http://127.0.0.1:5000/`. Here, you can upload transaction data, view analysis results, and monitor potential fraudulent activities.

   
Play the video below: ⬇️

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/7bQwA4Ntvqk/0.jpg)](https://www.youtube.com/watch?v=7bQwA4Ntvqk)


## How It Works

1. **Data Collection**:
   - The system uses a dataset of UPI transactions containing details such as transaction ID, amount, timestamp, sender and receiver details, and status.

2. **Data Preprocessing**:
   - The raw transaction data undergoes cleaning and transformation, including handling missing values, encoding categorical variables, and normalizing numerical fields.

3. **Feature Engineering**:
   - Important features such as transaction frequency, amount deviation, and user behavior patterns are extracted to improve model accuracy.

4. **Model Training & Prediction**:
   - Machine learning models such as Random Forest, Decision Trees, or Neural Networks are trained on labeled transaction data.
   - The trained model classifies new transactions as either "Legitimate" or "Fraudulent."

5. **Fraud Detection & Alerting**:
   - When a transaction is classified as fraudulent, an alert is generated, notifying the user or the system administrator.

6. **Dashboard & Visualization**:
   - The results are displayed on an interactive dashboard that provides insights into fraud trends and high-risk transactions.




