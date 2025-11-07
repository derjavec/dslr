# 🧙 Hogwarts House Prediction

Predict Hogwarts Houses for students using **logistic regression** and feature selection.  
This project implements both a **custom logistic regression model** and a comparison with **scikit-learn's logistic regression**, including statistical feature selection to avoid data leakage.

---

## 🌟 Features

- Train-test split with reproducible shuffling
- Feature selection based on:
  - Mean and median dispersion
  - Standard deviation and interquartile range
  - Multicollinearity analysis
- Custom logistic regression implementation
- Comparison with scikit-learn's logistic regression
- Generates CSV files with predictions and evaluation scores
- Fully reproducible results

---

## 🛠 Requirements

- Python 3.13+
- pandas
- numpy
- scikit-learn

Install dependencies:

```bash
pip install pandas numpy scikit-learn

