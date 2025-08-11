# 🌱 Crop Water Requirement Analysis using AI/ML

![Python](https://img.shields.io/badge/Python-3.9-blue)
![CatBoost](https://img.shields.io/badge/Model-CatBoost-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This project predicts the **water requirements for various crops** using CatBoost, based on parameters like temperature, humidity, rainfall, soil type, crop type, and growth stage. It includes data preprocessing, feature engineering, and a **comparative study with Random Forest, XGBoost, and LightGBM** to ensure robust and accurate predictions.

The goal is to help farmers, researchers, and policymakers **optimize irrigation**, reduce water wastage, and support **sustainable agriculture**.

## 🚀 Features
- Predicts water requirements with high accuracy
- Handles both categorical and numerical features efficiently
- Includes performance comparison with three ML models
- Evaluation metrics: RMSE, MAE, and R² score
- Easy to adapt for other agricultural prediction tasks

## 📊 Dataset
- **Source:** Custom prepared dataset  
- **Features include:** Crop type, soil type, temperature, humidity, rainfall, and growth stage  
- **Target:** Water requirement (liters/day or mm/day)

## ⚙️ Tech Stack
- Python 3.9
- CatBoost
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Jupyter Notebook

## 📈 Workflow
1. **Data Preprocessing**: Handle missing values, encode categorical features, scale numeric values where needed.
2. **Feature Engineering**: Select and create meaningful parameters affecting crop water needs.
3. **Model Training**: Train CatBoost and compare with Random Forest, XGBoost, LightGBM.
4. **Evaluation**: Use RMSE, MAE, R² score to compare models.
5. **Results**: CatBoost provides the most accurate predictions.

## 📦 Installation & Usage
```bash
# Clone this repository
git clone https://github.com/04-shreyas/Crop-Water-Requirement-Analysis-using-AIML.git

# Navigate to the project folder
cd Crop-Water-Requirement-Analysis-using-AIML

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Crop_Water_Prediction.ipynb
