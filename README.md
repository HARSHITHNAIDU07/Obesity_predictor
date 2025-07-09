# 🏥 Obesity Category Predictor

A user-friendly Streamlit web application that predicts a person's **obesity category** based on lifestyle, demographic, and health factors.

This project uses a **Random Forest Classifier** trained on the `ObesityDataSet_raw_and_data_sinthetic.csv` dataset and provides an interactive form for users to input their details and get a prediction.

---

## 🚀 Features

- 🧠 Predicts exact obesity category (e.g., `Normal_Weight`, `Obesity_Type_I`, etc.)
- 📋 Form-based user input
- 🧮 Real-time prediction using a trained machine learning model
- ⚙️ Automatic encoding and scaling
- 💻 Clean, responsive interface powered by Streamlit

---

## 📊 Obesity Categories

The model predicts one of the following categories:

- Insufficient_Weight  
- Normal_Weight  
- Overweight_Level_I  
- Overweight_Level_II  
- Obesity_Type_I  
- Obesity_Type_II  
- Obesity_Type_III

---

## 📁 Dataset

- Source: [`ObesityDataSet_raw_and_data_sinthetic.csv`](https://www.kaggle.com/datasets/sanchesalvador/obesity-dataset)
- File path used in the app: `/content/ObesityDataSet_raw_and_data_sinthetic.csv`

You can change the path if your file is stored elsewhere.

---


## 🛠 Setup Instructions

### 🔧 Install Dependencies

```bash
pip install streamlit pandas scikit-learn
```


##▶️ Run the App
```bash
streamlit run app.py
```

##🌐 Open in Browser
```bash
http://localhost:8501
```

##🧬 Model Details

- Algorithm: Random Forest Classifier

- Preprocessing:

- Label encoding for categorical variables

- StandardScaler for continuous variables

= Training: Model is trained and cached on first load

##👨‍💻 Author
- Made By Harshith Naidu

