# 🏥 Chest Cancer Detection using MLOps & DVC

![GitHub stars](https://img.shields.io/github/stars/AryanDhanuka10/end-to-end-ml_project-chest-cancer-detection-using-mlops-and-dvc?style=social)
![GitHub forks](https://img.shields.io/github/forks/AryanDhanuka10/end-to-end-ml_project-chest-cancer-detection-using-mlops-and-dvc?style=social)
![GitHub license](https://img.shields.io/github/license/AryanDhanuka10/end-to-end-ml_project-chest-cancer-detection-using-mlops-and-dvc)

## 📌 Project Overview
This is an **End-to-End Machine Learning project** for **Chest Cancer Detection** using **MLOps and DVC (Data Version Control)**. The model predicts whether a given chest X-ray falls into one of the four categories:

- **Normal**
- **Adenocarcinoma (Left Lower Lobe, T2 N0 M0 I B)**
- **Large Cell Carcinoma (Left Hilum, T2 N2 M0 IIIA)**
- **Squamous Cell Carcinoma (Left Hilum, T1 N2 M0 IIIA)**

🚀 **Live Demo:** [Streamlit App](https://end-to-end-mlproject-chest-cancer-detection-using-mlops-and-dv.streamlit.app/)

## 📂 Project Structure
```
📁 end-to-end-ml_project-chest-cancer-detection-using-mlops-and-dvc
│── 📂 dvc/                      # DVC configuration for dataset & model versioning
│── 📂 .github/workflows/        # CI/CD automation using GitHub Actions
│── 📂 config/                   # Configuration files for training & inference
│── 📂 research/                 # Jupyter notebooks for exploratory data analysis
│── 📂 src/cnnClassifier/        # CNN architecture & ML pipeline scripts
│── 📂 templates/                # HTML templates for web app UI
│── 📄 app.py                    # Flask-based web API
│── 📄 streamlit_app.py          # Streamlit app for predictions
│── 📄 FastAPI_app.py            # FastAPI backend for RESTful predictions
│── 📄 requirements.txt          # Dependencies for setting up the project
```

## ⚙️ Installation & Setup
Follow these steps to set up and run the project:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AryanDhanuka10/end-to-end-ml_project-chest-cancer-detection-using-mlops-and-dvc.git
cd end-to-end-ml_project-chest-cancer-detection-using-mlops-and-dvc
```

### 2️⃣ Create a Virtual Environment
```bash
python3 -m venv chest
source chest/bin/activate  # For Windows use: chest\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Install DVC & Fetch Data
```bash
pip install dvc
dvc pull  # Pull dataset & model files
```

### 5️⃣ Model Training & Evaluation
Modify the `config/config.yaml` file with dataset paths and hyperparameters.
```bash
python src/cnnClassifier/train.py  # Train the CNN Model
```

### 6️⃣ Run Streamlit Web App
```bash
streamlit run streamlit_app.py
```

## 🚀 Technologies Used
- **Deep Learning:** CNN (Convolutional Neural Networks)
- **MLOps:** DVC (Data Version Control)
- **Web Frameworks:** Flask, FastAPI, Streamlit
- **CI/CD:** GitHub Actions
- **Version Control:** Git & DVC

## 📊 Model Performance
| Model                  | Accuracy |
|------------------------|----------|
| CNN                    | 63.25%   |
| Logistic Regression    | 72.14%   |
| Random Forest          | TBD      |

## 🔗 Useful Links
- **GitHub Repository:** [GitHub](https://github.com/AryanDhanuka10/end-to-end-ml_project-chest-cancer-detection-using-mlops-and-dvc)
- **Live Web App:** [Streamlit App](https://end-to-end-mlproject-chest-cancer-detection-using-mlops-and-dv.streamlit.app/)
- **Author:** [Aryan Dhanuka](https://www.linkedin.com/in/aryan-dhanuka-07b338292/)

## 📜 License
This project is licensed under the **MIT License**.

---
🌟 **If you found this project helpful, give it a star ⭐ on GitHub!**

