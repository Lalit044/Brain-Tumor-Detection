# 🧠 AI Brain Tumor Detection System

An AI-powered web application for multi-class brain tumor classification using Deep Learning and Explainable AI (Grad-CAM).

This system detects brain tumors from MRI images and provides:

- Tumor classification (4 classes)
- Confidence score
- Risk level indicator
- AI focus visualization (Grad-CAM)
- Confusion matrix performance evaluation
- Downloadable PDF medical report

---

## 🚀 Features

### 🔍 1. Multi-Class Tumor Detection
Classifies MRI scans into:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

Built using **MobileNetV2 Transfer Learning**.

---

### 🧠 2. Explainable AI (Grad-CAM)
Highlights the region where the model focuses while making predictions.

Improves transparency and trust in AI-based medical systems.

---

### 📊 3. Model Evaluation Dashboard
Includes:
- Confusion Matrix
- Performance visualization

---

### 📄 4. Auto-Generated PDF Medical Report
Generates downloadable report including:
- Diagnosis
- Confidence
- Risk Level
- Symptoms
- Treatment Suggestions
- Timestamp

---

### 🎨 5. Professional Web Interface
Built with:
- Flask backend
- Responsive UI
- Multi-page navigation (Home, About, Team, Confusion Matrix)

---

## 🏗️ Project Structure

brain_tumor_detection/
│
├── dataset/
│ ├── train/
│ ├── test/
│
├── model/
│ └── brain_tumor_model.h5
│
├── static/
│ ├── uploads/
│ ├── gradcam.jpg
│ ├── report.pdf
│ ├── confusion_matrix.png
│
├── templates/
│ ├── index.html
│ ├── result.html
│ ├── about.html
│ ├── team.html
│ ├── confusion_matrix.html
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md


---

## 🛠️ Technologies Used

- **TensorFlow / Keras** (Deep Learning)
- **MobileNetV2** (Transfer Learning)
- **Flask** (Backend Web Framework)
- **OpenCV** (Image Processing)
- **NumPy**
- **Matplotlib & Seaborn** (Visualization)
- **Scikit-learn** (Confusion Matrix)
- **ReportLab** (PDF Generation)
- **HTML / CSS** (Frontend)

---

## 🧪 Model Details

- Architecture: MobileNetV2 (Pretrained on ImageNet)
- Fine-tuned for 4-class tumor classification
- Image Size: 224x224
- Optimizer: Adam
- Loss: Categorical Crossentropy

---

## 📈 Explainable AI Implementation

Grad-CAM is implemented to:

- Extract convolutional feature maps
- Compute gradient-based importance weights
- Generate heatmap overlay
- Highlight tumor regions visually

This improves model interpretability in medical diagnosis scenarios.

---

## ▶️ How To Run Locally

### 1️⃣ Clone Repository

git clone https://github.com/YOUR_USERNAME/AI-Brain-Tumor-Detection.git
cd AI-Brain-Tumor-Detection


### 2️⃣ Create Virtual Environment

python -m venv venv
venv\Scripts\activate


### 3️⃣ Install Dependencies

pip install -r requirements.txt


### 4️⃣ Run Application

python app.py


Visit:

http://127.0.0.1:5000


---

## ⚠️ Disclaimer

This AI system is developed for educational and research purposes only.

It is not a substitute for professional medical diagnosis.

Always consult a qualified healthcare provider.

---

## 👨‍💻 Developed By

MCA (Artificial Intelligence & Data Science) Students

Mini Project – 2026

---

## ⭐ Future Improvements

- Model deployment on cloud (Render / AWS)
- Doctor authentication dashboard
- Model performance graphs (accuracy/loss curves)
- Support for additional MRI formats
- Database integration for patient records

---

## 📌 Project Highlights

✔ Transfer Learning  
✔ Multi-Class Classification  
✔ Explainable AI (Grad-CAM)  
✔ Automated Medical Reporting  
✔ Research-Level Evaluation  

---

If you like this project, consider giving it a ⭐ on GitHub.
