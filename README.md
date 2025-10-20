# 📧 Streamlit Spam Detector

A simple **Machine Learning-powered web application** built with **Streamlit** that detects whether an email or SMS message is **Spam** or **Not Spam**.  
The model uses **TF-IDF vectorization** and a **Multinomial Naive Bayes** classifier.

---

## 🚀 Live Demo
👉 *(Add your Streamlit Cloud or local URL here once deployed)*  
Example:  
`https://streamlit-spam-detector.streamlit.app`

---

## 🧠 Project Overview

The **Streamlit Spam Detector** is a text classification project trained on the **SMS Spam Collection Dataset**.  
It demonstrates how NLP techniques can be applied to classify messages as **Spam** or **Not Spam (Ham)** in real-time.

### Key Features
- **Text Preprocessing**: Tokenization, stopword removal, and stemming.
- **Feature Extraction**: Uses **TF-IDF Vectorizer** to convert text into numerical features.
- **Model Training**: Implements a **Multinomial Naive Bayes classifier**.
- **Real-Time Prediction**: Users can enter messages via a **Streamlit UI** and get instant predictions.

### Dataset
- **SMS Spam Collection Dataset (UCI)**  
  Contains 5,572 messages labeled as `ham` (not spam) or `spam`.


---

## 🛠️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend / UI** | Streamlit |
| **Machine Learning** | Scikit-learn |
| **Text Processing** | NLTK |
| **Language** | Python |
| **Model** | Multinomial Naive Bayes |
| **Vectorizer** | TF-IDF |

---

## 📂 Project Structure

streamlit-spam-detector/
│
- `app.py` – Streamlit web application  
- `spam_detector_model.pkl` – Pre-trained Naive Bayes model  
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer for feature extraction  
- `spam.csv` – Optional dataset  
- `requirements.txt` – Python dependencies  
- `README.md` – Documentation



---

## ⚙️ Installation and Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/streamlit-spam-detector.git
cd streamlit-spam-detector
```
2️⃣ Create a virtual environment

Copy code
python -m venv env
# Activate the environment
env\Scripts\activate       # For Windows
source env/bin/activate    # For Mac/Linux

3️⃣ Install dependencies
```
pip install -r requirements.txt
```
4️⃣ Download required NLTK resources
```
python
Copy code
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```
5️⃣ Run the Streamlit app
```
streamlit run app.py
```

🧩 Sample Messages
✅ Not Spam
```
Hey, are we still meeting for lunch tomorrow?
```

```
Please review the attached report and send your feedback.
```
🚨 Spam

```
Congratulations! You've won a $1000 Walmart gift card. Click here to claim!
perl
```
Copy code
```
URGENT! Your account will be suspended unless you verify immediately.
```
## 📊 Model Performance

| Metric      | Value                     |
|------------|---------------------------|
| Accuracy    | ~97%                      |
| Precision   | ~0.98                     |
| Classifier  | Multinomial Naive Bayes   |
| Features    | 3000 TF-IDF features      |


💾 Dataset
Dataset used:
📁 SMS Spam Collection Dataset (UCI)
Contains 5572 messages labeled as "ham" (not spam) or "spam".

✨ Future Enhancements
🌐 Deploy to Streamlit Cloud or Hugging Face Spaces

📊 Add data visualization for spam/ham statistics

📱 Provide an API endpoint for app integration

🔠 Extend support for multilingual spam detection

👨‍💻 Author
Aniket Dhamecha
💼 Developer & ML Enthusiast

⭐ If you like this project, give it a star on GitHub!


---

