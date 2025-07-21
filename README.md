# 💬 Sentiment Analysis of Social Media Data (Twitter & Reddit)

This project is an end-to-end **Natural Language Processing (NLP)** task focused on sentiment analysis. The goal is to build and evaluate machine learning models capable of classifying text from **Twitter** and **Reddit** into three sentiment categories: **Positive**, **Negative**, and **Neutral**.

The entire workflow is documented in the `Sentiment-Analysis.ipynb` Jupyter Notebook, covering everything from data loading and preprocessing to model training, evaluation, and prediction.

---

## 🚀 Project Overview

This project follows a standard machine learning pipeline:

- **Data Loading**:
  - `Twitter_Data.csv`
  - `Reddit_Data.csv`

- **Data Cleaning & Preprocessing**:
  - Handle encoding issues and special characters.
  - Remove URLs, mentions, hashtags.
  - Convert text to lowercase.
  - Tokenization, stopword removal, and lemmatization using **NLTK**.

- **Exploratory Data Analysis (EDA)**:
  - Visualize sentiment distribution and word counts using **Matplotlib** and **Seaborn**.

- **Model Training**:
  - Logistic Regression  
  - Random Forest  
  - XGBoost

- **Model Evaluation**:
  - Evaluate using **Accuracy**, **Precision**, **Recall**, and **F1-score**
  - Visualize results with **Confusion Matrices**

---

## 🛠️ Technologies & Libraries Used

- Python 3.x  
- Jupyter Notebook  
- Pandas, NumPy  
- NLTK (Natural Language Toolkit)  
- Scikit-learn  
- Matplotlib & Seaborn  

---

## 📂 Project Structure

```
📦 Sentiment Analysis Project
├── Reddit_Data.csv
├── Twitter_Data.csv
├── Sentiment Analysis.ipynb      # Jupyter notebook with all code
└── README.md
```

---

## 🏁 How to Run the Project

1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

2. **Install Required Packages:**
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn jupyterlab
```

3. **Download Required NLTK Data:**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

4. **Launch the Notebook:**
```bash
jupyter notebook Sentiment Analysis.ipynb
```

---

## 📈 Results

Models were trained on a combined dataset of ~200,000 text samples.

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | **84.0%** |
| Random Forest       | 81.0%    |
| XGBoost             | 79.0%    |

✅ Logistic Regression was the best performer and saved for future inference. It shows reliable accuracy, especially for **positive** and **neutral** sentiment.

---

## 🔮 Future Improvements

- 🔧 **Hyperparameter Tuning** using GridSearchCV  
- 🧠 Explore **Word Embeddings** (e.g., Word2Vec, GloVe, or BERT)  
- 🤖 Try **Deep Learning models** such as LSTMs or Transformers  

---

## 📜 License

This project is open-source and available under the **MIT License**.
