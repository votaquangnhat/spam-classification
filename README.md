# SMS Spam Classification

A comprehensive machine learning project that implements two different approaches for SMS spam classification: **Naive Bayes** and **Deep Learning with k-Nearest Neighbors**, complete with a Streamlit web application for interactive demo.

## 🎯 Project Overview

This project classifies SMS messages as either "spam" or "ham" (legitimate) using two distinct machine learning approaches:

1. **Traditional ML Approach**: Naive Bayes with text preprocessing and feature extraction
2. **Deep Learning Approach**: Transformer embeddings with k-NN classification using FAISS

## 📊 Dataset

- **Source**: SMS Spam Collection dataset (`2cls_spam_text_cls.csv`)
- **Format**: CSV with two columns - `Category` (ham/spam) and `Message`
- **Size**: 5,575 SMS messages
- **Distribution**: Imbalanced dataset with more legitimate messages than spam

## 🚀 Features

### Multiple Model Implementations
- **4 Naive Bayes variants**: Gaussian/Multinomial × BoW/TF-IDF
- **Deep Learning model**: Multilingual E5-base transformer with k-NN (k=1-10)
- **Interactive web interface** for real-time classification
- **Model comparison** functionality

### Advanced Text Processing
- Lowercasing and punctuation removal
- Tokenization with NLTK
- Stopword removal
- Stemming with Porter Stemmer
- Feature extraction (BoW/TF-IDF)

### Performance Analysis
- Comprehensive model evaluation
- Error analysis with detailed neighbor information
- Best performing model identification
- Cross-validation results

## 🏗️ Project Structure

```
spam-classification/
├── 2cls_spam_text_cls.csv          # Dataset
├── app.py                          # Streamlit web application
├── NB approach.ipynb               # Naive Bayes implementation
├── DL approach.ipynb               # Deep Learning implementation
├── error_analysis.json             # Detailed error analysis
├── requirements.txt                # Dependencies
├── models/                         # Trained models and artifacts (download separately)
│   ├── *_nb_models_config.pkl     # NB model configurations
│   ├── gaussian_bow_model.pkl     # Gaussian NB with BoW
│   ├── gaussian_tfidf_model.pkl   # Gaussian NB with TF-IDF
│   ├── multinomial_bow_model.pkl  # Multinomial NB with BoW
│   ├── multinomial_tfidf_model.pkl# Multinomial NB with TF-IDF
│   ├── dl_model_config.pkl        # DL model configuration
│   ├── faiss_index.bin           # FAISS similarity index
│   ├── train_metadata.pkl        # Training data metadata
│   ├── *_label_encoder.pkl       # Label encoders
│   ├── dictionary.pkl            # BoW dictionary
│   └── vectorizer.pkl            # TF-IDF vectorizer
└── README.md
```

## 📈 Model Performance

### Deep Learning Model (E5-base + k-NN)
- **Best Performance**: k=3 with **99.28%** accuracy
- **Model**: `intfloat/multilingual-e5-base`
- **Architecture**: Transformer embeddings → FAISS index → k-NN classification
- **Range**: 98.56% - 99.28% accuracy across different k values

### Naive Bayes Models
Multiple variants with different feature extraction methods:
- **Gaussian NB + BoW/TF-IDF**
- **Multinomial NB + BoW/TF-IDF**

(Specific performance metrics available in model configuration files)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd spam-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download pre-trained models**

Download from: [here](https://drive.google.com/drive/folders/15WylZWKlEyjrgSfjTgqls8BKj2MS9qQb?usp=sharing)

Extract to the project root directory

4. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```

## 🎮 Usage

### Web Application

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

**Features:**
- Real-time spam classification
- Model selection (NB variants or Deep Learning)
- Parameter tuning (k-value for k-NN)
- Model comparison mode
- Confidence scores and explanations

### Jupyter Notebooks

**Naive Bayes Approach** (`NB approach.ipynb`):
- Text preprocessing pipeline
- Feature extraction (BoW/TF-IDF)
- Model training and evaluation
- Performance comparison

**Deep Learning Approach** (`DL approach.ipynb`):
- Transformer embedding generation
- FAISS index construction
- k-NN classification
- Error analysis and optimization