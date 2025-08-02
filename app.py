import streamlit as st
import numpy as np
import pickle
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import string
import nltk
import os

# Import our model downloader
from download_models import setup_models
from config import MODELS_DRIVE_URL

# --- NLTK Data Download and Preprocessing Functions (from NB notebook) ---
# These are the same preprocessing steps as in the NB notebook.
try:
    # Attempt to download NLTK data.
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    st.warning(f"Could not download NLTK data: {e}. Preprocessing functions may not work as intended.")
    pass

def lowercase(text):
    """Converts a string to lowercase."""
    return text.lower()

def punctuation_removal(text):
    """Removes punctuation from a string."""
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

def tokenize(text):
    """Tokenizes a string into a list of words."""
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    """Removes common stopwords from a list of tokens."""
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return [token for token in tokens if token not in stop_words]

def stemming(tokens):
    """Applies stemming to a list of tokens."""
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    """Combines all preprocessing steps for a given text."""
    if not isinstance(text, str):
        return ""
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return " ".join(tokens)

# --- Model Loading and Prediction (using st.cache_resource) ---
@st.cache_resource
def load_deep_learning_models():
    """
    Loads all saved model components for inference, based on the provided notebook code.
    This function is cached by Streamlit to run only once.
    """
    try:
        st.write("Loading Deep Learning model components from models folder...")
        
        models_dir = "models"
        
        # Load configuration
        dl_model_config_path = os.path.join(models_dir, 'dl_model_config.pkl')
        with open(dl_model_config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Load the embedding model and tokenizer from HuggingFace
        MODEL_NAME = config["model_name"]
        dl_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        dl_model = AutoModel.from_pretrained(MODEL_NAME)
        
        # Set device and prepare model
        dl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dl_model = dl_model.to(dl_device)
        dl_model.eval()
        
        # Load FAISS index
        faiss_index_path = os.path.join(models_dir, "faiss_index.bin")
        faiss_index = faiss.read_index(faiss_index_path)
        
        # Load metadata
        train_metadata_path = os.path.join(models_dir, 'train_metadata.pkl')
        with open(train_metadata_path, 'rb') as f:
            dl_train_metadata = pickle.load(f)

        # Assuming dl_train_metadata is a list of dictionaries as per user's last message
        dl_train_messages = [item['message'] for item in dl_train_metadata]
        dl_train_labels = [item['label_encoded'] for item in dl_train_metadata]
        
        # Load label encoder (DL-specific filename)
        dl_label_encoder_path = os.path.join(models_dir, 'dl_label_encoder.pkl')
        with open(dl_label_encoder_path, 'rb') as f:
            dl_le = pickle.load(f)
        
        st.success(f"Loaded model: {MODEL_NAME}")
        st.write(f"Training samples: {len(dl_train_messages)}")
        st.write(f"Device: {dl_device}")
        if 'accuracy_results' in config:
            best_k = max(config['accuracy_results'], key=config['accuracy_results'].get)
            best_accuracy = config['accuracy_results'][best_k]
            st.write(f"Best performance: k={best_k} with accuracy={best_accuracy:.4f}")
        
        return dl_tokenizer, dl_model, dl_device, faiss_index, dl_train_messages, dl_train_labels, dl_le

    except FileNotFoundError as e:
        st.error(f"Could not find model file: {e}. Please ensure the correct .pkl, .bin files are in the models directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading DL models: {e}")
        st.stop()

@st.cache_resource
def load_naive_bayes_model(nb_type, feature_type):
    """Loads the specified Naive Bayes model and feature extractor."""
    try:
        st.write(f"Loading {nb_type} Naive Bayes model with {feature_type} features from models folder...")
        
        models_dir = "models"

        if feature_type == "TF-IDF":
            feature_type = "TFIDF"
        
        # Load the specific model based on user choice
        model_filename = f"{nb_type.lower()}_{feature_type.lower()}_model.pkl"
        model_path = os.path.join(models_dir, model_filename)
        nb_model = pickle.load(open(model_path, 'rb'))
        
        # Load appropriate feature extractor
        if feature_type == "BoW":
            dictionary_path = os.path.join(models_dir, 'dictionary.pkl')
            feature_extractor = pickle.load(open(dictionary_path, 'rb'))
        else:  # TF-IDF
            vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
            feature_extractor = pickle.load(open(vectorizer_path, 'rb'))
        
        # Load label encoder
        label_encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
        le = pickle.load(open(label_encoder_path, 'rb'))
        
        return nb_model, feature_extractor, le
    except FileNotFoundError as e:
        st.error(f"Could not find model file: {e}. Please ensure the correct .pkl files are in the models directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading Naive Bayes models: {e}")
        st.stop()

@st.cache_resource
def load_nb_model_info():
    """Loads all Naive Bayes model configurations for display."""
    try:
        models_dir = "models"
        all_config_path = os.path.join(models_dir, 'all_nb_models_config.pkl')
        with open(all_config_path, 'rb') as f:
            all_config = pickle.load(f)
        return all_config
    except FileNotFoundError:
        return None
    except Exception:
        return None

def create_features(tokens, dictionary):
    """Create BoW features from tokens and dictionary"""
    features = np.zeros(len(dictionary))
    for token in tokens.split():
        if token in dictionary:
            features[dictionary.index(token)] += 1
    return features

def predict_nb(text, nb_model, feature_extractor, le, feature_type):
    """Classifies a text using the Naive Bayes model."""
    preprocessed_text = preprocess_text(text)
    
    if feature_type == "TF-IDF":
        features = feature_extractor.transform([preprocessed_text]).toarray()
    else: # BoW
        features = create_features(preprocessed_text, feature_extractor)
        features = np.array(features).reshape(1, -1)
    
    prediction_score = nb_model.predict_proba(features)[0]
    prediction_cls_idx = np.argmax(prediction_score)
    prediction_cls = le.inverse_transform([prediction_cls_idx])[0]
    
    return prediction_cls, prediction_score[prediction_cls_idx], preprocessed_text

def predict_dl(text, dl_tokenizer, dl_model, dl_device, faiss_index, dl_train_messages, dl_train_labels, dl_le, k_value):
    """Classifies a text using the Deep Learning model and k-NN."""
    # Create the embedding for the input text
    encoded_input = dl_tokenizer([text], padding=True, truncation=True, return_tensors='pt').to(dl_device)
    with torch.no_grad():
        model_output = dl_model(**encoded_input)
        sentence_embedding = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
    
    # Search the FAISS index for the k nearest neighbors
    distances, indices = faiss_index.search(sentence_embedding, k_value)
    
    # Get the labels of the neighbors
    neighbor_labels_encoded = np.array(dl_train_labels)[indices[0]]
    
    # Count the occurrences of each label to determine the final prediction
    unique_labels_encoded, counts = np.unique(neighbor_labels_encoded, return_counts=True)
    predicted_label_encoded = unique_labels_encoded[np.argmax(counts)]
    
    # Decode the predicted label
    predicted_label = dl_le.inverse_transform([predicted_label_encoded])[0]
    
    # Calculate a simple confidence score
    confidence = np.max(counts) / k_value
    
    # Prepare the neighbors for display
    top_neighbors = []
    for i in range(k_value):
        top_neighbors.append({
            "message": dl_train_messages[indices[0][i]],
            "label": dl_le.inverse_transform([dl_train_labels[indices[0][i]]])[0],
            "score": distances[0][i]  # Using distance as a score
        })
        
    return predicted_label, confidence, top_neighbors

# --- Streamlit App UI ---
st.set_page_config(page_title="Spam Classifier Demo", layout="wide")

st.title("Spam Text Classifier Demo")
st.markdown(
    """
    This is a demonstration of a spam classification application based on two different machine learning approaches:
    a Deep Learning model with a k-Nearest Neighbors (k-NN) classifier and a Naive Bayes model.
    
    ***Note: This application loads models from the models directory.***
    """
)

# Setup models (download if needed)
if not setup_models(MODELS_DRIVE_URL):
    st.error("❌ Failed to set up models. Please check your internet connection and try again.")
    st.stop()

# Load DL models once at the start
dl_tokenizer, dl_model, dl_device, faiss_index, dl_train_messages, dl_train_labels, dl_le = load_deep_learning_models()

# Load NB model info for display (optional)
nb_all_config = load_nb_model_info()

# Display model information
col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 Deep Learning Model")
    st.info("Uses transformer embeddings with k-NN classification")

with col2:
    st.subheader("📊 Naive Bayes Models")
    if nb_all_config:
        best_model_info = nb_all_config['best_model_info']
        st.info(f"Best model: {best_model_info['name']} ({best_model_info['val_acc']:.4f} accuracy)")
    else:
        st.info("Multiple NB models available for selection")

# User input for the message
user_input = st.text_area(
    "Enter a message to classify:",
    "FREE!! Click here to win $1000 NOW! Limited time offer!",
    height=150
)

# Model selection
model_choice = st.selectbox(
    "Choose a classification model:",
    ("Deep Learning (E5-base + k-NN)", "Naive Bayes", "Compare Both Models")
)

# Conditional UI based on model choice
if model_choice == "Deep Learning (E5-base + k-NN)":
    k_value = st.slider(
        "Select the number of neighbors (k) for k-NN:",
        min_value=1, max_value=10, value=3, step=1
    )
    st.info(f"The Deep Learning model uses a transformer to create sentence embeddings and then finds the {k_value} nearest neighbors in a pre-computed FAISS index to classify the message.")
    
elif model_choice == "Naive Bayes":
    nb_type = st.radio(
        "Select Naive Bayes Type:",
        ("Gaussian", "Multinomial")
    )
    feature_type = st.radio(
        "Select Feature Extraction Method:",
        ("BoW", "TF-IDF")
    )
    st.info(f"Using {nb_type} Naive Bayes model with {feature_type} features.")

elif model_choice == "Compare Both Models":
    k_value = st.slider(
        "Select the number of neighbors (k) for k-NN:",
        min_value=1, max_value=10, value=3, step=1
    )
    nb_type = st.radio(
        "Select Naive Bayes Type:",
        ("Gaussian", "Multinomial")
    )
    feature_type = st.radio(
        "Select Feature Extraction Method:",
        ("BoW", "TF-IDF")
    )
    st.info("Both models will be used to classify the message for comparison.")

# Classification button
if st.button("Classify Message"):
    if not user_input.strip():
        st.error("Please enter a message to classify.")
    else:
        with st.spinner("Classifying..."):
            if model_choice == "Deep Learning (E5-base + k-NN)":
                predicted_label, confidence, top_neighbors = predict_dl(
                    user_input, dl_tokenizer, dl_model, dl_device, faiss_index, dl_train_messages, dl_train_labels, dl_le, k_value
                )
                
                st.subheader("Classification Result (Deep Learning)")
                st.metric("Predicted Class", predicted_label.upper())
                st.metric("Confidence Score", f"{confidence*100:.2f}%")
                
                st.subheader(f"Top {k_value} Nearest Neighbors")
                for i, neighbor in enumerate(top_neighbors):
                    st.markdown(f"""
                    **Neighbor {i+1}**
                    - **Label**: `{neighbor['label'].upper()}`
                    - **Message**: `{neighbor['message']}`
                    - **Similarity Score**: `{neighbor['score']:.4f}`
                    """)
            
            elif model_choice == "Naive Bayes":
                # Load the specific NB model based on user's choice
                nb_model, nb_feature_extractor, nb_le = load_naive_bayes_model(nb_type, feature_type)
                
                predicted_label, confidence, preprocessed_text = predict_nb(
                    user_input, nb_model, nb_feature_extractor, nb_le, feature_type
                )
                
                st.subheader(f"Classification Result ({nb_type} NB + {feature_type})")
                st.metric("Predicted Class", predicted_label.upper())
                st.metric("Confidence Score", f"{confidence*100:.2f}%")
                st.markdown(f"**Preprocessed text:** `{preprocessed_text}`")
                
            elif model_choice == "Compare Both Models":
                # Deep Learning prediction
                dl_predicted_label, dl_confidence, top_neighbors = predict_dl(
                    user_input, dl_tokenizer, dl_model, dl_device, faiss_index, dl_train_messages, dl_train_labels, dl_le, k_value
                )
                
                # Naive Bayes prediction
                nb_model, nb_feature_extractor, nb_le = load_naive_bayes_model(nb_type, feature_type)
                nb_predicted_label, nb_confidence, preprocessed_text = predict_nb(
                    user_input, nb_model, nb_feature_extractor, nb_le, feature_type
                )
                
                # Display comparison
                st.subheader("Model Comparison Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🤖 Deep Learning Model")
                    st.metric("Predicted Class", dl_predicted_label.upper())
                    st.metric("Confidence Score", f"{dl_confidence*100:.2f}%")
                
                with col2:
                    st.markdown(f"### 📊 {nb_type} NB + {feature_type}")
                    st.metric("Predicted Class", nb_predicted_label.upper())
                    st.metric("Confidence Score", f"{nb_confidence*100:.2f}%")
                
                # Agreement check
                if dl_predicted_label.lower() == nb_predicted_label.lower():
                    st.success("✅ Both models agree on the classification!")
                else:
                    st.warning("⚠️ Models disagree on the classification.")
                
                # Show additional details
                with st.expander("View Detailed Results"):
                    st.markdown("**Preprocessed text (NB):** " + preprocessed_text)
                    st.markdown(f"**Top {k_value} neighbors (DL):**")
                    for i, neighbor in enumerate(top_neighbors[:3]):  # Show top 3
                        st.markdown(f"- {neighbor['label']}: {neighbor['message'][:100]}...")

# Show model performance comparison
with st.expander("📈 Model Performance Information"):
    if nb_all_config:
        st.markdown("### Naive Bayes Models Ranking")
        for rank, model_info in enumerate(nb_all_config['ranked_models'], 1):
            emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📉"
            st.markdown(f"{rank}. {emoji} {model_info['name']} - Val: {model_info['val_acc']:.4f}, Test: {model_info['test_acc']:.4f}")
    else:
        st.markdown("### Naive Bayes Models")
        st.markdown("4 different combinations available: Gaussian/Multinomial × BoW/TF-IDF")
    
    st.markdown("### Deep Learning Model")
    st.markdown("Uses multilingual E5-base transformer with FAISS similarity search for k-NN classification.")
