"""
Model downloader utility for Streamlit deployment.
Downloads the models folder from Google Drive if it doesn't exist locally.
"""

import os
import streamlit as st
from pathlib import Path
import gdown

def get_file_id_from_url(drive_url):
    """
    Extract folder ID from Google Drive folder URL.
    
    Args:
        drive_url (str): Google Drive folder sharing URL
    
    Returns:
        str: Folder ID for direct download
    """
    if '/folders/' in drive_url:
        return drive_url.split('/folders/')[1].split('?')[0]
    elif 'id=' in drive_url:
        return drive_url.split('id=')[1].split('&')[0]
    else:
        # Assume it's already a folder ID
        return drive_url

def download_folder_from_google_drive(folder_id, destination_folder="models"):
    """
    Download a folder from Google Drive using the folder ID.
    
    Args:
        folder_id (str): Google Drive folder ID
        destination_folder (str): Local folder name where files should be saved
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        # Use gdown to download the entire folder
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=destination_folder, quiet=False, use_cookies=False)
        return True
    except Exception as e:
        st.error(f"Error downloading folder: {e}")
        st.info("💡 Tip: Make sure the folder is shared publicly and the URL is correct.")
        return False

def download_models_if_needed(drive_url=None, force_download=False):
    """
    Check if models folder exists, and download from Google Drive if not.
    Only supports folder downloads.
    
    Args:
        drive_url (str): Google Drive URL for the models folder
        force_download (bool): Whether to force download even if folder exists
    
    Returns:
        bool: True if models are available, False otherwise
    """
    models_dir = Path("models")
    
    # Check if models folder already exists and has content
    if models_dir.exists() and any(models_dir.iterdir()) and not force_download:
        st.success("✅ Models folder found locally!")
        return True
    
    if not drive_url:
        st.error("❌ Models folder not found and no Google Drive URL provided!")
        st.info("Please provide the Google Drive folder URL in the configuration.")
        return False
    
    try:
        # Handle folder download
        with st.spinner("🔄 Downloading models folder from Google Drive... This may take a few minutes."):
            st.info("📁 Downloading folder from Google Drive...")
            folder_id = get_file_id_from_url(drive_url)
            
            # Remove existing models folder if it exists
            if models_dir.exists():
                import shutil
                shutil.rmtree(models_dir)
            
            success = download_folder_from_google_drive(folder_id, "models")
            
            if success:
                st.success("✅ Models folder downloaded successfully!")
                return True
            else:
                st.error("❌ Failed to download folder. Please check the URL and permissions.")
                return False
                
    except Exception as e:
        st.error(f"❌ Failed to download models: {e}")
        return False

def check_models_integrity():
    """
    Check if all required model files are present.
    
    Returns:
        tuple: (bool, list) - (all_files_present, missing_files)
    """
    required_files = [
        "dl_model_config.pkl",
        "faiss_index.bin",
        "train_metadata.pkl",
        "dl_label_encoder.pkl",
        "all_nb_models_config.pkl",
        "gaussian_bow_model.pkl",
        "gaussian_tfidf_model.pkl",
        "multinomial_bow_model.pkl",
        "multinomial_tfidf_model.pkl",
        "label_encoder.pkl",
        "dictionary.pkl",
        "vectorizer.pkl"
    ]
    
    models_dir = Path("models")
    missing_files = []
    
    for file in required_files:
        if not (models_dir / file).exists():
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def setup_models(drive_url=None):
    """
    Main function to set up models for the Streamlit app.
    
    Args:
        drive_url (str): Google Drive URL for the models.zip file
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    st.info("🔍 Checking for models...")
    
    # Check if models exist and are complete
    models_complete, missing_files = check_models_integrity()
    
    if models_complete:
        st.success("✅ All model files are present!")
        return True
    
    if missing_files:
        st.warning(f"⚠️ Missing model files: {', '.join(missing_files)}")
    
    # Download models if needed
    return download_models_if_needed(drive_url)

if __name__ == "__main__":
    # For testing purposes
    # Replace with your actual Google Drive URL
    test_url = "https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing"
    setup_models(test_url)
