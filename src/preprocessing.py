# src/preprocessing.py
"""
Module tien xu ly du lieu:
- Load du lieu tu Khan Academy va Coursera
- Xu ly missing values, duplicates
- Chuan hoa van ban
- Tao combined_text va TF-IDF
- Sinh nhan do kho tu metadata
"""

import pandas as pd
import numpy as np
import re
import os
import string
import joblib
import warnings
warnings.filterwarnings('ignore')

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ---- Download NLTK data ----
def download_nltk():
    for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk()

# ---- Constants ----
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DIR  = os.path.join(DATA_DIR, 'raw')
PROC_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

for d in [RAW_DIR, PROC_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- Text Normalization ----
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))

def normalize_text(text: str) -> str:
    """Lowercase, remove punct, stopwords, lemmatize."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # lowercase
    text = text.lower()
    # remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # remove punctuation & digits
    text = re.sub(r'[^a-z\s]', ' ', text)
    # tokenize
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    # remove stopwords & lemmatize
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stop_words and len(t) > 2
    ]
    return ' '.join(tokens)

# ---- Difficulty Labeling ----
def difficulty_from_gunning_fog(fog_score) -> str:
    """
    Gunning Fog Index -> difficulty label
    < 8  : Easy (grade school)
    8-12 : Medium (high school)
    > 12 : Hard (college+)
    """
    try:
        fog = float(fog_score)
        if fog < 8:
            return 'Easy'
        elif fog <= 12:
            return 'Medium'
        else:
            return 'Hard'
    except Exception:
        return 'Medium'

def difficulty_from_coursera_category(category: str, content: str = "") -> str:
    """Suy ra do kho tu category/content Coursera."""
    easy_kw = ['beginner', 'introduction', 'intro', 'basic', 'fundamental', 'getting started',
                '101', 'for beginners', 'learn', 'first', 'elementary', 'simple', 'overview']
    hard_kw = ['advanced', 'expert', 'professional', 'deep', 'master', 'phd', 'graduate',
                'specialization', 'architecture', 'research', 'optimization', 'theory']
    
    combined = (str(category) + ' ' + str(content)).lower()
    
    if any(k in combined for k in hard_kw):
        return 'Hard'
    elif any(k in combined for k in easy_kw):
        return 'Easy'
    else:
        return 'Medium'

# ---- Load Khan Academy ----
def load_khan_academy(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """Load va tien xu ly Khan Academy dataset."""
    print(f"[Khan Academy] Loading {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Raw shape: {df.shape}")
    
    # Chon cot quan trong
    cols = ['title', 'description', 'channel_title', 'view_count', 'like_count',
            'duration', 'desc_gunning_fog', 'desc_flesch_reading_ease',
            'desc_flesch_kincaid_grade', 'title_gunning_fog',
            'title_sentiment_polarity', 'desc_sentiment_polarity', 'videoId']
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    
    # Xu ly missing
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['channel_title'] = df['channel_title'].fillna('Unknown')
    df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)
    df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0)
    df['desc_gunning_fog'] = pd.to_numeric(df.get('desc_gunning_fog', pd.Series()), errors='coerce')
    
    # Loai bo trung lap
    before = len(df)
    df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
    print(f"  After dedup: {len(df)} (removed {before - len(df)})")
    
    # Sample neu can
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    # Tao difficulty label
    df['difficulty'] = df['desc_gunning_fog'].apply(difficulty_from_gunning_fog)
    
    # Combined text (raw)
    df['raw_text'] = df['title'] + ' ' + df['description']
    
    # Normalized text
    print("  Normalizing text...")
    df['clean_text'] = df['raw_text'].apply(normalize_text)
    
    # Source
    df['source'] = 'khan'
    df['id'] = 'khan_' + df.index.astype(str)
    df['category'] = df['channel_title'].apply(lambda x: str(x).split('|')[0].strip())
    
    return df[['id', 'title', 'raw_text', 'clean_text', 'category',
               'difficulty', 'source', 'view_count', 'like_count']].copy()

# ---- Load Coursera ----
def load_coursera(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """Load va tien xu ly Coursera dataset."""
    print(f"[Coursera] Loading {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Raw shape: {df.shape}")
    
    # Rename cols
    rename_map = {}
    if 'name' in df.columns: rename_map['name'] = 'title'
    if 'content' in df.columns: rename_map['content'] = 'description'
    df = df.rename(columns=rename_map)
    
    # Fallback
    if 'title' not in df.columns:
        df['title'] = df.iloc[:, 0]
    if 'description' not in df.columns:
        df['description'] = ''
    
    for col in ['title', 'description', 'category', 'skills', 'what_you_learn']:
        if col not in df.columns:
            df[col] = ''
    
    # Xu ly missing
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['category'] = df['category'].fillna('General')
    df['skills'] = df['skills'].fillna('')
    df['what_you_learn'] = df['what_you_learn'].fillna('')
    
    # Loai bo trung lap
    before = len(df)
    df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
    print(f"  After dedup: {len(df)} (removed {before - len(df)})")
    
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    # Combined text
    df['raw_text'] = (df['title'] + ' ' + df['description'] + ' ' + 
                      df['what_you_learn'] + ' ' + df['skills'])
    
    # Difficulty label
    df['difficulty'] = df.apply(
        lambda r: difficulty_from_coursera_category(r['category'], r['raw_text']), axis=1
    )
    
    # Normalized text
    print("  Normalizing text...")
    df['clean_text'] = df['raw_text'].apply(normalize_text)
    
    df['source'] = 'coursera'
    df['id'] = 'coursera_' + df.index.astype(str)
    df['view_count'] = 0
    df['like_count'] = 0
    
    return df[['id', 'title', 'raw_text', 'clean_text', 'category',
               'difficulty', 'source', 'view_count', 'like_count']].copy()

# ---- Merge & TF-IDF ----
def build_unified_dataset(khan_df: pd.DataFrame, coursera_df: pd.DataFrame) -> pd.DataFrame:
    """Gop 2 dataset."""
    unified = pd.concat([khan_df, coursera_df], ignore_index=True)
    unified = unified[unified['clean_text'].str.len() > 10].reset_index(drop=True)
    print(f"[Unified] Total records: {len(unified)}")
    print(f"  Difficulty distribution:\n{unified['difficulty'].value_counts()}")
    return unified

def build_tfidf(df: pd.DataFrame, max_features: int = 5000, save_path: str = None):
    """Sinh TF-IDF matrix."""
    print(f"[TF-IDF] Fitting on {len(df)} documents (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    print(f"  TF-IDF shape: {tfidf_matrix.shape}")
    
    if save_path:
        joblib.dump(vectorizer, save_path)
        print(f"  Vectorizer saved to {save_path}")
    
    return tfidf_matrix, vectorizer

def encode_labels(df: pd.DataFrame):
    """Label encode difficulty."""
    le = LabelEncoder()
    label_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}
    df['difficulty_encoded'] = df['difficulty'].map(label_map).fillna(1).astype(int)
    return df, label_map

# ---- Full Pipeline ----
def run_preprocessing(
    khan_file: str,
    coursera_file: str,
    khan_sample: int = 20000,
    coursera_sample: int = 15000,
    tfidf_features: int = 5000
):
    """Chay toan bo pipeline tien xu ly."""
    khan_df = load_khan_academy(khan_file, sample_size=khan_sample)
    coursera_df = load_coursera(coursera_file, sample_size=coursera_sample)
    unified_df = build_unified_dataset(khan_df, coursera_df)
    unified_df, label_map = encode_labels(unified_df)
    
    # Save processed data
    proc_path = os.path.join(PROC_DIR, 'unified_data.csv')
    unified_df.to_csv(proc_path, index=False)
    print(f"[Saved] Processed data -> {proc_path}")
    
    # Build TF-IDF
    vec_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    tfidf_matrix, vectorizer = build_tfidf(unified_df, max_features=tfidf_features, save_path=vec_path)
    
    # Save TF-IDF matrix (sparse)
    from scipy.sparse import save_npz
    mat_path = os.path.join(PROC_DIR, 'tfidf_matrix.npz')
    save_npz(mat_path, tfidf_matrix)
    print(f"[Saved] TF-IDF matrix -> {mat_path}")
    
    return unified_df, tfidf_matrix, vectorizer, label_map


if __name__ == '__main__':
    import sys
    base = os.path.dirname(os.path.dirname(__file__))
    khan_file = os.path.join(base, 'youtube_khan_academy.csv')
    coursera_file = os.path.join(base, 'courses_en.csv')
    run_preprocessing(khan_file, coursera_file)
