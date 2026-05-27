"""
Script tao cac Jupyter Notebooks cho do an
"""
import json
import os

BASE_DIR = r"c:\Users\tanbt\Desktop\Cuoiky"
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def make_nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.9"}
        },
        "cells": cells
    }

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text, "id": os.urandom(8).hex()}

def code(src, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": src,
        "id": os.urandom(8).hex()
    }

# ============================================================
# NOTEBOOK 1: DATA EXPLORATION
# ============================================================
nb1 = make_nb([
    md("# 01 - Khảo sát Dữ liệu\n## Data Exploration\n\nMục tiêu: Phân tích chi tiết cấu trúc, phân bố và đặc trưng của 2 dataset:\n- **Khan Academy YouTube Dataset** (~52,000 videos)\n- **Coursera Courses Dataset** (~41,000 courses)"),
    
    md("## 1.1 Import Libraries"),
    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Tao thu muc luu anh
os.makedirs('../report/images', exist_ok=True)
os.makedirs('../data/raw', exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
print("Libraries loaded successfully!")"""),
    
    md("## 1.2 Load Khan Academy Dataset"),
    code("""# Load Khan Academy
khan_df = pd.read_csv('../youtube_khan_academy.csv', low_memory=False)
print(f"Shape: {khan_df.shape}")
print(f"\\nColumns ({len(khan_df.columns)}):")
for i, col in enumerate(khan_df.columns):
    print(f"  [{i+1:2d}] {col}: {khan_df[col].dtype}")"""),
    
    code("""# Xem 5 dong dau
khan_df.head()"""),
    
    code("""# Thong ke mo ta
khan_df.describe(include='all').T.head(30)"""),
    
    md("## 1.3 Phân tích Missing Values - Khan Academy"),
    code("""# Gia tri thieu
missing = khan_df.isnull().sum()
missing_pct = (missing / len(khan_df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print(f"So cot co gia tri thieu: {len(missing_df)}")
missing_df"""),
    
    code("""# Bieu do missing values
fig, ax = plt.subplots(figsize=(12, 6))
missing_df.head(20)['Missing %'].plot(kind='barh', ax=ax, color='#4C72B0')
ax.set_title('Missing Values Percentage - Khan Academy', fontsize=14, fontweight='bold')
ax.set_xlabel('Missing %')
plt.tight_layout()
plt.savefig('../report/images/nb01_khan_missing.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    
    md("## 1.4 Load Coursera Dataset"),
    code("""# Load Coursera
coursera_df = pd.read_csv('../courses_en.csv', low_memory=False)
print(f"Shape: {coursera_df.shape}")
print(f"\\nColumns: {coursera_df.columns.tolist()}")
coursera_df.head()"""),
    
    code("""# Missing values Coursera
missing_c = coursera_df.isnull().sum()
missing_c_pct = (missing_c / len(coursera_df) * 100).round(2)
missing_c_df = pd.DataFrame({'Count': missing_c, 'Pct': missing_c_pct})
missing_c_df[missing_c_df['Count'] > 0].sort_values('Count', ascending=False)"""),
    
    md("## 1.5 Phân bố Dữ liệu"),
    code("""# Histogram - View Count
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Khan Academy Video Statistics', fontsize=16, fontweight='bold')

vc = pd.to_numeric(khan_df['view_count'], errors='coerce').dropna()
vc_filtered = vc[vc < vc.quantile(0.95)]
axes[0].hist(vc_filtered, bins=50, color='#4C72B0', alpha=0.8, edgecolor='white')
axes[0].set_title('View Count Distribution')
axes[0].set_xlabel('View Count')
axes[0].set_ylabel('Frequency')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

lc = pd.to_numeric(khan_df['like_count'], errors='coerce').dropna()
lc_filtered = lc[lc < lc.quantile(0.95)]
axes[1].hist(lc_filtered, bins=50, color='#DD8452', alpha=0.8, edgecolor='white')
axes[1].set_title('Like Count Distribution')
axes[1].set_xlabel('Like Count')
plt.tight_layout()
plt.savefig('../report/images/nb01_khan_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"View count stats: mean={vc.mean():.0f}, median={vc.median():.0f}, max={vc.max():.0f}")"""),
    
    code("""# Bar Chart - Coursera Categories
cat_counts = coursera_df['category'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(12, 7))
cat_counts.plot(kind='barh', ax=ax, color=plt.cm.viridis(np.linspace(0, 1, len(cat_counts))))
ax.set_title('Top 15 Coursera Course Categories', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Courses')
for i, v in enumerate(cat_counts.values):
    ax.text(v + 10, i, str(v), va='center')
plt.tight_layout()
plt.savefig('../report/images/nb01_coursera_categories.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    
    code("""# Pie Chart - Coursera Language
if 'language' in coursera_df.columns:
    lang_counts = coursera_df['language'].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.pie(lang_counts.values, labels=lang_counts.index,
           autopct='%1.1f%%', startangle=90,
           colors=plt.cm.Set3(np.linspace(0, 1, len(lang_counts))))
    ax.set_title('Coursera Course Languages', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../report/images/nb01_coursera_languages.png', dpi=150, bbox_inches='tight')
    plt.show()"""),
    
    code("""# Correlation Heatmap - Khan Academy Numeric Features
numeric_cols = ['view_count', 'like_count', 'comment_count',
                'title_sentiment_polarity', 'desc_sentiment_polarity',
                'desc_gunning_fog', 'desc_flesch_reading_ease',
                'desc_flesch_kincaid_grade']
available = [c for c in numeric_cols if c in khan_df.columns]
corr_data = khan_df[available].apply(pd.to_numeric, errors='coerce').corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', ax=ax, square=True, linewidths=0.5,
            annot_kws={'size': 9})
ax.set_title('Correlation Heatmap - Khan Academy', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../report/images/nb01_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    
    code("""# Readability scores distribution
readability_cols = [c for c in khan_df.columns if 'gunning' in c or 'flesch' in c]
if readability_cols:
    fig, axes = plt.subplots(1, min(3, len(readability_cols)), figsize=(15, 5))
    for i, col in enumerate(readability_cols[:3]):
        data = pd.to_numeric(khan_df[col], errors='coerce').dropna()
        data = data[(data > 0) & (data < 30)]
        axes[i].hist(data, bins=40, color=f'C{i}', alpha=0.8, edgecolor='white')
        axes[i].set_title(col.replace('_', ' ').title())
    plt.suptitle('Readability Score Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../report/images/nb01_readability.png', dpi=150, bbox_inches='tight')
    plt.show()"""),
    
    md("## 1.6 Summary"),
    code("""print("="*60)
print("DATASET SUMMARY")
print("="*60)
print(f"Khan Academy:")
print(f"  - Total videos: {len(khan_df):,}")
print(f"  - Columns: {len(khan_df.columns)}")
print(f"  - Missing cells: {khan_df.isnull().sum().sum():,}")
print(f"  - Duplicates (title): {khan_df['title'].duplicated().sum():,}")
print(f"  - Date range: {khan_df['published_at'].min() if 'published_at' in khan_df.columns else 'N/A'} to {khan_df['published_at'].max() if 'published_at' in khan_df.columns else 'N/A'}")
print()
print(f"Coursera:")
print(f"  - Total courses: {len(coursera_df):,}")
print(f"  - Columns: {len(coursera_df.columns)}")
print(f"  - Missing cells: {coursera_df.isnull().sum().sum():,}")
if 'category' in coursera_df.columns:
    print(f"  - Unique categories: {coursera_df['category'].nunique()}")
print()
print("All charts saved to: ../report/images/")"""),
])

with open(os.path.join(NOTEBOOKS_DIR, '01_data_exploration.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb1, f, ensure_ascii=False, indent=2)
print("01_data_exploration.ipynb created")

# ============================================================
# NOTEBOOK 2: PREPROCESSING
# ============================================================
nb2 = make_nb([
    md("# 02 - Tiền xử lý Dữ liệu\n## Data Preprocessing\n\nMục tiêu:\n- Xử lý missing values và duplicates\n- Chuẩn hóa văn bản: lowercase, remove punctuation, stopwords, lemmatization\n- Ghép các trường thành `combined_text`\n- Sinh TF-IDF vector"),

    code("""import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import re
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
for r in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
    nltk.download(r, quiet=True)

print("Setup complete!")"""),

    md("## 2.1 Load Raw Data"),
    code("""khan_df = pd.read_csv('../youtube_khan_academy.csv', low_memory=False, nrows=20000)
coursera_df = pd.read_csv('../courses_en.csv', low_memory=False, nrows=15000)
print(f"Khan: {khan_df.shape}")
print(f"Coursera: {coursera_df.shape}")"""),

    md("## 2.2 Xử lý Missing Values và Duplicates"),
    code("""# Khan Academy
print("=== KHAN ACADEMY ===")
print(f"Before: {len(khan_df)} rows")
khan_df['title'] = khan_df['title'].fillna('')
khan_df['description'] = khan_df['description'].fillna('')
khan_df['channel_title'] = khan_df['channel_title'].fillna('Unknown')
khan_df = khan_df.drop_duplicates(subset=['title']).reset_index(drop=True)
print(f"After dedup: {len(khan_df)} rows")

print("\\n=== COURSERA ===")
print(f"Before: {len(coursera_df)} rows")
coursera_df = coursera_df.rename(columns={'name': 'title', 'content': 'description'})
for col in ['title', 'description', 'category', 'skills', 'what_you_learn']:
    if col not in coursera_df.columns:
        coursera_df[col] = ''
    else:
        coursera_df[col] = coursera_df[col].fillna('')
coursera_df = coursera_df.drop_duplicates(subset=['title']).reset_index(drop=True)
print(f"After dedup: {len(coursera_df)} rows")"""),

    md("## 2.3 Tạo Nhãn Độ khó"),
    code("""def difficulty_from_fog(fog_score):
    \"\"\"Gunning Fog Index -> Easy/Medium/Hard\"\"\"
    try:
        fog = float(fog_score)
        if fog < 8: return 'Easy'
        elif fog <= 12: return 'Medium'
        else: return 'Hard'
    except: return 'Medium'

def difficulty_from_content(category, content=''):
    \"\"\"Keyword-based difficulty from Coursera metadata\"\"\"
    easy_kw = ['beginner', 'introduction', 'intro', 'basic', 'fundamental', 
               'getting started', 'elementary', 'overview', '101']
    hard_kw = ['advanced', 'expert', 'professional', 'deep', 'master',
               'specialization', 'architecture', 'research', 'graduate']
    combined = (str(category) + ' ' + str(content)).lower()
    if any(k in combined for k in hard_kw): return 'Hard'
    elif any(k in combined for k in easy_kw): return 'Easy'
    return 'Medium'

# Assign labels
if 'desc_gunning_fog' in khan_df.columns:
    khan_df['difficulty'] = pd.to_numeric(khan_df['desc_gunning_fog'], errors='coerce').apply(difficulty_from_fog)
else:
    khan_df['difficulty'] = 'Medium'

coursera_df['difficulty'] = coursera_df.apply(
    lambda r: difficulty_from_content(r.get('category', ''), r.get('description', '')), axis=1
)

print("Khan Academy difficulty distribution:")
print(khan_df['difficulty'].value_counts())
print("\\nCoursera difficulty distribution:")
print(coursera_df['difficulty'].value_counts())"""),

    md("## 2.4 Chuẩn hóa Văn bản"),
    code("""lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def normalize_text(text):
    \"\"\"
    Pipeline chuẩn hóa:
    1. Lowercase
    2. Remove URLs
    3. Remove punctuation & digits
    4. Tokenize
    5. Remove stopwords
    6. Lemmatize
    \"\"\"
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove URLs
    text = re.sub(r'https?://\\S+|www\\.\\S+', ' ', text)
    
    # Step 3: Remove punctuation & digits
    text = re.sub(r'[^a-z\\s]', ' ', text)
    
    # Step 4: Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Step 5 & 6: Remove stopwords + Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens 
              if t not in stop_words and len(t) > 2]
    
    return ' '.join(tokens)

# Test
sample = "Introduction to Calculus: Derivatives and Integrals for Beginners!"
print(f"Original: {sample}")
print(f"Cleaned:  {normalize_text(sample)}")"""),

    code("""%%time
# Apply normalization to Khan Academy
print("Normalizing Khan Academy texts...")
khan_df['raw_text'] = khan_df['title'] + ' ' + khan_df['description']
khan_df['clean_text'] = khan_df['raw_text'].apply(normalize_text)

# Apply normalization to Coursera
print("Normalizing Coursera texts...")
coursera_df['raw_text'] = (coursera_df['title'] + ' ' + coursera_df['description'] + 
                            ' ' + coursera_df.get('what_you_learn', '') + 
                            ' ' + coursera_df.get('skills', ''))
coursera_df['clean_text'] = coursera_df['raw_text'].apply(normalize_text)

# Stats
print(f"\\nKhan clean text avg length: {khan_df['clean_text'].str.len().mean():.0f} chars")
print(f"Coursera clean text avg length: {coursera_df['clean_text'].str.len().mean():.0f} chars")"""),

    md("## 2.5 Ghép Dataset"),
    code("""# Add source and ID
khan_df['source'] = 'khan'
khan_df['id'] = 'khan_' + khan_df.index.astype(str)
khan_df['category'] = khan_df['channel_title'].apply(lambda x: str(x).split('|')[0].strip())
khan_df['view_count'] = pd.to_numeric(khan_df.get('view_count', 0), errors='coerce').fillna(0)
khan_df['like_count'] = pd.to_numeric(khan_df.get('like_count', 0), errors='coerce').fillna(0)

coursera_df['source'] = 'coursera'
coursera_df['id'] = 'coursera_' + coursera_df.index.astype(str)
coursera_df['view_count'] = 0
coursera_df['like_count'] = 0

# Merge
select_cols = ['id', 'title', 'raw_text', 'clean_text', 'category', 
               'difficulty', 'source', 'view_count', 'like_count']

unified_df = pd.concat([
    khan_df[[c for c in select_cols if c in khan_df.columns]],
    coursera_df[[c for c in select_cols if c in coursera_df.columns]]
], ignore_index=True)

# Remove empty texts
unified_df = unified_df[unified_df['clean_text'].str.len() > 10].reset_index(drop=True)

# Encode labels
label_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}
unified_df['difficulty_encoded'] = unified_df['difficulty'].map(label_map).fillna(1).astype(int)

print(f"Unified dataset: {unified_df.shape}")
print(f"\\nDifficulty distribution:")
print(unified_df['difficulty'].value_counts())
unified_df.head()"""),

    md("## 2.6 Sinh TF-IDF Vector"),
    code("""from sklearn.feature_extraction.text import TfidfVectorizer

print("Building TF-IDF matrix...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(unified_df['clean_text'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Vocab size: {len(vectorizer.vocabulary_):,}")
print(f"Sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4f}")

# Top terms
feature_names = vectorizer.get_feature_names_out()
print(f"\\nSample features: {feature_names[:20].tolist()}")"""),

    code("""# Save processed data
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../models', exist_ok=True)

unified_df.to_csv('../data/processed/unified_data.csv', index=False)
joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')

from scipy.sparse import save_npz
save_npz('../data/processed/tfidf_matrix.npz', tfidf_matrix)

print("Saved:")
print("  ../data/processed/unified_data.csv")
print("  ../models/tfidf_vectorizer.pkl")
print("  ../data/processed/tfidf_matrix.npz")"""),

    md("## 2.7 Kiểm tra kết quả"),
    code("""# Text length distribution
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
unified_df['text_len'] = unified_df['clean_text'].str.split().str.len()

for i, src in enumerate(['khan', 'coursera']):
    src_data = unified_df[unified_df['source'] == src]['text_len']
    axes[i].hist(src_data, bins=50, color=f'C{i}', alpha=0.8, edgecolor='white')
    axes[i].set_title(f'{src.title()} - Text Word Count')
    axes[i].axvline(src_data.median(), color='red', linestyle='--', 
                    label=f'Median: {src_data.median():.0f}')
    axes[i].legend()

plt.suptitle('Text Length Distribution After Preprocessing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../report/images/nb02_text_length.png', dpi=150, bbox_inches='tight')
plt.show()
print("Preprocessing complete!")"""),
])

with open(os.path.join(NOTEBOOKS_DIR, '02_preprocessing.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb2, f, ensure_ascii=False, indent=2)
print("02_preprocessing.ipynb created")

# ============================================================
# NOTEBOOK 3: EM CLUSTERING
# ============================================================
nb3 = make_nb([
    md("# 03 - Phân cụm EM (Expectation Maximization)\n## EM Clustering with Gaussian Mixture Model\n\nMục tiêu:\n- Giảm chiều TF-IDF bằng TruncatedSVD\n- Thử nhiều giá trị k, đánh giá BIC, AIC, Silhouette\n- Chọn k tối ưu và train GMM\n- Hiển thị top keywords và vẽ PCA/t-SNE"),

    code("""import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

os.makedirs('../report/images', exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')
print("Ready!")"""),

    md("## 3.1 Load Data"),
    code("""# Load TF-IDF matrix và data
tfidf_matrix = load_npz('../data/processed/tfidf_matrix.npz')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
df = pd.read_csv('../data/processed/unified_data.csv')

print(f"TF-IDF matrix: {tfidf_matrix.shape}")
print(f"Dataset: {df.shape}")"""),

    md("## 3.2 Giảm chiều bằng TruncatedSVD (LSA)"),
    code("""# SVD reduction
n_components = 100
print(f"Reducing to {n_components} components...")

svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(tfidf_matrix)
X_reduced = normalize(X_reduced)

explained_var = svd.explained_variance_ratio_.sum()
print(f"Explained variance: {explained_var:.3f} ({explained_var:.1%})")
print(f"Reduced shape: {X_reduced.shape}")

# Plot explained variance
fig, ax = plt.subplots(figsize=(10, 5))
cumsum = np.cumsum(svd.explained_variance_ratio_)
ax.plot(range(1, len(cumsum)+1), cumsum, 'b-', lw=2)
ax.axhline(0.8, color='red', linestyle='--', label='80% threshold')
ax.set_title('Cumulative Explained Variance (SVD)', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.legend()
plt.tight_layout()
plt.savefig('../report/images/nb03_svd_variance.png', dpi=150, bbox_inches='tight')
plt.show()
joblib.dump(svd, '../models/svd_model.pkl')"""),

    md("## 3.3 Đánh giá với nhiều giá trị k"),
    code("""# Test multiple k values
k_values = [3, 5, 7, 8, 10, 12, 15]
results = []

print(f"Testing k = {k_values}...")
print(f"{'k':>4} | {'BIC':>12} | {'AIC':>12} | {'Silhouette':>12}")
print("-"*46)

for k in k_values:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type='diag',
        max_iter=200,
        n_init=3,
        random_state=42
    )
    gmm.fit(X_reduced)
    labels = gmm.predict(X_reduced)
    
    bic = gmm.bic(X_reduced)
    aic = gmm.aic(X_reduced)
    
    # Silhouette trên subset
    if len(X_reduced) > 5000:
        idx = np.random.choice(len(X_reduced), 5000, replace=False)
        sil = silhouette_score(X_reduced[idx], labels[idx])
    else:
        sil = silhouette_score(X_reduced, labels)
    
    results.append({'k': k, 'bic': bic, 'aic': aic, 'silhouette': sil, 'model': gmm})
    print(f"{k:>4} | {bic:>12.1f} | {aic:>12.1f} | {sil:>12.4f}")"""),

    code("""results_df = pd.DataFrame([{k2: v for k2, v in r.items() if k2 != 'model'} for r in results])

# Plot evaluation metrics
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('EM Clustering Evaluation', fontsize=16, fontweight='bold')

axes[0].plot(results_df['k'], results_df['bic'], 'o-', color='#4C72B0', lw=2, ms=8, label='BIC')
axes[0].plot(results_df['k'], results_df['aic'], 's--', color='#DD8452', lw=2, ms=8, label='AIC')
axes[0].set_title('BIC & AIC vs k')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Score'); axes[0].legend()

axes[1].plot(results_df['k'], results_df['silhouette'], 'D-', color='#55A868', lw=2, ms=8)
axes[1].set_title('Silhouette Score vs k')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette')

# Best k highlight
best_idx = results_df['bic'].idxmin()
best_k = int(results_df.loc[best_idx, 'k'])
axes[0].axvline(best_k, color='red', linestyle=':', lw=2, label=f'Best k={best_k}')
axes[1].axvline(best_k, color='red', linestyle=':', lw=2, label=f'Best k={best_k}')
axes[0].legend(); axes[1].legend()

print(f"\\nBest k = {best_k}")
print(f"  BIC = {results_df.loc[best_idx, 'bic']:.1f}")
print(f"  AIC = {results_df.loc[best_idx, 'aic']:.1f}")
print(f"  Silhouette = {results_df.loc[best_idx, 'silhouette']:.4f}")

# Table
axes[2].axis('off')
table_data = results_df[['k', 'bic', 'aic', 'silhouette']].round(2).values.tolist()
table = axes[2].table(
    cellText=table_data,
    colLabels=['k', 'BIC', 'AIC', 'Silhouette'],
    loc='center', cellLoc='center'
)
table.auto_set_font_size(False); table.set_fontsize(10)
axes[2].set_title('Summary Table')

plt.tight_layout()
plt.savefig('../report/images/nb03_em_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("## 3.4 Train Model với k tối ưu"),
    code("""# Train final GMM
best_gmm = results[results_df['bic'].idxmin()]['model']
print(f"Training final GMM with k={best_k}...")

# Predict
labels = best_gmm.predict(X_reduced)
proba  = best_gmm.predict_proba(X_reduced)

# Save
joblib.dump(best_gmm, '../models/gmm_model.pkl')
np.save('../data/processed/cluster_labels.npy', labels)
np.save('../data/processed/X_reduced.npy', X_reduced)
np.save('../data/processed/cluster_proba.npy', proba)

# Summary
unique, counts = np.unique(labels, return_counts=True)
print(f"\\nCluster sizes:")
for c, cnt in zip(unique, counts):
    print(f"  Cluster {c}: {cnt:,} items ({cnt/len(labels):.1%})")"""),

    md("## 3.5 Top Keywords mỗi Cluster"),
    code("""feature_names = np.array(vectorizer.get_feature_names_out())
mat = tfidf_matrix.toarray()

cluster_keywords = {}
print("TOP KEYWORDS PER CLUSTER")
print("="*70)
for c in unique:
    mask = labels == c
    cluster_mean = mat[mask].mean(axis=0)
    top_idx = cluster_mean.argsort()[::-1][:15]
    keywords = feature_names[top_idx].tolist()
    cluster_keywords[c] = keywords
    print(f"Cluster {c:2d} ({counts[c]:5,} items): {', '.join(keywords[:10])}")

import json
with open('../data/processed/cluster_keywords.json', 'w') as f:
    json.dump({str(k): v for k, v in cluster_keywords.items()}, f, indent=2)
print("\\nKeywords saved!")"""),

    md("## 3.6 Biểu đồ PCA 2D"),
    code("""# PCA 2D
print("Running PCA...")
pca = PCA(n_components=2, random_state=42)
n_vis = min(10000, len(X_reduced))
X_pca = pca.fit_transform(X_reduced[:n_vis])
labels_vis = labels[:n_vis]

fig, ax = plt.subplots(figsize=(13, 10))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_vis,
                     cmap='tab20', alpha=0.4, s=8)
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.set_title(f'PCA 2D - EM Clusters (k={best_k})', fontsize=14, fontweight='bold')
ax.set_xlabel(f'PC1 (var={pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 (var={pca.explained_variance_ratio_[1]:.1%})')
plt.tight_layout()
plt.savefig('../report/images/nb03_pca.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("## 3.7 Biểu đồ t-SNE 2D"),
    code("""# t-SNE 2D
n_tsne = min(5000, len(X_reduced))
idx = np.random.choice(len(X_reduced), n_tsne, replace=False)
print(f"Running t-SNE on {n_tsne} samples (this may take a few minutes)...")

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
X_tsne = tsne.fit_transform(X_reduced[idx])
labels_tsne = labels[idx]

fig, ax = plt.subplots(figsize=(13, 10))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_tsne,
                     cmap='tab20', alpha=0.5, s=12)
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.set_title(f't-SNE 2D - EM Clusters (k={best_k})', fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig('../report/images/nb03_tsne.png', dpi=150, bbox_inches='tight')
plt.show()
print("Clustering complete!")"""),
])

with open(os.path.join(NOTEBOOKS_DIR, '03_em_clustering.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb3, f, ensure_ascii=False, indent=2)
print("03_em_clustering.ipynb created")

# ============================================================
# NOTEBOOK 4: NEURAL NETWORK  
# ============================================================
nb4 = make_nb([
    md("# 04 - Phân lớp Neural Network\n## Difficulty Classification with TensorFlow/Keras\n\nMục tiêu:\n- Phân loại độ khó: **Easy / Medium / Hard**\n- Huấn luyện MLP và Deep Dense NN\n- Đánh giá bằng Accuracy, F1, Confusion Matrix"),

    code("""import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from scipy.sparse import load_npz

os.makedirs('../report/images', exist_ok=True)
os.makedirs('../models', exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')
print(f"TensorFlow version: {tf.__version__}")"""),

    md("## 4.1 Load Data"),
    code("""tfidf_matrix = load_npz('../data/processed/tfidf_matrix.npz')
df = pd.read_csv('../data/processed/unified_data.csv')

X = tfidf_matrix.toarray().astype(np.float32)
y = df['difficulty_encoded'].values.astype(np.int32)

label_names = ['Easy', 'Medium', 'Hard']
print(f"X shape: {X.shape}")
print(f"y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")"""),

    md("## 4.2 Train/Val/Test Split"),
    code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)
input_dim = X_train.shape[1]
print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Input dim: {input_dim}")"""),

    md("## 4.3 Mô hình 1: MLP"),
    code("""def build_mlp(input_dim, n_classes=3):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ], name='MLP')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

mlp = build_mlp(input_dim)
mlp.summary()"""),

    code("""callbacks = [
    keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
    keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6)
]

print("Training MLP...")
hist_mlp = mlp.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=256,
    callbacks=callbacks, verbose=1
)"""),

    code("""# Evaluate MLP
y_pred_mlp = np.argmax(mlp.predict(X_test, verbose=0), axis=1)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')
print(f"MLP - Accuracy: {acc_mlp:.4f}, F1: {f1_mlp:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred_mlp, target_names=label_names))
mlp.save('../models/mlp_model.keras')"""),

    md("## 4.4 Mô hình 2: Deep Dense NN"),
    code("""def build_deep_nn(input_dim, n_classes=3):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(), layers.Dropout(0.35),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ], name='DeepDenseNN')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

dnn = build_deep_nn(input_dim)
dnn.summary()"""),

    code("""print("Training Deep Dense NN...")
hist_dnn = dnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=256,
    callbacks=callbacks, verbose=1
)

y_pred_dnn = np.argmax(dnn.predict(X_test, verbose=0), axis=1)
acc_dnn = accuracy_score(y_test, y_pred_dnn)
f1_dnn = f1_score(y_test, y_pred_dnn, average='weighted')
print(f"DeepNN - Accuracy: {acc_dnn:.4f}, F1: {f1_dnn:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred_dnn, target_names=label_names))
dnn.save('../models/deepnn_model.keras')"""),

    md("## 4.5 Training Curves"),
    code("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Neural Network Training History', fontsize=16, fontweight='bold')

# MLP
axes[0,0].plot(hist_mlp.history['loss'], label='Train', color='#4C72B0', lw=2)
axes[0,0].plot(hist_mlp.history['val_loss'], label='Val', color='#DD8452', lw=2, ls='--')
axes[0,0].set_title('MLP - Loss'); axes[0,0].legend()

axes[0,1].plot(hist_mlp.history['accuracy'], label='Train', color='#55A868', lw=2)
axes[0,1].plot(hist_mlp.history['val_accuracy'], label='Val', color='#C44E52', lw=2, ls='--')
axes[0,1].set_title('MLP - Accuracy'); axes[0,1].legend()

# Deep NN
axes[1,0].plot(hist_dnn.history['loss'], label='Train', color='#4C72B0', lw=2)
axes[1,0].plot(hist_dnn.history['val_loss'], label='Val', color='#DD8452', lw=2, ls='--')
axes[1,0].set_title('Deep NN - Loss'); axes[1,0].legend()

axes[1,1].plot(hist_dnn.history['accuracy'], label='Train', color='#55A868', lw=2)
axes[1,1].plot(hist_dnn.history['val_accuracy'], label='Val', color='#C44E52', lw=2, ls='--')
axes[1,1].set_title('Deep NN - Accuracy'); axes[1,1].legend()

for ax in axes.flat: ax.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig('../report/images/nb04_training_history.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("## 4.6 Confusion Matrix"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

for ax, y_pred, name in zip(axes, [y_pred_mlp, y_pred_dnn], ['MLP', 'DeepNN']):
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_title(f'{name}\\nAccuracy={acc:.4f}, F1={f1:.4f}')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')

plt.tight_layout()
plt.savefig('../report/images/nb04_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("## 4.7 Summary"),
    code("""best_model_name = 'MLP' if acc_mlp >= acc_dnn else 'DeepNN'
print(f"\\n{'='*50}")
print(f"MODEL COMPARISON")
print(f"{'='*50}")
print(f"MLP:    Acc={acc_mlp:.4f} | F1={f1_mlp:.4f}")
print(f"DeepNN: Acc={acc_dnn:.4f} | F1={f1_dnn:.4f}")
print(f"\\nBest model: {best_model_name}")

meta = {'best_model': best_model_name, 'input_dim': int(input_dim),
        'label_map': {'Easy': 0, 'Medium': 1, 'Hard': 2},
        'mlp': {'accuracy': float(acc_mlp), 'f1': float(f1_mlp)},
        'deepnn': {'accuracy': float(acc_dnn), 'f1': float(f1_dnn)}}
with open('../models/nn_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("\\nModels saved to ../models/")"""),
])

with open(os.path.join(NOTEBOOKS_DIR, '04_neural_network.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb4, f, ensure_ascii=False, indent=2)
print("04_neural_network.ipynb created")

# ============================================================
# NOTEBOOK 5: ASSOCIATION RULES
# ============================================================
nb5 = make_nb([
    md("# 05 - Luật kết hợp\n## Association Rules with FP-Growth\n\nMục tiêu:\n- Tạo transaction từ skills/tags\n- Áp dụng FP-Growth\n- Tính Support, Confidence, Lift\n- Xuất Top 20 rules tốt nhất"),

    code("""import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast, re, os, joblib, warnings
warnings.filterwarnings('ignore')

from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

os.makedirs('../report/images', exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')
print("Ready!")"""),

    code("""df = pd.read_csv('../data/processed/unified_data.csv')
coursera_raw = pd.read_csv('../courses_en.csv', low_memory=False)

# Merge skills
if 'skills' in coursera_raw.columns:
    coursera_mask = df['source'] == 'coursera'
    n_c = coursera_mask.sum()
    orig = coursera_raw.drop_duplicates(subset=['name'] if 'name' in coursera_raw.columns else coursera_raw.columns[1:2]).head(n_c)
    df.loc[coursera_mask, 'skills'] = orig['skills'].fillna('').values[:n_c]
else:
    df['skills'] = ''

print(f"Dataset: {df.shape}")
print(df['source'].value_counts())"""),

    md("## 5.1 Tạo Transactions"),
    code("""def parse_skills(s):
    if not isinstance(s, str) or not s.strip(): return []
    try:
        items = ast.literal_eval(s)
        if isinstance(items, list):
            return [str(i).strip().lower() for i in items if str(i).strip()]
    except: pass
    return [i.strip().lower() for i in re.split(r'[,;]', s) if i.strip()]

def extract_khan_tags(title, category):
    tags = []
    if isinstance(category, str) and category.strip():
        tags.append(category.strip().lower())
    subjects = ['math', 'algebra', 'calculus', 'chemistry', 'physics', 'biology',
                'history', 'economics', 'programming', 'computer', 'english']
    title_lower = str(title).lower()
    tags.extend([s for s in subjects if s in title_lower])
    return list(set(tags))

# Build transactions
transactions = []
for _, row in df.iterrows():
    items = []
    if row.get('source') == 'coursera':
        items = parse_skills(str(row.get('skills', '')))
    else:
        items = extract_khan_tags(row.get('title',''), row.get('category',''))
    
    diff = str(row.get('difficulty','')).strip().lower()
    if diff in ['easy','medium','hard']:
        items.append(f'difficulty:{diff}')
    
    if len(items) >= 2:
        transactions.append(list(set(items)))

print(f"Total transactions: {len(transactions):,}")
print(f"Sample: {transactions[0][:5] if transactions else []}")"""),

    md("## 5.2 FP-Growth"),
    code("""# Encode
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
te_df = pd.DataFrame(te_array, columns=te.columns_)

print(f"Transaction matrix: {te_df.shape}")
print(f"Unique items: {len(te.columns_)}")

# FP-Growth
freq_itemsets = fpgrowth(te_df, min_support=0.005, use_colnames=True)
print(f"Frequent itemsets: {len(freq_itemsets)}")
freq_itemsets.head(10)"""),

    code("""# Generate rules
rules = association_rules(freq_itemsets, metric='confidence', min_threshold=0.3)
rules = rules[rules['lift'] >= 1.0].copy()
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

print(f"Total rules generated: {len(rules)}")
print(f"\\nTop 5 rules by Lift:")
rules[['antecedents_str','consequents_str','support','confidence','lift']].head(5)"""),

    md("## 5.3 Top 20 Luật tốt nhất"),
    code("""top20 = rules[['antecedents_str','consequents_str','support','confidence','lift']].head(20)
print("\\n=== TOP 20 ASSOCIATION RULES (by Lift) ===")
print(top20.to_string(index=False))

# Save
rules.to_csv('../data/processed/association_rules.csv', index=False)
top20.to_csv('../data/processed/top20_rules.csv', index=False)
print("\\nRules saved!")"""),

    md("## 5.4 Biểu đồ trực quan"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Association Rules Analysis', fontsize=16, fontweight='bold')

# Scatter Support vs Confidence (colored by Lift)
scatter = axes[0].scatter(rules['support'], rules['confidence'],
                           c=rules['lift'], cmap='YlOrRd', alpha=0.6, s=60)
plt.colorbar(scatter, ax=axes[0], label='Lift')
axes[0].set_title('Support vs Confidence (color=Lift)')
axes[0].set_xlabel('Support'); axes[0].set_ylabel('Confidence')

# Top 20 by Lift - bar chart
y_pos = range(len(top20))
rule_labels = [f"{r['antecedents_str'][:18]}→{r['consequents_str'][:12]}" 
               for _, r in top20.iterrows()]
axes[1].barh(list(y_pos), top20['lift'].values, color=plt.cm.RdYlGn(np.linspace(0,1,len(top20))))
axes[1].set_yticks(list(y_pos)); axes[1].set_yticklabels(rule_labels, fontsize=7)
axes[1].set_title('Top 20 Rules by Lift'); axes[1].set_xlabel('Lift')

plt.tight_layout()
plt.savefig('../report/images/nb05_association_rules.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    code("""# Bubble chart
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(top20['support'], top20['confidence'],
                     s=top20['lift']*300, c=top20['lift'],
                     cmap='Reds', alpha=0.7, edgecolors='darkred', lw=1)
plt.colorbar(scatter, ax=ax, label='Lift')
for _, row in top20.head(10).iterrows():
    ax.annotate(f"{row['antecedents_str'][:15]}→{row['consequents_str'][:10]}",
                (row['support'], row['confidence']), fontsize=7, ha='center', va='bottom')
ax.set_title('Association Rules Bubble Chart (size=Lift)', fontsize=14, fontweight='bold')
ax.set_xlabel('Support'); ax.set_ylabel('Confidence')
plt.tight_layout()
plt.savefig('../report/images/nb05_ar_bubble.png', dpi=150, bbox_inches='tight')
plt.show()
print("Association Rules analysis complete!")"""),
])

with open(os.path.join(NOTEBOOKS_DIR, '05_association_rules.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb5, f, ensure_ascii=False, indent=2)
print("05_association_rules.ipynb created")

# ============================================================
# NOTEBOOK 6: RECOMMENDATION SYSTEM
# ============================================================
nb6 = make_nb([
    md("# 06 - Hệ thống Gợi ý Bài tập\n## Complete Recommendation System Demo\n\nMục tiêu:\n- Kết hợp tất cả kết quả: TF-IDF + GMM + NN + AR\n- Demo `recommend_exercises()` với nhiều ví dụ\n- Trực quan hóa pipeline"),

    code("""import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

plt.style.use('seaborn-v0_8-darkgrid')
print("Ready!")"""),

    md("## 6.1 Pipeline Overview"),
    code("""# Visualize pipeline
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis('off')

steps = [
    (0.5, 'Input Video\\nText', '#1565c0'),
    (2.5, 'Text\\nPreprocessing', '#2e7d32'),
    (4.5, 'TF-IDF\\nVectors', '#f57f17'),
    (6.5, 'EM Clustering\\n(GMM)', '#6a1b9a'),
    (8.5, 'Neural Network\\n(Difficulty)', '#c62828'),
    (10.5, 'Association\\nRules', '#00838f'),
    (12.5, 'Final\\nRecommendation', '#37474f')
]

for x, label, color in steps:
    rect = mpatches.FancyBboxPatch((x-0.8, 1.2), 1.6, 1.6,
                                    boxstyle='round,pad=0.1',
                                    facecolor=color, edgecolor='white', alpha=0.9)
    ax.add_patch(rect)
    ax.text(x, 2.0, label, ha='center', va='center', fontsize=8,
            color='white', fontweight='bold')

# Arrows
for i in range(len(steps)-1):
    x1 = steps[i][0] + 0.8
    x2 = steps[i+1][0] - 0.8
    ax.annotate('', xy=(x2, 2.0), xytext=(x1, 2.0),
                arrowprops=dict(arrowstyle='->', color='#90a4ae', lw=2))

ax.set_title('Recommendation System Pipeline', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../report/images/nb06_pipeline.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("## 6.2 Sử dụng recommend_exercises()"),
    code("""from src.recommender import recommend_exercises

# Test 1: Toán học
result1 = recommend_exercises(
    "Introduction to Algebra: Solving Linear Equations",
    top_n=5,
    verbose=True
)"""),

    code("""# Test 2: Hóa học
result2 = recommend_exercises(
    "Organic Chemistry: Reaction Mechanisms and Functional Groups",
    top_n=5,
    verbose=True
)"""),

    code("""# Test 3: Lập trình
result3 = recommend_exercises(
    "Machine Learning with Python: Building Neural Networks from Scratch",
    top_n=5,
    verbose=True
)"""),

    code("""# Test 4: Lịch sử
result4 = recommend_exercises(
    "World War II: Causes, Key Events and Long-term Consequences",
    top_n=5,
    verbose=True
)"""),

    code("""# Test 5: Kinh tế
result5 = recommend_exercises(
    "Microeconomics: Supply and Demand Analysis",
    top_n=5,
    verbose=True
)"""),

    md("## 6.3 Biểu đồ kết quả gợi ý"),
    code("""all_results = [result1, result2, result3, result4, result5]
queries = [
    "Algebra: Linear Equations",
    "Organic Chemistry",
    "Machine Learning / Neural Networks",
    "World War II",
    "Microeconomics"
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Recommendation System Results', fontsize=16, fontweight='bold')

# Topics detected
topics = [r['topic'] for r in all_results]
topic_counts = pd.Series(topics).value_counts()
axes[0].bar(topic_counts.index, topic_counts.values, color=plt.cm.viridis(np.linspace(0,1,len(topic_counts))))
axes[0].set_title('Topics Detected')
axes[0].set_ylabel('Count')

# Difficulties predicted
diffs = [r['difficulty'] for r in all_results]
diff_counts = pd.Series(diffs).value_counts()
colors_d = {'Easy': '#4caf50', 'Medium': '#ff9800', 'Hard': '#f44336'}
axes[1].bar(diff_counts.index, diff_counts.values, 
            color=[colors_d.get(d, '#90a4ae') for d in diff_counts.index])
axes[1].set_title('Difficulties Predicted')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('../report/images/nb06_recommendation_results.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("## 6.4 Export kết quả"),
    code("""# Export sample recommendations
export_rows = []
for q, r in zip(queries, all_results):
    for ex in r['exercises']:
        export_rows.append({
            'query': q,
            'topic': r['topic'],
            'difficulty': r['difficulty'],
            'cluster': r['cluster'],
            'rank': ex['rank'],
            'exercise': ex['exercise'],
            'score': ex['score']
        })

export_df = pd.DataFrame(export_rows)
export_df.to_csv('../data/processed/sample_recommendations.csv', index=False)
print(f"Exported {len(export_df)} recommendations")
export_df.head(15)"""),

    md("## 6.5 Summary"),
    code("""print("="*60)
print("RECOMMENDATION SYSTEM - SUMMARY")
print("="*60)
print(f"\\nPipeline components:")
print(f"  1. TF-IDF Vectorizer: 5000 features, bigrams")
print(f"  2. TruncatedSVD: 100 components")
print(f"  3. GaussianMixture: optimal k from BIC")
print(f"  4. Neural Network: MLP + DeepNN")
print(f"  5. Association Rules: FP-Growth")
print(f"\\nTest results:")
for q, r in zip(queries, all_results):
    n_ex = r['metadata']['n_recommendations']
    print(f"  [{r['topic']:12s}] {r['difficulty']:6s} | Cluster {r['cluster']} | {n_ex} exercises")
print(f"\\nStreamlit demo: streamlit run ../app.py")"""),
])

with open(os.path.join(NOTEBOOKS_DIR, '06_recommendation_system.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb6, f, ensure_ascii=False, indent=2)
print("06_recommendation_system.ipynb created")

print("\\nAll 6 notebooks created successfully!")
